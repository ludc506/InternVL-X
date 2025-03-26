# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers.activations import ACT2FN
from timm.models.layers import DropPath

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel

logger = logging.get_logger(__name__)

torch.set_printoptions(threshold=float('inf'))

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


def pixel_shuffle(x, scale_factor=0.5, ps_version='v2'):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))
    if ps_version == 'v1':
        warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                        'which results in a transposed image.')
    else:
        x = x.permute(0, 2, 1, 3).contiguous()
    return x


def func_aggregation(x, image_ratio, h, w):
    x = x.reshape(image_ratio[0] * image_ratio[1], h, w, -1)
    x = x.transpose(1, 2) 
    x = x.reshape(image_ratio[0], image_ratio[1] * w, h, x.shape[-1])
    x = x.transpose(1, 2) 
    x = x.reshape(1, image_ratio[0] * h, image_ratio[1] * w, x.shape[-1])
    return x


def func_transform(x, block_height, block_width):
    b = x.shape[0]
    C = x.shape[-1]

    num_blocks_height = x.shape[1] // block_height 
    num_blocks_width = x.shape[2] // block_width
    
    x = x.reshape(b, num_blocks_height, block_height, num_blocks_width, block_width, C)
    x = x.transpose(3, 2)
    x = x.reshape(-1, block_height, block_width, C)
    x = x.view(-1, block_height * block_width, C)
    return x


def func_padding(x, max_length=4):
    current_length = x.shape[1]
    C = x.shape[-1]
    
    if current_length < max_length:
        padding_length = max_length - current_length
        padded_tensor = torch.cat([x, torch.zeros([256, padding_length, C], dtype=x.dtype, device=x.device)], dim=1)
    else:
        padded_tensor = x

    attention_ones = torch.ones([256, 1, current_length], dtype=x.dtype, device=x.device)
    attention_zeros = torch.zeros([256, 1, max_length - current_length], dtype=x.dtype, device=x.device)
    attention_mask = torch.cat([attention_ones, attention_zeros], dim=2)
    attention_mask = attention_mask.to(dtype=torch.bool)

    return padded_tensor, attention_mask


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        InternRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class InternAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.norm1 = InternRMSNorm(self.embed_dim)
        self.norm2 = InternRMSNorm(self.embed_dim)

    def _naive_attn(self, q, kv, mask=None):
        q = self.norm1(q)
        k = v = self.norm2(kv)

        B, N_q, C = q.shape
        N_kv = kv.shape[1]
        
        q = self.q(q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = ((q * self.scale) @ k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        return x

    def forward(self, 
                hidden_states_q: torch.Tensor, 
                hidden_states_kv: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:

        x = self._naive_attn(hidden_states_q, hidden_states_kv, attention_mask)  
        return x


class InternMLP(nn.Module):
    def __init__(self, embed_dim, act):
        super().__init__()
        self.act = ACT2FN[act]
        self.w1 = nn.Linear(embed_dim, 4 * embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, 4 * embed_dim, bias=False)
        self.w2 = nn.Linear(4 * embed_dim, embed_dim, bias=False)
        self.norm = InternRMSNorm(embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        hidden_states = self.norm(hidden_states)
        hidden_states = self.w2(self.act(self.w1(hidden_states)) * self.w3(hidden_states))

        return hidden_states


class InternEncoderLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = 16
        self.act = 'silu'
        self.drop_path_rate = 0.1

        self.attn = InternAttention(self.embed_dim, self.num_heads)
        self.mlp = InternMLP(self.embed_dim, self.act)

        self.drop_path1 = DropPath(self.drop_path_rate) if self.drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(self.drop_path_rate) if self.drop_path_rate > 0. else nn.Identity()


    def forward(
            self,
            hidden_states_q: torch.Tensor,
            hidden_states_kv: torch.Tensor,
            attn_mask: torch.Tensor = None
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        hidden_states = hidden_states_q + self.drop_path1(self.attn(hidden_states_q, hidden_states_kv, attn_mask))
        hidden_states = hidden_states + self.drop_path2(self.mlp(hidden_states))

        return hidden_states


class VisionProjector(nn.Module):
    def __init__(self, vit_hidden_size, llm_hidden_size, downsample_ratio, ps_version, num_image_token):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.ps_version = ps_version

        self.mlp1 = nn.Sequential(
            InternRMSNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size, bias=False),
            nn.SiLU()
        )

        self.mlp2 = nn.Sequential(
            InternRMSNorm(vit_hidden_size),
            nn.Linear(vit_hidden_size, llm_hidden_size, bias=False),
            nn.SiLU()
        )

        self.mlp3 = nn.Sequential(
            InternRMSNorm(vit_hidden_size),
            nn.Linear(vit_hidden_size, llm_hidden_size, bias=False),
            nn.SiLU()
        )

        self.cls_scale = nn.Parameter(torch.randn([1, int(num_image_token ** 0.5), int(num_image_token ** 0.5), llm_hidden_size]))

        self.attn_global = InternEncoderLayer(llm_hidden_size)
        self.attn_local = InternEncoderLayer(llm_hidden_size)


    def forward(self, vit_embeds):
        cls_embds = vit_embeds[:, 0, :]
        vit_embeds = vit_embeds[:, 1:, :]

        b = vit_embeds.shape[0]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(b, h, w, -1)

        vit_embeds_q = pixel_shuffle(vit_embeds, self.downsample_ratio, self.ps_version)
        vit_embeds_q = self.mlp1(vit_embeds_q)
        vit_embeds_q = func_transform(vit_embeds_q, 1, 1)

        vit_embeds_cls = self.mlp2(cls_embds)
        vit_embeds_cls = vit_embeds_cls.reshape(b, 1, 1, -1).expand(-1, int(self.downsample_ratio * h), int(self.downsample_ratio * w), -1)
        cls_scale = self.cls_scale.expand(b, -1, -1, -1)
        vit_embeds_cls = vit_embeds_cls * cls_scale
        vit_embeds_cls = func_transform(vit_embeds_cls, 1, 1)

        vit_embeds_kv = self.mlp3(vit_embeds)
        vit_embeds_kv = func_transform(vit_embeds_kv, int(1 / self.downsample_ratio), int(1 / self.downsample_ratio))

        vit_embeds_q = self.attn_local(vit_embeds_q, vit_embeds_kv)

        vit_embeds_cls = self.attn_global(vit_embeds_cls, vit_embeds_kv)

        vit_embeds = vit_embeds_q + vit_embeds_cls
        vit_embeds = vit_embeds.reshape(b, int(self.downsample_ratio * h), int(self.downsample_ratio * w), -1)

        return vit_embeds


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.projector = VisionProjector(vit_hidden_size, llm_hidden_size, self.downsample_ratio, self.ps_version, self.num_image_token)

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]

        vit_embeds = self.projector(vit_embeds)

        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
