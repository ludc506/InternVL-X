# InternVL-X

### Documents

- Get Started

  - Installation: [\[Environment\]](https://internvl.readthedocs.io/en/latest/get_started/installation.html)  [\[requirements.txt\]](./requirements.txt)
  - Evaluation Data Preparation: [\[InternVL Evaluation\]](https://internvl.readthedocs.io/en/latest/get_started/eval_data_preparation.html)
  - Chat Data Format: [\[Meta File\]](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#meta-file)  [\[Pure Text\]](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#pure-text-data)  [\[Single-Image\]](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#single-image-data)

### HF weight

   [**InternVL-X-2B**](https://huggingface.co/LLCC506/InternVL-X-2B) | 
   [**InternVL-X-2B-HD**](https://huggingface.co/LLCC506/InternVL-X-2B-HD) | 
   [**InternVL-X-8B**](https://huggingface.co/LLCC506/InternVL-X-8B) | 
   [**InternVL-X-8B-HD**](https://huggingface.co/LLCC506/InternVL-X-8B-HD)

### Inference
```
import numpy as np
import time
import math
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    block_h = math.ceil(orig_height / image_size) 
    block_w = math.ceil(orig_width / image_size)
    max_num_new = block_h * block_w
    if max_num_new > max_num:
        max_num_new = max_num
    max_num = max_num_new

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


path = 'InternVL-X-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attention_2=False,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=1024, do_sample=False)

# For InternVL-X-2B and InternVL-X-8B
pixel_values = load_image('examples/image1.jpg', max_num=1).to(torch.bfloat16).cuda()

# For InternVL-X-2B-HD and InternVL-X-8B-HD
# pixel_values = load_image('examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()

# single-image single-round conversation (单图单轮对话)
question = '<image>\nDescribe this image in datail'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')

# single-image multi-round conversation (单图多轮对话)
question = '<image>\nPlease describe the image in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Please write a story according to the image.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')
```

### Finetune

- Prepare your data following the Chat Data Format above
- Replace the model_name_or_path and meta_path in the shell
- Finetune, for example:
- ```bash shell/internvlx/2nd_finetune/internvlx_2b_internlm2_1_8b_2nd_finetune.sh```

### Evaluation

We evaluate TextVQA, DocVQA, ChartQA, OKVQA, InfoVQA, GQA, VQAv2, VizWiz, MMB, MME, MMVet, MMMU, POPE, SEED and AI2D within the InternVL-X repository.

- Prepare the evaluation data, please follow the [Eval data prepare guide](https://internvl.readthedocs.io/en/latest/get_started/eval_data_preparation.html)
- Put model in InternVL-X/internvl_chat
- For performing the evaluation, refer to the [InternVL2 eval guide](https://internvl.readthedocs.io/en/latest/internvl2.0/evaluation.html). For example:
    - Run the following command for evaluation:
    - ```bash evaluate.sh InternVL-X-2B vqa-textvqa-val```
    - For dynamic evaluation, use:
    - ```bash evaluate.sh InternVL-X-2B-HD vqa-textvqa-val --dynamic --max-num 6```

Additionally, we assess datasets such as ScienceQA, HallusionBench, ‌MathVista, LLaVABench and RealWorldQA using the VLMEvalKit.

### Acknowledgement

InternVL-X is built with reference to the code of the following projects: [InternVL](https://github.com/OpenGVLab/InternVL), [OpenAI CLIP](https://github.com/openai/CLIP), [Open CLIP](https://github.com/mlfoundations/open_clip), [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark), [EVA](https://github.com/baaivision/EVA/tree/master), [InternImage](https://github.com/OpenGVLab/InternImage), [ViT-Adapter](https://github.com/czczup/ViT-Adapter), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Transformers](https://github.com/huggingface/transformers), [DINOv2](https://github.com/facebookresearch/dinov2), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [Qwen-VL](https://github.com/QwenLM/Qwen-VL/tree/master/eval_mm), and [LLaVA-1.5](https://github.com/haotian-liu/LLaVA). Thanks for their awesome work!

