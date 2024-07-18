from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch
from PIL import PngImagePlugin
import jsonlines
import json
import os
import requests
from tqdm import tqdm
from argparse import ArgumentParser
import datasets
from datasets import load_dataset
from PIL import Image
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--quantize", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    dataset = load_dataset("imagefolder", data_dir="/home/ubuntu/sdxl300", split=datasets.Split.TRAIN)
    model_id = "google/paligemma-3b-mix-224"
    cache_dir = "../.cache/huggingface/hub/"
    token = ""
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, token=token)

    if args.quantize == 0:
        bnb_config = None
    elif args.quantize == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif args.quantize == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"":0},
        # attn_implementation="flash_attention_2",
        cache_dir=cache_dir,
        token=token
    ).eval()
    end_index = len(dataset) if args.end_index == -1 else args.end_index
    f = jsonlines.open(f'output/captions_{args.start_index}_{end_index}.jsonl', 'w')
    error_indices = []
    prompts = ["generate caption for this image with stype of DiffusionDB and LAION dataset"]
    for i in tqdm(range(args.start_index, end_index, args.batch_size), total=len(range(args.start_index, end_index, args.batch_size))):
        # try:
        batch = dataset[i: i + args.batch_size] if i + args.batch_size < end_index else dataset[i: end_index]
        n = len(batch["image"])
        print(n, len(batch["image"]))
        inputs = processor(prompts * n, images=batch["image"], padding=True, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=128)
        generated_text = processor.batch_decode(output, skip_special_tokens=True)
        for j in range(n):
            img_index = i + j
            text = generated_text[j]
            f.write({"image_index": img_index, "img_path": batch["path"][j], "caption": text})
        # except:
        #     error_indices.extend(list(range(i, i + args.batch_size)))

        #     with open(f"output/error_indices/error_indices_{args.start_index}_{end_index}.json", 'w', encoding="utf-8") as f:
        #         json.dump(error_indices, f, ensure_ascii=False, indent=4)

        f.close()

