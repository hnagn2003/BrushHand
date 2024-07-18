from argparse import ArgumentParser
import datasets
from PIL import Image
import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
import jsonlines
import os

class MyDataset(datasets.Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = [file for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff'))]

    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, index):
        image_names = self.image_files[index]
        image_paths = [os.path.join(self.folder_path, image_name) for image_name in image_names]
        images = [Image.open(image_path) for image_path in image_paths]
        res = {
            'image' : images,
            'image_name' : image_names
        }
        return res

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--quantize", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_folder", type=str, default="")
    
    args = parser.parse_args()
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    batch_size = args.batch_size
    start_index = args.start_index
    end_index = args.end_index
    quantize = args.quantize
    dataset = MyDataset(args.image_folder)
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
    )
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    end_index = len(dataset) if args.end_index == -1 else args.end_index
    f = jsonlines.open(f'{output_folder}/captions_{args.start_index}_{end_index}.jsonl', 'w')

    prompt = "USER: <image>\nCaptioning for this image with style LAION DiffusionDB\nASSISTANT:"
    model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

    for i in tqdm(range(args.start_index, end_index, args.batch_size), total=len(range(args.start_index, end_index, args.batch_size))):
        batch = dataset[i: i + args.batch_size] if i + args.batch_size < end_index else dataset[i: end_index]
        n = len(batch["image"])
        inputs = processor([prompt]*n, batch["image"], padding=True, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=80)
        generated_text = processor.batch_decode(output, skip_special_tokens=True)
        for j in range(n):
            img_index = i + j
            text = generated_text[j].split("ASSISTANT: ")[-1]
            f.write({"img_path": batch["image_name"][j], "caption": text})
    f.close()
