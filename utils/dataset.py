
from torch.utils.data import ConcatDataset
import os
import os.path
import random
import json
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageOps
import json
import cv2
import datasets
import jsonlines
DATASET_DIR = {
    'COCO': {
        'image_folder': "data/COCO/train2017",
        'caption_file': "data/COCO/coco_wholebody_train_v1.0.json",
        'mask_folder': "data/COCO/coco_mask"
    },
    'hico': {
        'image_folder': "data/halpe/hico_20160224_det/images/train2015",
        # 'segmentation_dir': "/lustre/scratch/client/vinai/users/quangnqv/dataset/genai_sb_handfacebody/dataset/DeepFashion_multimodal/segm",
        'caption_file': "data/halpe/hico_20160224_det/halpe_hico_caption.jsonl",
        'mask_folder': "data/hico/halpe_mask"
    },
    '13k': {
        'image_folder': "data/hand13k/training_dataset/training_data/images",
        'caption_file': "data/hand13k/captions_4069_sorted.jsonl",
        'mask_folder': "data/hand13k/13khands_mask"
    }
}
def center_crop(image, crop_size):
    new_width, new_height = crop_size
    width, height = image.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return image.crop((left, top, right, bottom))

#Note that Hico and 13k has the same format
class HicoDataset(datasets.Dataset):
    def __init__(self, image_folder, caption_file, mask_path, image_size, transform):
        self.image_folder = image_folder
        self.caption_file = caption_file
        self.mask_path = mask_path
        self.transform = transform
        self.image_size = image_size
        self.image_files = []
        self.image_captions = []
        with jsonlines.open(caption_file) as reader:
            for line in reader:
                img_path = line['img_path']
                caption = line['caption'].lower()
                self.image_files.append(img_path)
                self.image_captions.append(caption)

    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_folder, image_name)
        mask_path = os.path.join(self.mask_path, image_name)
        caption = self.image_captions[index]
        image = Image.open(image_path).convert('RGB')
        background = Image.new('RGB', image.size, (0, 0, 0))
        if not os.path.exists(mask_path):
            mask = Image.new('L', (image.size))
            # print(image_path + "dont have mask\n")
        else:         
            mask = Image.open(mask_path).convert('L')
            if mask.size != image.size:
                print(image_path)
                mask = Image.new('L', (image.size))
        masked_image = Image.composite(image, background, mask)

        res = {
            'image' : image,
            'img_name' : image_path,
            'mask': mask,
            'text': caption,
            'masked_image' : masked_image
        }
        return self.transform(res)
    
class CocoDataset(data.Dataset):
    def __init__(self, img_dir, label_dir, mask_path, image_size, transform):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.mask_path = mask_path

        self.infor = self.preprocess()
        self.imgid = list(self.infor.keys())
        self.image_size = image_size
        self.transform = transform
     
    def preprocess(self):
        f = json.load(open(self.label_dir, 'rb'))
        return f
    
    def __len__(self):
        return len(self.infor)
    def __getitem__(self, index):
        """ 
            bbox in here is xywh
        """
        imgid = self.imgid[index]
        infor = self.infor[imgid]
        img_path = os.path.join(self.img_dir, infor['file_name'])
        caption = infor['caption']
      
        image = Image.open(img_path).convert('RGB')
        mask_path = os.path.join(self.mask_path, infor['file_name'])
        if not os.path.exists(mask_path):
            mask = Image.new('L', (image.size))
        else:         
            mask = Image.open(mask_path).convert('L')
            if mask.size != image.size:
                mask = Image.new('L', (image.size))
        background = Image.new('RGB', image.size, (0, 0, 0))
        
        masked_image = Image.composite(image, background, mask)
        return_dict = {
            'image': image,
            'text': caption,
            'img_name': img_path,
            'mask': mask,
            'masked_image' : masked_image

        }
        return self.transform(return_dict)

def tokenize_captions(self, caption, tokenizer, is_train=True):
    if random.random() < 0:
        caption=""
    elif isinstance(caption, str):
        caption=caption
    elif isinstance(caption, (list, np.ndarray)):
        # take a random caption if there are multiple
        caption=random.choice(caption) if is_train else caption[0]
    else:
        raise ValueError(
            f"Caption column `{caption}` should contain either strings or lists of strings."
        )
    inputs = tokenizer(
        caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

def get_dataset(dataset_names, transform, image_size):
    dataset_names = dataset_names.split(',')
    dataset_list = []
    for dataset_name in dataset_names:
        if dataset_name == 'COCO':
            _data =  CocoDataset(
                img_dir=DATASET_DIR['COCO']['image_folder'],
                label_dir=DATASET_DIR['COCO']['caption_file'],
                mask_path=DATASET_DIR['COCO']['mask_folder'],
                transform=transform,
                image_size=image_size
            )
            print(f"COCO: {len(_data)}")
        elif dataset_name == 'hico':
            _data =  HicoDataset(
                image_folder=DATASET_DIR['hico']['image_folder'],
                caption_file=DATASET_DIR['hico']['caption_file'],
                mask_path=DATASET_DIR['hico']['mask_folder'],
                transform=transform,
                image_size=image_size
        )
            print(f"Hico: {len(_data)}")

        elif dataset_name == '13k':
            _data =  HicoDataset(
                image_folder=DATASET_DIR['13k']['image_folder'],
                caption_file=DATASET_DIR['13k']['caption_file'],
                mask_path=DATASET_DIR['13k']['mask_folder'],
                transform=transform,
                image_size=image_size
        )
            print(f"13k: {len(_data)}")

        else:
            raise ValueError(f'Unknown dataset name: {dataset_name}')

        dataset_list.append(_data)
    if len(dataset_list) > 1: 
        print("using ", dataset_names)
        return ConcatDataset(dataset_list)
    else:
        return dataset_list[0]

#usage
'''
def preprocess_train(examples):
    images = examples['image']
    masks = examples['mask']
    examples["pixel_values"] = torch.stack([train_transforms(image) for image in images]).squeeze()
    examples["mask"] = torch.stack([mask_transforms(mask) for mask in masks]).squeeze()
    return examples
train_transforms = transforms.Compose(
    [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
mask_transforms = transforms.Compose(
    [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ]
)
def preprocess_train(examples):
    examples["pixel_values"] = train_transforms(examples["image"])
    examples["mask"] = mask_transforms(examples["mask"])
    examples["conditioning_pixel_values"] = train_transforms(examples["masked_image"])
    return examples
train_dataset = get_dataset(
    dataset_names=args.train_data_type,
    transform=preprocess_train,
    image_size=resolution
)
def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
        conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
        masks = torch.stack([example["mask"] for example in examples])
        masks = masks.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([tokenize_captions(example['text'])[0] for example in examples])

        # input_ids = torch.stack(input_ids)
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "masks":masks,
            "image_name":[example["img_name"] for example in examples],
            "input_ids": input_ids,
            }
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=8
    )
    train_dataloader_len=len(train_dataset)//args.train_batch_size

'''
def main():
    resolution = 512
    def preprocess_train(examples):
        images = examples['image']
        masks = examples['mask']
        examples["pixel_values"] = torch.stack([train_transforms(image) for image in images]).squeeze()
        # examples["input_ids"] = tokenize_captions(examples['text'])
        examples["mask"] = torch.stack([mask_transforms(mask) for mask in masks]).squeeze()
        return examples
    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    mask_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
    )
    def preprocess_train(examples):
        # examples["pixel_values"] = [train_transforms(image) for image in examples['image']]
        # examples["mask"] = [mask_transforms(mask) for mask in examples['mask']]
        # examples["conditioning_pixel_values"] = [examples["pixel_values"][i]*mask for i, mask in enumerate(examples['mask'])]
        examples["pixel_values"] = train_transforms(examples["image"])
        examples["mask"] = mask_transforms(examples["mask"])
        examples["conditioning_pixel_values"] = train_transforms(examples["masked_image"])
        return examples
    train_dataset = get_dataset(
        dataset_names='COCO,hico,13k',
        transform=preprocess_train,
        image_size=resolution
    )
    print("len dataset: ", len(train_dataset))
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
        conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
        masks = torch.stack([example["mask"] for example in examples])
        masks = masks.to(memory_format=torch.contiguous_format).long()

        # input_ids = torch.stack(input_ids)
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "masks":masks,
            "image_name":[example["img_name"] for example in examples],
            # "input_ids": input_ids,
            }
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=128,
        num_workers=8
    )
    from tqdm import tqdm
    for batch in tqdm(train_dataloader):
        pass

if __name__ == "__main__":
    main()
