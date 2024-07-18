# HaMeR: Hand Mesh Recovery

## Installation

```bash
conda create -p ./env -f environment.yml
conda activate ./env 
```

## Image Captioning 
You need to modify absolute data path in `gen.sh` before running.
With llava-1.5-7b model
```bash
CUDA_VISIBLE_DEVICES=7 python3 gen.py --image_folder= \
                                    --batch_size= \
                                    --output_folder=output/
```

