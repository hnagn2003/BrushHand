export LD_LIBRARY_PATH=/home/ubuntu/llava/env/lib/python3.10/site-packages/torch/lib/:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=7 python3 gen.py --image_folder=../halpe/hico_20160224_det/images/train2015 \
                                    --batch_size=32 \
                                    --output_folder=output/halpe_hico
