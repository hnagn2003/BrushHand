export EGL_DEVICE_ID=0 

python3 test_demo.py \
    --data_type hico \
    --batch_size=32 --side_view --full_frame --idx_part 10 --save_mesh
    # --img_folder /lustre/scratch/client/vinai/users/ngannh9/diffuser/output/sdxl \
