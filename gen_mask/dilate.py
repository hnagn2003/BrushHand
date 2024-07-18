import os
from tqdm import tqdm
import argparse
import cv2
import numpy as np
def gau_rand(mean, std_dev, size, low, high):
    # Generate Gaussian distributed random numbers
    random_numbers = np.random.normal(loc=mean, scale=std_dev, size=size)
    
    # Scale and shift the numbers to the desired range (1 to 10)
    scaled_numbers = np.clip(random_numbers, low, high)
    
    return scaled_numbers
def dilate_mask(image, dilation_percentage):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Calculate the kernel size for dilation
    h, w = binary_mask.shape
    kernel_size = int(min(h, w) * dilation_percentage)
    kernel_size = max(1, kernel_size)  # Ensure the kernel size is at least 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Dilate the mask
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    
    return dilated_mask

def process_masks(folder_path, output_folder, dis_per_up, dis_per_low):
    # List of image files in the folder
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in tqdm(files):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)

        if img is not None:
            dilation_percentage = gau_rand(0.03, 0.005, 1, dis_per_low, dis_per_up)
            dilated_mask = dilate_mask(img, dilation_percentage)
            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, dilated_mask)
        else:
            print(f"Failed to read: {img_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random numbers between 0.02 and 0.04 with a Gaussian distribution.")
    parser.add_argument('--input', type=str, default="", help="Lower bound of the range")
    parser.add_argument('--output', type=str, default="", help="Upper bound of the range")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    args = parser.parse_args()

    process_masks(args.input, args.output, 0.04, 0.02)
    print("Mask dilation process completed.")
