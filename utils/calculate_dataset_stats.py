# calculate_dataset_stats_raw.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

## Configuration
# Define the root directory where your images are located.
# Adjust this path to point to your image folder (e.g., 'images' or 'samples').
IMAGE_ROOT_DIR = './Traces_NOA_structured/images' 
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
MAX_IMAGES_TO_SAMPLE = None  # Set to a number (e.g., 5000) for testing, or None for entire dataset

def get_global_pixel_stats_from_raw(image_root_dir, max_samples=None):
    """
    Calculates the global per-channel pixel mean and standard deviation
    by iterating over all raw images in the specified directory structure.
    """
    if not os.path.isdir(image_root_dir):
        print(f"Error: Image root directory not found at {image_root_dir}")
        return None, None

    all_image_paths = []
    
    # Recursively find all image files under the root directory
    for root, _, files in os.walk(image_root_dir):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                all_image_paths.append(os.path.join(root, file))

    if not all_image_paths:
        print("No images found in the specified directory structure.")
        return None, None

    if max_samples is not None:
        all_image_paths = all_image_paths[:max_samples]

    total_pixel_count = 0
    channel_sum = np.zeros(3)
    channel_sum_sq = np.zeros(3)
    
    print(f"Processing {len(all_image_paths)} images to calculate statistics...")

    for img_path in tqdm(all_image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            img_arr = np.array(img).astype(np.float64)  # Use float64 for precision
            
            if img_arr.ndim != 3 or img_arr.shape[2] < 3:
                continue

            pixels = img_arr.reshape(-1, 3)
            num_pixels = pixels.shape[0]
            
            # Sum and Sum of Squares calculation
            channel_sum += pixels.sum(axis=0)
            channel_sum_sq += (pixels ** 2).sum(axis=0)
            total_pixel_count += num_pixels

        except Exception:
            continue # Skip corrupted or unreadable files

    if total_pixel_count == 0:
        print("No valid pixels were processed.")
        return None, None

    # Calculate Mean
    global_mean = channel_sum / total_pixel_count

    # Calculate Standard Deviation: std = sqrt( (sum_sq / N) - mean^2 )
    mean_of_squares = channel_sum_sq / total_pixel_count
    global_std = np.sqrt(mean_of_squares - (global_mean ** 2))
    
    global_std = np.maximum(global_std, 1e-6) # Ensure stability

    return global_mean, global_std

if __name__ == '__main__':
    print("--- DetAny3D Raw Dataset Statistics Calculator ---")
    
    # 1. Calculate Stats
    mean, std = get_global_pixel_stats_from_raw(IMAGE_ROOT_DIR, MAX_IMAGES_TO_SAMPLE)
    
    # 2. Output Results
    if mean is not None:
        print("\n✅ Calculation Complete.")
        print("-" * 30)
        print("Global Pixel Mean (R, G, B) for Preprocessing:")
        print(f"List: {[round(m, 3) for m in mean]}")
        print("-" * 30)
        print("Global Pixel Std (R, G, B) for Preprocessing:")
        print(f"List: {[round(s, 3) for s in std]}")
        print("-" * 30)
        print("\nThese values should be used to update `cfg.dataset.pixel_mean` and `cfg.dataset.pixel_std` in your DetAny3D configuration file before training.")
    else:
        print("Failed to calculate statistics.")