import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_mask(datadir):
    sky_mask_dir = os.path.join(datadir, "sky_mask")
    dynamic_mask_dir = os.path.join(datadir, "dynamic_mask")
    test_mask_dir = os.path.join(datadir, "merge_mask")
    os.makedirs(test_mask_dir, exist_ok=True)
    sky_mask_filenames_all = sorted(glob(os.path.join(sky_mask_dir, "*.jpg")))
    for filename in tqdm(sky_mask_filenames_all):
        image_basename = os.path.basename(filename)
        test_filename = os.path.join(test_mask_dir, image_basename)
        dynamic_mask_filename = os.path.join(dynamic_mask_dir, image_basename.replace(".jpg", ".png"))
        sky_mask = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        dynamic_mask = cv2.imread(str(dynamic_mask_filename), cv2.IMREAD_GRAYSCALE)

        # Move masks to GPU
        sky_mask = torch.tensor(sky_mask, device="cuda")
        dynamic_mask = torch.tensor(dynamic_mask, device="cuda")

        mask = torch.logical_or(sky_mask == 255, dynamic_mask == 255).to(torch.uint8)
        mask = torch.where(mask, torch.tensor(0, device="cuda"), torch.tensor(1, device="cuda")).to(torch.uint8)

        # Move mask back to CPU for saving
        mask = mask.cpu().numpy()
        Image.fromarray(mask * 255).save(test_filename)
        # mask = mask.transpose(0, 1)
        # mask = torch.from_numpy(mask)
        # masks[cam].append(mask)


def parallel_load_masks(root_dir, num_threads):
    subdirs = [
        os.path.join(root_dir, subdir)
        for subdir in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, subdir))
    ]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(load_mask, subdirs)


# Example usage:
parallel_load_masks("/mnt/ssd4t/occgs/waymo_val", 20)
