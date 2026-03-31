import json
import os
from glob import glob

import numpy as np
import torch
import webdataset as wds
from torchvision import transforms
from torchvision.transforms import functional as F

# from diffsynth.utils.auxiliary import load_video

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# HOT3D dataset columns:
#   - image_214-1.jpg   : RGB camera (1408x1408)
#   - image_1201-1.jpg  : SLAM left camera (640x480, fisheye)
#   - image_1201-2.jpg  : SLAM right camera (640x480, fisheye)
#   - cameras.json      : per-camera calibration + world-from-camera poses
#   - hand_crops.json   : crop camera transforms for left/right hands per camera
#   - info.json         : device, participant_id, sequence_id, timestamps
#   - __key__, __url__  : sample identifier and source shard URL

# Image transforms for each camera type
rgb_transforms = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# slam_transforms = transforms.Compose([
#     transforms.Resize((518, 518)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# Raw columns to drop after preprocessing
HOT3D_RAW_COLUMNS = [
    "image_214-1.jpg", "image_1201-1.jpg", "image_1201-2.jpg",
    "cameras.json", "hand_crops.json", "info.json",
    "__key__", "__url__"
]

def preprocess_batch(batch):
    """
    Preprocess a streamed batch from bop-benchmark/hot3d.
    Transforms the RGB image and extracts camera + hand metadata.
    """
    rgb_images = batch["image_214-1.jpg"]

    # Transform images
    batch["pixel_values"] = [rgb_transforms(img.convert("RGB")) for img in rgb_images]

    #TODO MANO -> Image segment
    # process_training_data(batch)

    return batch


# def pull_hot3d():

#     urls = [
#         f"https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/main/train_quest3/clip-{i:06d}.tar"
#         for i in range(3)
#     ]

#     dataset = wds.WebDataset(urls, shardshuffle=False).shuffle(10)

#     q = 1
#     for i, sample in enumerate(dataset):
#         # print(sample["hands.json"])
#         # print(sample["cameras.json"])
#         print(sample.keys())

#         if q == 1:
#             break


if __name__ == "__main__":
    pull_hot3d()
