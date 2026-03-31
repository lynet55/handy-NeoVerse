import os

import torch
from torch.utils.data import DataLoader
# from torchvision.utils import save_image
from datasets import load_dataset

from helpers import preprocess_batch, HOT3D_RAW_COLUMNS


def setup_streaming_dataloader(batch_size=16, shuffle_buffer=1000):
    """
    Load the HOT3D dataset from HuggingFace in streaming mode.

    Args:
        batch_size: samples per batch
        shuffle_buffer: buffer size for stream shuffling (0 to disable)
    """
    print("Initializing HOT3D dataset stream...")

    dataset = load_dataset(
        "bop-benchmark/hot3d",
        name="default",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    if shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    dataset = dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=HOT3D_RAW_COLUMNS,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    return dataloader


if __name__ == "__main__":

    save_dir = "test_samples"
    os.makedirs(save_dir, exist_ok=True)

    dataloader = setup_streaming_dataloader(batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")

    for step, batch in enumerate(dataloader):
        images = batch["pixel_values"]  # (B, 3, 518, 518)
        print(f"\n--- Batch {step} ---")
        print(f"  shape : {images.shape}")
        print(f"  dtype : {images.dtype}")
        print(f"  range : [{images.min():.3f}, {images.max():.3f}]")
        print(f"  tensor[0,:,0,0] : {images[0, :, 0, 0]}")

        # Save each image in the batch (undo ImageNet normalization for viewing)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        for i, img in enumerate(images):
            denorm = img.cpu() * std + mean
            denorm = denorm.clamp(0, 1)
            path = os.path.join(save_dir, f"batch{step}_img{i}.png")
            # save_image(denorm, path)
            print(f"  saved -> {path}")

        if step >= 2:
            print("\nPreview complete.")
            break
