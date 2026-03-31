from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader


class Hot3DClipsDataset(Dataset):
    """
    Args:
        input_dir: Path to directory containing input images.
        ground_truth_dir: Path to segmented / ground truth images.
    """

    def __init__(
        self,
        input_dir: str,
        ground_truth_dir: str,
        transform=None,
    ):
        self.input_dir = Path(input_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.transform = transform

        # Collect file list once (standard practice)
        self.files = sorted(self.input_dir.glob("*"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        input_path = self.files[idx]
        gt_path = self.ground_truth_dir / input_path.name

        # Load images
        image = Image.open(input_path).convert("RGB")
        mask = Image.open(gt_path)

        # Convert to tensor (common default)
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        sample = {
            "image": image,
            "ground_truth_image": mask,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


# ---------------------------------------------------------------------------
# Torch wrapper
# ---------------------------------------------------------------------------

train_dataset = Hot3DClipsDataset(
    input_dir="dataset/train/input",
    ground_truth_dir="dataset/train/gt",
)

val_dataset = Hot3DClipsDataset(
    input_dir="dataset/val/input",
    ground_truth_dir="dataset/val/gt",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test HOT3D-Clips loader")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--ground_truth_dir", type=str, required=True)
    args = parser.parse_args()

    dataset = Hot3DClipsDataset(
        input_dir=args.input_dir,
        ground_truth_dir=args.ground_truth_dir,
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]
    print("Image shape:", sample["image"].shape)
    print("GT shape:", sample["ground_truth_image"].shape)