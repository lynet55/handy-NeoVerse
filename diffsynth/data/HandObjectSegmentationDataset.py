from pathlib import Path
from PIL import Image
import torch
import numpy as np
from torch.utils.data import DataLoader

def load_image(path: Path) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

def load_binary_mask(path: Path, shape) -> torch.Tensor:
    H, W = shape
    if path is None or not path.exists():
        return torch.zeros((H,W), dtype=torch.bool)
    #mask = np.array(Image.open(path).convert("L"))
    mask = Image.open(path).convert("L")
    if mask.size != (W, H):
        mask = mask.resize((W, H), resample=Image.NEAREST)
    mask = np.array(mask) > 0
    return torch.from_numpy(mask)

class HandObjectSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, mask_root, stream="stream1201-1"):
        self.image_root = Path(image_root) / stream
        self.mask_root = Path(mask_root)
        self.stream = stream

        self.samples = []
        self._build_index()

    def _build_index(self):
        """
        Builds a list of (image_path, mask_paths) entries.
        """
        for clip_dir in sorted(self.image_root.iterdir()):
            if not clip_dir.is_dir():
                continue

            clip_name  = clip_dir.name
            mask_clip_dir = self.mask_root / clip_name

            if not mask_clip_dir.exists():
                continue

            # index masks by frame
            masks_by_frame = {}

        
            for p in mask_clip_dir.glob(f"*_{self.stream}_*.png"):
                frame_id = p.name.split("_")[0]

                masks_by_frame.setdefault(frame_id, {})
                if "hand_LEFT" in p.name:
                    masks_by_frame[frame_id]["left"] = p
                elif "hand_RIGHT" in p.name:
                    masks_by_frame[frame_id]["right"] = p
                elif "object" in p.name:
                    masks_by_frame[frame_id]["object"] = p

            # match images to masks
            for img_path in sorted(clip_dir.glob("*.png")):
                frame_id = img_path.stem
                if frame_id not in masks_by_frame:
                    continue

                self.samples.append({
                    "image": img_path,
                    **masks_by_frame[frame_id]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = load_image(sample["image"])
        _, H, W = image.shape

        right = load_binary_mask(sample.get("right"), (H, W))
        left  = load_binary_mask(sample.get("left"),  (H, W))
        obj   = load_binary_mask(sample.get("object"),(H, W))
        assert right.shape == left.shape == obj.shape == (H, W)

        background = ~(right | left | obj)

        target_mask = torch.stack([right, left, obj, background], dim=0)

        ## Merge into class-index tensor
        #target = torch.zeros((H, W), dtype=torch.long)
        #target[right] = 1
        #target[left]  = 2
        #target[obj]   = 3

        return image, target_mask

dataset = HandObjectSegmentationDataset(
    image_root="images",
    mask_root="merged_masks",
    stream="stream1201-2",
)

image, target = dataset[0]
print(image.shape)
print(target.shape)
print(len(dataset))
assert target.dtype == torch.bool
assert image.shape[0] == 3
assert target.shape[0] == 4


dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
)


for images, targets in dataloader:
    print(images.shape, targets.shape)
    break

