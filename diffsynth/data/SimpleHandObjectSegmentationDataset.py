from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Sampler

STREAMS = ["stream1201-1", "stream1201-2"]


def load_image(path: Path) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def load_binary_mask(path: Path, shape) -> torch.Tensor:
    H, W = shape
    if path is None or not path.exists():
        return torch.zeros((H, W), dtype=torch.bool)
    mask = Image.open(path).convert("L")
    if mask.size != (W, H):
        mask = mask.resize((W, H), resample=Image.NEAREST)
    mask = np.array(mask) > 0
    return torch.from_numpy(mask)


class HandObjectSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_root: str = "diffsynth/data/training_images/",
        mask_root: str = "diffsynth/data/training_masks/",
        streams: list[str] = None,
    ):
        self.image_root_base = Path(image_root)
        self.mask_root = Path(mask_root)
        self.streams = streams or STREAMS

        self.samples: list[dict] = []
        self._build_index()

    def _build_index(self) -> None:
        """Index all clips and frames across all streams."""
        for stream in self.streams:
            stream_root = self.image_root_base / stream
            if not stream_root.exists():
                continue

            for clip_dir in sorted(stream_root.iterdir()):
                if not clip_dir.is_dir():
                    continue

                clip_name = clip_dir.name
                mask_clip_dir = self.mask_root / clip_name
                if not mask_clip_dir.exists():
                    continue

                masks_by_frame: dict[str, dict] = {}
                for p in mask_clip_dir.glob(f"*_{stream}_*.png"):
                    frame_id = p.name.split("_")[0]
                    masks_by_frame.setdefault(frame_id, {})
                    if "hand_LEFT" in p.name:
                        masks_by_frame[frame_id]["left"] = p
                    elif "hand_RIGHT" in p.name:
                        masks_by_frame[frame_id]["right"] = p
                    elif "object" in p.name:
                        masks_by_frame[frame_id]["object"] = p

                for img_path in sorted(clip_dir.glob("*.png")):
                    frame_id = img_path.stem
                    if frame_id not in masks_by_frame:
                        continue

                    self.samples.append(
                        {
                            "image": img_path,
                            "clip_name": clip_name,
                            "stream": stream,
                            **masks_by_frame[frame_id],
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image = load_image(sample["image"])
        _, H, W = image.shape

        right = load_binary_mask(sample.get("right"), (H, W))
        left  = load_binary_mask(sample.get("left"),  (H, W))
        obj   = load_binary_mask(sample.get("object"), (H, W))
        assert right.shape == left.shape == obj.shape == (H, W)

        background = ~(right | left | obj)
        target_mask = torch.stack([right, left, obj, background], dim=0).float()

        return image, target_mask, sample["clip_name"], sample["stream"]


class ClipStreamSampler(Sampler):
    """Yields indices so that all frames of one (clip, stream) pair are
    emitted consecutively and in ascending frame order before the next pair.

    Pass ``shuffle_clips=True`` to randomise pair order while keeping
    within-pair frame order intact.
    """

    def __init__(self, dataset: HandObjectSegmentationDataset, shuffle_clips: bool = False):
        self.dataset = dataset
        self.shuffle_clips = shuffle_clips

        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            groups[(sample["clip_name"], sample["stream"])].append(idx)

        self._groups: list[list[int]] = list(groups.values())

    def __iter__(self):
        groups = self._groups
        if self.shuffle_clips:
            order = torch.randperm(len(groups)).tolist()
            groups = [groups[i] for i in order]
        for group in groups:
            yield from group

    def __len__(self) -> int:
        return len(self.dataset)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = HandObjectSegmentationDataset()
    print(f"Total samples: {len(dataset)}")
    assert len(dataset) > 0, "No samples found — check image_root and mask_root."

    sampler = ClipStreamSampler(dataset, shuffle_clips=False)
    loader = DataLoader(dataset, sampler=sampler, batch_size=10, num_workers=0)

    i = 0
    for images, gt_mask, clip_names, stream_names in loader:
        assert gt_mask.dtype == torch.float
        assert images.shape[1] == 3
        assert gt_mask.shape[1] == 4
        if not i % 10:
            print(f"Batch {i} — clip: {clip_names[0]}, stream: {stream_names[0]}")
        i += 1

    print("Smoke-test passed.") 
