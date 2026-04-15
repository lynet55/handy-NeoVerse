from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

STREAMS = ["stream1201-1", "stream1201-2"]


class HandObjectSegmentationDataset(torch.utils.data.Dataset):
    """Loads per-clip NPZ files from training_data/.

    Each sample is one (clip, stream, frame) triple. __getitem__ opens the
    NPZ with mmap_mode='r' so only the requested frame is paged in — the OS
    keeps pages warm while the ClipStreamSampler iterates through a clip
    consecutively.
    """

    def __init__(
        self,
        data_root: str = "diffsynth/data/training_data",
        streams: list[str] = None,
    ):
        self.data_root = Path(data_root)
        self.streams = streams or STREAMS
        self.samples: list[dict] = []
        self._build_index()

    def _build_index(self) -> None:
        for npz_path in sorted(self.data_root.glob("clip-*.npz")):
            clip_name = npz_path.stem
            npz = np.load(str(npz_path), mmap_mode="r")
            # Infer frame count from the first images array found
            n_frames = next(
                npz[k].shape[0] for k in npz.files if k.startswith("images_")
            )
            for stream in self.streams:
                if f"images_{stream}" not in npz.files:
                    continue
                for frame_idx in range(n_frames):
                    self.samples.append(
                        {"clip_name": clip_name, "stream": stream, "frame_idx": frame_idx}
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        clip_name, stream, frame_idx = s["clip_name"], s["stream"], s["frame_idx"]

        npz = np.load(str(self.data_root / f"{clip_name}.npz"), mmap_mode="r")

        image = torch.tensor(
            npz[f"images_{stream}"][frame_idx], dtype=torch.float32
        ).permute(2, 0, 1) / 255.0

        right = torch.tensor(npz[f"masks_{stream}_hand_RIGHT"][frame_idx] > 0)
        left  = torch.tensor(npz[f"masks_{stream}_hand_LEFT"][frame_idx]  > 0)
        obj   = torch.tensor(npz[f"masks_{stream}_object"][frame_idx]     > 0)

        background = ~(right | left | obj)
        target_mask = torch.stack([right, left, obj, background], dim=0).float()

        return image, target_mask, clip_name, stream


class ClipStreamSampler(Sampler):
    """Yields indices so all frames of one (clip, stream) pair come out
    consecutively before the next pair. Optionally shuffles pair order."""

    def __init__(self, dataset: HandObjectSegmentationDataset, shuffle_clips: bool = False):
        self.dataset = dataset
        self.shuffle_clips = shuffle_clips

        groups: dict[tuple, list[int]] = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            groups[(sample["clip_name"], sample["stream"])].append(idx)
        self._groups = list(groups.values())

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
    assert len(dataset) > 0, "No samples found — check data_root."

    sampler = ClipStreamSampler(dataset, shuffle_clips=False)
    loader = DataLoader(dataset, sampler=sampler, batch_size=10, num_workers=0)

    for i, (images, gt_mask, clip_names, stream_names) in enumerate(loader):
        assert gt_mask.dtype == torch.float
        assert images.shape[1] == 3
        assert gt_mask.shape[1] == 4
        if i % 10 == 0:
            print(f"Batch {i} — clip: {clip_names[0]}, stream: {stream_names[0]}")

    print("Smoke-test passed.")
