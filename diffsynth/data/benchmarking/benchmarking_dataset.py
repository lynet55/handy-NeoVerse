from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

STREAMS = ["stream1201-1", "stream1201-2"]


class BenchmarkingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str = "diffsynth/data/test_data", # TODO: fix the default root
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
            for stream in self.streams:
                if f"images_{stream}" in npz.files:
                    self.samples.append({
                        "npz_path": str(npz_path),
                        "clip_name": clip_name,
                        "stream": stream,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        npz = np.load(s["npz_path"], mmap_mode="r")
        stream = s["stream"]

        images = torch.tensor(
            npz[f"images_{stream}"][:], dtype=torch.float32
        ).permute(0, 3, 1, 2) / 255.0

        right = torch.tensor(npz[f"masks_{stream}_hand_RIGHT"][:] > 0)
        left  = torch.tensor(npz[f"masks_{stream}_hand_LEFT"][:]  > 0)
        obj   = torch.tensor(npz[f"masks_{stream}_object"][:]     > 0)

        background = ~(right | left | obj)
        target_mask = torch.stack([right, left, obj, background], dim=0).float()
        
        return images, target_mask, s["clip_name"], stream

