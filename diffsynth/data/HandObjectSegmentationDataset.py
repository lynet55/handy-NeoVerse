import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

STREAMS = ["stream1201-1", "stream1201-2"]


class HandObjectSegmentationDataset(torch.utils.data.Dataset):
    """Loads per-clip NPZ files from training_data/.

    Supports online use: call poll_for_new_clips() between epochs to pick up
    clips that were processed while training was running.
    """

    def __init__(
        self,
        data_root: str = "diffsynth/data/training_data",
        streams: list[str] = None,
    ):
        self.data_root = Path(data_root)
        self.streams = streams or STREAMS
        self.samples: list[dict] = []
        self._indexed_clips: set[str] = set()
        self._build_index()

    def _index_clip(self, clip_name: str) -> None:
        npz_path = self.data_root / f"{clip_name}.npz"
        npz = np.load(str(npz_path), mmap_mode="r")
        n_frames = next(npz[k].shape[0] for k in npz.files if k.startswith("images_"))
        for stream in self.streams:
            if f"images_{stream}" not in npz.files:
                continue
            for frame_idx in range(n_frames):
                self.samples.append(
                    {"clip_name": clip_name, "stream": stream, "frame_idx": frame_idx}
                )

    def _build_index(self) -> None:
        for npz_path in sorted(self.data_root.glob("clip-*.npz")):
            clip_name = npz_path.stem
            if clip_name not in self._indexed_clips:
                self._index_clip(clip_name)
                self._indexed_clips.add(clip_name)

    def poll_for_new_clips(self) -> list[str]:
        """Index clips that finished processing since the last call.
        Call this between epochs from the main process only."""
        new_clips = []
        for npz_path in sorted(self.data_root.glob("clip-*.npz")):
            clip_name = npz_path.stem
            if clip_name not in self._indexed_clips:
                self._index_clip(clip_name)
                self._indexed_clips.add(clip_name)
                new_clips.append(clip_name)
                print(f"Dataset: ingested new clip '{clip_name}'")
        return new_clips

    def remove_clips(self, clip_names: list[str]) -> None:
        """Remove clips from the index. Call only after an epoch finishes."""
        clip_set = set(clip_names)
        self.samples = [s for s in self.samples if s["clip_name"] not in clip_set]
        self._indexed_clips -= clip_set

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


def dataloader_with_cleanup(
    dataset: HandObjectSegmentationDataset,
    shuffle_clips: bool = False,
    poll_new_clips: bool = False,
    poll_interval_s: float = 30.0,
    **dataloader_kwargs,
):
    """Wrap a DataLoader, delete each clip's NPZ once all its streams have been
    fully yielded, and optionally poll for new clips between epochs.

    Yields:
        (image, target_mask) batches.
    """
    for forbidden in ("sampler", "batch_sampler", "shuffle"):
        if forbidden in dataloader_kwargs:
            raise ValueError(f"Do not pass '{forbidden}' to dataloader_with_cleanup.")

    if len(dataset) == 0:
        print("No clips available yet. Waiting for the first clip...")
        while len(dataset) == 0:
            time.sleep(poll_interval_s)
            dataset.poll_for_new_clips()
        print(f"First clip ready — starting training ({len(dataset)} frames).")

    while True:
        pair_counts: dict[tuple, int] = defaultdict(int)
        for sample in dataset.samples:
            pair_counts[(sample["clip_name"], sample["stream"])] += 1

        seen_pair_counts: dict[tuple, int] = defaultdict(int)
        streams_remaining: dict[str, int] = defaultdict(int)
        for (clip_name, _) in pair_counts:
            streams_remaining[clip_name] += 1

        sampler = ClipStreamSampler(dataset, shuffle_clips=shuffle_clips)
        loader = DataLoader(dataset, sampler=sampler, **dataloader_kwargs)
        clips_to_remove: list[str] = []

        for image, target_mask, clip_names, stream_names in loader:
            for clip_name, stream in zip(clip_names, stream_names):
                key = (clip_name, stream)
                seen_pair_counts[key] += 1
                if seen_pair_counts[key] == pair_counts[key]:
                    del seen_pair_counts[key]
                    streams_remaining[clip_name] -= 1
                    if streams_remaining[clip_name] == 0:
                        npz_path = dataset.data_root / f"{clip_name}.npz"
                        if npz_path.exists():
                            npz_path.unlink()
                            print(f"Deleted {npz_path}")
                        clips_to_remove.append(clip_name)
                        del streams_remaining[clip_name]
            yield image, target_mask

        if clips_to_remove:
            dataset.remove_clips(clips_to_remove)

        if not poll_new_clips:
            break

        print("Epoch complete. Waiting for new clips from extractor...")
        while True:
            new_clips = dataset.poll_for_new_clips()
            if new_clips:
                print(f"Starting new epoch with {len(new_clips)} new clip(s): {new_clips}")
                break
            if (dataset.data_root / "ALL_DONE").exists():
                print("Extractor finished and no more clips are pending. Done.")
                return
            time.sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = HandObjectSegmentationDataset()
    dataloader = dataloader_with_cleanup(
        dataset,
        shuffle_clips=False,
        poll_new_clips=True,
        poll_interval_s=30,
        batch_size=10,
    )
    for i, (images, gt_mask) in enumerate(dataloader):
        print(f"Batch {i}")
