import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Sampler

TAR_DIR = Path("diffsynth/data/tar_recv")  # adjust if needed
SENTINEL_DIR = Path("diffsynth/data/ready_sentinels/")

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


def delete_clip(
    clip_name: str,
    image_root_base: Path,
    mask_root: Path,
    streams: list[str],
    sentinel_dir: Path,
) -> None:
    """Delete all data for a finished clip once ALL streams are done:
    image folders for every stream, the shared mask folder, the tar file,
    and the sentinel file."""
    for stream in streams:
        image_clip_dir = image_root_base / stream / clip_name
        if image_clip_dir.exists():
            shutil.rmtree(image_clip_dir)
            print(f"Deleted: {image_clip_dir}")

    mask_clip_dir = mask_root / clip_name
    if mask_clip_dir.exists():
        shutil.rmtree(mask_clip_dir)
        print(f"Deleted: {mask_clip_dir}")

    tar_path = TAR_DIR / f"{clip_name}.tar"
    if tar_path.exists():
        tar_path.unlink()
        print(f"Deleted: {tar_path}")

    sentinel_path = sentinel_dir / f"{clip_name}.ready"
    if sentinel_path.exists():
        sentinel_path.unlink()
        print(f"Deleted: {sentinel_path}")


class HandObjectSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_root: str = "diffsynth/data/training_images/",
        mask_root: str = "diffsynth/data/training_masks/",
        sentinel_dir: str = str(SENTINEL_DIR),
        streams: list[str] = None,
    ):
        self.image_root_base = Path(image_root)
        self.mask_root = Path(mask_root)
        self.sentinel_dir = Path(sentinel_dir)
        self.streams = streams or STREAMS

        self.samples: list[dict] = []
        # Tracks which clips have already been indexed so we never double-count.
        self._indexed_clips: set[str] = set()

        self._build_index()

    def _index_clip(self, clip_name: str) -> None:
        """Index all frames for a single clip across all streams and append
        them to self.samples. Safe to call only once per clip."""
        for stream in self.streams:
            clip_dir = self.image_root_base / stream / clip_name
            if not clip_dir.exists():
                continue

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

    def remove_clips(self, clip_names: list[str]) -> None:
        """Remove all samples for the given clips from self.samples.
        Only call this AFTER the DataLoader epoch has finished iterating,
        never while a loader is still active — the sampler's pre-computed
        indices would go out of range if the list shrinks mid-epoch."""
        clip_set = set(clip_names)
        self.samples = [s for s in self.samples if s["clip_name"] not in clip_set]
        self._indexed_clips -= clip_set

    def _build_index(self) -> None:
        """Index all clips that already have a sentinel file at startup."""
        for sentinel in sorted(self.sentinel_dir.glob("*.ready")):
            clip_name = sentinel.stem
            if clip_name not in self._indexed_clips:
                self._index_clip(clip_name)
                self._indexed_clips.add(clip_name)

    def poll_for_new_clips(self) -> list[str]:
        """Scan the sentinel directory and index any clips that have finished
        processing since the last call. Returns the list of newly added clip names.

        Call this between epochs from the main process — NOT inside __getitem__,
        to avoid races with DataLoader workers.
        """
        new_clips = []
        for sentinel in sorted(self.sentinel_dir.glob("*.ready")):
            clip_name = sentinel.stem
            if clip_name not in self._indexed_clips:
                self._index_clip(clip_name)
                self._indexed_clips.add(clip_name)
                new_clips.append(clip_name)
                print(f"Dataset: ingested new clip '{clip_name}'")
        return new_clips

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


def dataloader_with_cleanup(
    dataset: HandObjectSegmentationDataset,
    shuffle_clips: bool = False,
    poll_new_clips: bool = False,
    poll_interval_s: float = 30.0,
    **dataloader_kwargs,
):
    """Wrap a DataLoader and delete each clip's data once ALL streams have been
    fully yielded. Optionally polls for newly processed clips between epochs.

    Args:
        dataset:          A HandObjectSegmentationDataset instance.
        shuffle_clips:    Randomise (clip, stream) pair order while keeping
                          within-pair frame order intact.
        poll_new_clips:   If True, call dataset.poll_for_new_clips() after each
                          full pass and wait for at least one new clip before
                          starting the next epoch. Use when extract_training_data
                          is running concurrently.
        poll_interval_s:  Seconds between polls when no new clip has arrived
                          (only relevant when poll_new_clips=True).

    Yields:
        (image, target_mask) batches — same as iterating the DataLoader directly.
    """
    for forbidden in ("sampler", "batch_sampler", "shuffle"):
        if forbidden in dataloader_kwargs:
            raise ValueError(
                f"Do not pass '{forbidden}' to dataloader_with_cleanup; "
                "ordering is managed by ClipStreamSampler."
            )

    # Wait until at least one clip is ready before starting the first epoch.
    if len(dataset) == 0:
        print("No clips available yet. Waiting for the first clip...")
        while len(dataset) == 0:
            time.sleep(poll_interval_s)
            dataset.poll_for_new_clips()
        print(f"First clip ready — starting training ({len(dataset)} frames).")

    while True:
        # Snapshot per-(clip, stream) frame counts from the current sample list.
        # Built once per epoch so counts are stable while the loader runs.
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        for sample in dataset.samples:
            pair_counts[(sample["clip_name"], sample["stream"])] += 1

        # How many frames seen so far for each (clip, stream) pair.
        seen_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        # How many streams are still pending per clip.
        streams_remaining: dict[str, int] = defaultdict(int)
        for (clip_name, _stream) in pair_counts:
            streams_remaining[clip_name] += 1

        sampler = ClipStreamSampler(dataset, shuffle_clips=shuffle_clips)
        loader = DataLoader(dataset, sampler=sampler, **dataloader_kwargs)

        # Clips fully consumed this epoch — deleted from disk but not yet
        # removed from dataset.samples (that would shift indices mid-epoch).
        clips_to_remove: list[str] = []

        for image, target_mask, clip_names, stream_names in loader:
            for clip_name, stream in zip(clip_names, stream_names):
                key = (clip_name, stream)
                seen_pair_counts[key] += 1
                if seen_pair_counts[key] == pair_counts[key]:
                    del seen_pair_counts[key]
                    streams_remaining[clip_name] -= 1
                    if streams_remaining[clip_name] == 0:
                        # Files can be deleted immediately — we won't read
                        # them again this epoch since the sampler is
                        # contiguous and this stream group is exhausted.
                        delete_clip(
                            clip_name,
                            dataset.image_root_base,
                            dataset.mask_root,
                            dataset.streams,
                            dataset.sentinel_dir,
                        )
                        clips_to_remove.append(clip_name)
                        del streams_remaining[clip_name]

            yield image, target_mask

        # Epoch complete — now safe to remove consumed entries from the sample
        # list. The old sampler and loader are done so no indices are live.
        if clips_to_remove:
            dataset.remove_clips(clips_to_remove)

        if not poll_new_clips:
            break

        # End of epoch: wait until at least one new clip is ready, or stop if
        # the extractor is done and no further clips will ever arrive.
        print("Epoch complete. Waiting for new clips from extractor...")
        while True:
            new_clips = dataset.poll_for_new_clips()
            if new_clips:
                print(f"Starting new epoch with {len(new_clips)} new clip(s): {new_clips}")
                break
            if (dataset.sentinel_dir / "ALL_DONE").exists():
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
    i = 0
    for step, (images, gt_mask) in enumerate(dataloader):
        #if not i % 10:
        print(f"Batch number: {i} iterated")
        i += 1
        time.sleep(0.1)
