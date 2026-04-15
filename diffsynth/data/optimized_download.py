#!/usr/bin/env python3
"""
Download and pre-process HOT3D clips into per-clip NPZ files.

Each clip produces one file:  training_data/clip-XXXXXX.npz
Arrays inside:
  images_stream1201-1          (N, 280, 280, 3)  uint8
  images_stream1201-2          (N, 280, 280, 3)  uint8
  masks_stream1201-1_hand_LEFT  (N, 280, 280)     uint8
  masks_stream1201-1_hand_RIGHT (N, 280, 280)     uint8
  masks_stream1201-1_object     (N, 280, 280)     uint8
  (same for stream1201-2)

Run modes
---------
Download + process new clips (multi-node safe):
  node 0: python optimized_download.py --clip-start 191 --clip-end 526  --tar-dir diffsynth/data/tar_recv
  node 1: python optimized_download.py --clip-start 527 --clip-end 862  --tar-dir diffsynth/data/tar_recv_1
  node 2: python optimized_download.py --clip-start 863 --clip-end 1199 --tar-dir diffsynth/data/tar_recv_2

Convert existing PNG clips to NPZ (frees inodes as it goes):
  python optimized_download.py --convert
  python optimized_download.py --convert --dry-run
"""
import argparse
import io
import logging
import multiprocessing
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Any, Dict, Optional

import clip_util
import numpy as np
from PIL import Image

from tar_hf_import import tar_hf_import
from hand_tracking_toolkit import rasterizer
from hand_tracking_toolkit.dataset import HandShapeCollection
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel


OUTPUT_SIZE: Optional[tuple] = (280, 280)  # (width, height); None = no resize
STREAMS = ["stream1201-1", "stream1201-2"]

_WORKER_MANO_MODEL: Optional[MANOHandModel] = None
_WORKER_HAND_TYPE: Optional[str] = None


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------

def setup_worker(
    log_queue: multiprocessing.Queue,
    hand_type: str,
    mano_model_dir: Optional[str],
    log_level: int = logging.INFO,
) -> None:
    global _WORKER_MANO_MODEL, _WORKER_HAND_TYPE
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(QueueHandler(log_queue))
    root.setLevel(log_level)
    _WORKER_HAND_TYPE = hand_type
    _WORKER_MANO_MODEL = MANOHandModel(mano_model_dir) if hand_type == "mano" and mano_model_dir else None


def setup_parent_logging(log_queue: multiprocessing.Queue, log_level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(QueueHandler(log_queue))
    root.setLevel(log_level)


def resize_image(arr: np.ndarray, size: Optional[tuple], is_mask: bool = False) -> np.ndarray:
    if size is None:
        return arr
    resample = Image.NEAREST if is_mask else Image.LANCZOS
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return np.array(Image.fromarray(arr).resize(size, resample=resample))


# ---------------------------------------------------------------------------
# Core processing: tar -> NPZ
# ---------------------------------------------------------------------------

def process_clip(clip_path: str, undistort: bool, output_dir: str) -> str:
    """Process a single clip tar into one NPZ. Deletes the tar on success."""
    log = logging.getLogger(__name__)
    mano_model = _WORKER_MANO_MODEL
    hand_type = _WORKER_HAND_TYPE

    with open(clip_path, "rb") as f:
        tar_bytes = f.read()
    log.debug(f"[worker] read {len(tar_bytes)/1e6:.1f} MB from {os.path.basename(clip_path)}")
    tar = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r")

    clip_name = os.path.basename(clip_path).split(".tar")[0]
    hand_shape: Optional[HandShapeCollection] = clip_util.load_hand_shape(tar)
    total_frames = clip_util.get_number_of_frames(tar)
    log.info(f"Processing {clip_name} ({total_frames} frames)")

    # Accumulators: stream_key -> list of per-frame arrays
    images_acc: Dict[str, list] = {}
    masks_acc: Dict[str, list] = {}   # key: "stream{sk}_hand_LEFT" etc.

    for frame_id in range(total_frames):
        frame_key = f"{frame_id:06d}"
        cameras, _ = clip_util.load_cameras(tar, frame_key)
        image_streams = sorted(cameras.keys(), key=lambda x: int(x.split("-")[0]))

        hands: Optional[Dict[str, Any]] = clip_util.load_hand_annotations(tar, frame_key)
        objects: Optional[Dict[str, Any]] = clip_util.load_object_annotations(tar, frame_key)

        hand_meshes = {}
        if hand_shape is not None and hands is not None:
            hand_meshes = clip_util.get_hand_meshes(hands, hand_shape, hand_type, mano_model)

        merged_object_masks: Dict[str, np.ndarray] = {}

        for stream_id in image_streams:
            sk = str(stream_id)
            camera_model = cameras[stream_id]
            if undistort:
                camera_model = clip_util.convert_to_pinhole_camera(camera_model)

            # --- Image ---
            image = clip_util.load_image(tar, frame_key, sk)
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            image = np.rot90(image, k=3)
            image = resize_image(image, OUTPUT_SIZE, is_mask=False)
            H, W = image.shape[:2]
            images_acc.setdefault(sk, []).append(image)

            # --- Object masks (accumulate across instances) ---
            if objects is not None:
                for instance_list in objects.values():
                    for instance in instance_list:
                        mask_rle = instance.get("masks_amodal", {}).get(sk)
                        if mask_rle is None:
                            continue
                        mask = clip_util.decode_binary_mask_rle(mask_rle)
                        existing = merged_object_masks.get(sk)
                        if existing is None:
                            merged_object_masks[sk] = mask
                        else:
                            np.bitwise_or(existing, mask, out=existing)

            # --- Hand masks (rasterize per stream, zeros if absent) ---
            rasterized: Dict[str, np.ndarray] = {}
            for hand_side, hand_mesh in hand_meshes.items():
                _, mask, _ = rasterizer.rasterize_mesh(
                    verts=hand_mesh.vertices,
                    faces=hand_mesh.faces,
                    vert_normals=hand_mesh.vertex_normals,
                    camera=camera_model,
                )
                mask_out = np.rot90(mask.astype(np.uint8) * 255, k=3)
                mask_out = resize_image(mask_out, (W, H), is_mask=True)
                rasterized[hand_side.name] = mask_out

            for side in ["LEFT", "RIGHT"]:
                key = f"stream{sk}_hand_{side}"
                masks_acc.setdefault(key, []).append(
                    rasterized.get(side, np.zeros((H, W), dtype=np.uint8))
                )

            # --- Object mask for this stream ---
            key = f"stream{sk}_object"
            if sk in merged_object_masks:
                obj = resize_image(
                    np.rot90(merged_object_masks[sk].astype(np.uint8) * 255, k=3),
                    (W, H), is_mask=True,
                )
            else:
                obj = np.zeros((H, W), dtype=np.uint8)
            masks_acc.setdefault(key, []).append(obj)

        if (frame_id + 1) % 25 == 0 or (frame_id + 1) == total_frames:
            log.info(f"{clip_name}: {frame_id + 1}/{total_frames} frames")

    tar.close()

    # --- Build and save NPZ (atomic: .tmp -> final) ---
    npz_data = {}
    for sk, frames in images_acc.items():
        npz_data[f"images_stream{sk}"] = np.stack(frames)
    for key, frames in masks_acc.items():
        npz_data[f"masks_{key}"] = np.stack(frames)

    npz_path = os.path.join(output_dir, f"{clip_name}.npz")
    tmp_stem = os.path.join(output_dir, f"{clip_name}.tmp")
    np.savez_compressed(tmp_stem, **npz_data)   # writes clip-XXXXXX.tmp.npz
    os.rename(tmp_stem + ".npz", npz_path)
    log.info(f"Saved {npz_path}")

    try:
        os.remove(clip_path)
        log.debug(f"Deleted tar {clip_path}")
    except OSError as e:
        log.warning(f"Could not delete tar {clip_path}: {e}")

    return clip_name


# ---------------------------------------------------------------------------
# Convert existing PNG clips to NPZ  (frees inodes as it goes)
# ---------------------------------------------------------------------------

def _load_png(path: Path, shape_hw: tuple) -> np.ndarray:
    import imageio.v2 as iio
    H, W = shape_hw
    if path.exists():
        arr = iio.imread(str(path))
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr
    # Return zeros of the appropriate shape
    channels = 3 if "image" in str(path) else 0
    return np.zeros((H, W, 3) if channels else (H, W), dtype=np.uint8)


def convert_existing(
    masks_root: Path,
    images_root: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> None:
    """Convert all completed PNG clips to NPZ, deleting PNGs per clip as we go."""
    import imageio.v2 as iio
    import shutil

    clip_names = sorted(
        d.name for d in masks_root.iterdir()
        if d.is_dir() and not (output_dir / f"{d.name}.npz").exists()
    )
    print(f"Clips to convert: {len(clip_names)}")
    if dry_run:
        for c in clip_names:
            print(f"  {c}")
        return

    for i, clip_name in enumerate(clip_names, 1):
        mask_clip_dir = masks_root / clip_name

        frame_ids = sorted(set(p.name.split("_")[0] for p in mask_clip_dir.glob("*.png")))
        if not frame_ids:
            print(f"[{i}/{len(clip_names)}] {clip_name}: no frames found, skipping")
            continue

        # Infer spatial shape from first available image
        H, W = 280, 280
        for stream in STREAMS:
            first = images_root / stream / clip_name / f"{frame_ids[0]}.png"
            if first.exists():
                arr = iio.imread(str(first))
                H, W = arr.shape[:2]
                break

        # Load all PNGs into memory first
        arrays: dict = {}
        for stream in STREAMS:
            imgs, ml, mr, mo = [], [], [], []
            for fid in frame_ids:
                ip = images_root / stream / clip_name / f"{fid}.png"
                img = iio.imread(str(ip)) if ip.exists() else np.zeros((H, W, 3), dtype=np.uint8)
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                imgs.append(img)

                for store, mtype in [(ml, "hand_LEFT"), (mr, "hand_RIGHT"), (mo, "object")]:
                    mp = mask_clip_dir / f"{fid}_mask_{stream}_{mtype}.png"
                    store.append(iio.imread(str(mp)) if mp.exists() else np.zeros((H, W), dtype=np.uint8))

            arrays[f"images_{stream}"]           = np.stack(imgs)
            arrays[f"masks_{stream}_hand_LEFT"]   = np.stack(ml)
            arrays[f"masks_{stream}_hand_RIGHT"]  = np.stack(mr)
            arrays[f"masks_{stream}_object"]      = np.stack(mo)

        # Delete PNG dirs BEFORE writing NPZ to free inodes first
        shutil.rmtree(mask_clip_dir, ignore_errors=True)
        for stream in STREAMS:
            img_dir = images_root / stream / clip_name
            if img_dir.exists():
                shutil.rmtree(img_dir)

        # Save NPZ atomically: savez_compressed auto-appends .npz, so use a
        # tmp stem without .npz and rename the resulting file.
        npz_path = output_dir / f"{clip_name}.npz"
        tmp_stem = output_dir / f"{clip_name}.tmp"
        np.savez_compressed(str(tmp_stem), **arrays)   # writes clip-XXXXXX.tmp.npz
        (output_dir / f"{clip_name}.tmp.npz").rename(npz_path)

        print(f"[{i}/{len(clip_names)}] {clip_name} ({len(frame_ids)} frames)")

    print("Conversion complete.")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_clip_index(filename: str) -> int:
    return int(filename.split(".tar")[0].split("clip-")[1])


def already_processed(clip_name: str, output_dir: str) -> bool:
    """True if the clip's NPZ already exists."""
    return os.path.isfile(os.path.join(output_dir, f"{clip_name}.npz"))


# ---------------------------------------------------------------------------
# Main: download + process pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process HOT3D clips into NPZ files (multi-node safe).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--convert", action="store_true",
                        help="Convert existing PNG clips to NPZ instead of downloading.")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --convert: list clips without converting.")
    parser.add_argument("--masks-root", default="diffsynth/data/training_masks",
                        help="Source mask dir (only used with --convert).")
    parser.add_argument("--images-root", default="diffsynth/data/training_images",
                        help="Source images dir (only used with --convert).")
    parser.add_argument("--clip-start", type=int, default=191)
    parser.add_argument("--clip-end",   type=int, default=-1,
                        help="-1 = no upper bound.")
    parser.add_argument("--tar-dir",    default="diffsynth/data/tar_recv")
    parser.add_argument("--output-dir", default="diffsynth/data/training_data",
                        help="Output directory for NPZ files (shared across nodes).")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-stored-clips", type=int, default=4)
    parser.add_argument("--mano-model-dir", default="./diffsynth/data/mano/models/")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Convert mode ---
    if args.convert:
        convert_existing(
            masks_root=Path(args.masks_root),
            images_root=Path(args.images_root),
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
        return

    # --- Download + process mode ---
    clips_dir      = args.tar_dir
    mano_model_dir = args.mano_model_dir
    hand_type      = "mano"
    clip_start     = args.clip_start
    clip_end       = args.clip_end
    num_workers    = args.num_workers
    max_stored_clips = args.max_stored_clips
    undistort      = False

    os.makedirs(clips_dir, exist_ok=True)

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue: multiprocessing.Queue = multiprocessing.Manager().Queue(-1)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s %(message)s", "%H:%M:%S")
    )
    listener = QueueListener(log_queue, console_handler, respect_handler_level=True)
    listener.start()
    setup_parent_logging(log_queue, log_level)
    log = logging.getLogger(__name__)

    range_str = f"{clip_start}–{clip_end if clip_end >= 0 else '∞'}"
    log.info(f"Starting pipeline | clips: {range_str} | tar-dir: {clips_dir} | "
             f"workers: {num_workers} | max-on-disk: {max_stored_clips}")

    # Pre-populate done set from output_dir
    processed_clips: set[str] = set()
    for name in output_dir.iterdir() if output_dir.exists() else []:
        if name.suffix == ".npz":
            processed_clips.add(name.stem + ".tar")
    log.info(f"Already processed: {len(processed_clips)} clips (will skip)")

    # Clean up stale tars for already-processed clips
    for p in os.listdir(clips_dir):
        if p.endswith(".tar") and p in processed_clips:
            try:
                os.remove(os.path.join(clips_dir, p))
                log.info(f"Cleaned up stale tar: {p}")
            except OSError as e:
                log.warning(f"Could not remove {p}: {e}")

    download_index = clip_start
    more_available = True

    try:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=setup_worker,
            initargs=(log_queue, hand_type, mano_model_dir, log_level),
        ) as executor:
            futures: dict = {}

            while more_available or futures:
                # Download loop
                while more_available and (clip_end < 0 or download_index <= clip_end):
                    on_disk = len([p for p in os.listdir(clips_dir) if p.endswith(".tar")])
                    if on_disk >= max_stored_clips:
                        log.debug(f"[download] disk full ({on_disk}/{max_stored_clips}), pausing")
                        break
                    clip_filename = f"clip-{download_index:06d}.tar"
                    if clip_filename in processed_clips:
                        log.debug(f"[download] skipping already-done clip {download_index}")
                        download_index += 1
                        continue
                    log.info(f"Downloading clip {download_index}")
                    try:
                        found = tar_hf_import(download_index, dest_dir=clips_dir)
                    except Exception as exc:
                        log.error(f"Download error for clip {download_index}: {exc}")
                        break
                    if not found:
                        log.info(f"Clip {download_index} returned 404 — end of dataset.")
                        more_available = False
                        break
                    download_index += 1

                # Submit ready tars
                in_flight = set(futures.values())
                pending_files = [
                    p for p in os.listdir(clips_dir)
                    if p.endswith(".tar")
                    and p not in processed_clips
                    and p not in in_flight
                    and get_clip_index(p) >= clip_start
                    and (clip_end < 0 or get_clip_index(p) <= clip_end)
                ]
                for clip in pending_files:
                    future = executor.submit(
                        process_clip,
                        os.path.join(clips_dir, clip),
                        undistort,
                        str(output_dir),
                    )
                    futures[future] = clip
                    log.info(f"Submitted {clip}")

                # Collect completed
                done = [f for f in futures if f.done()]
                for future in done:
                    clip = futures.pop(future)
                    try:
                        clip_name = future.result()
                        processed_clips.add(clip)
                        log.info(f"Done: {clip_name}")
                    except Exception as exc:
                        log.exception(f"ERROR processing {clip}: {exc}")

                if futures:
                    wait(futures, timeout=2, return_when=FIRST_COMPLETED)

        log.info("All clips processed.")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
