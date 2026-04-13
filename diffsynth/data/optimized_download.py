#!/usr/bin/env python3
"""
Download and pre-process HOT3D clips (masks + images) in a streaming pipeline.

Designed to run on up to 3 independent CPU nodes in parallel.  Each node
operates on a non-overlapping clip range and its own tar staging directory,
while writing into the *shared* training_masks/ and training_images/ trees
(safe because every clip writes only into its own sub-directory).

Suggested 3-node split for clips 191-1199 (assuming 0-190 already done):
  node 0: --clip-start 191 --clip-end 526  --tar-dir diffsynth/data/tar_recv
  node 1: --clip-start 527 --clip-end 862  --tar-dir diffsynth/data/tar_recv_1
  node 2: --clip-start 863 --clip-end 1199 --tar-dir diffsynth/data/tar_recv_2

The script automatically skips clips whose output directory already exists and
is non-empty, so re-running is safe and already-processed clips (0-190) are
never re-downloaded or re-processed.
"""
import argparse
import io
import logging
import multiprocessing
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from logging.handlers import QueueHandler, QueueListener
from typing import Any, Dict, Optional

import clip_util
import imageio
import numpy as np
from PIL import Image
from pathlib import Path

from tar_hf_import import tar_hf_import
from hand_tracking_toolkit import rasterizer
from hand_tracking_toolkit.dataset import HandShapeCollection
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel


# Target size for saved images and masks. Set to None to skip resizing.
OUTPUT_SIZE: Optional[tuple] = (280, 280)  # (width, height)

# Worker-process globals populated by the initializer.
_WORKER_MANO_MODEL: Optional[MANOHandModel] = None
_WORKER_HAND_TYPE: Optional[str] = None


def setup_worker(
    log_queue: multiprocessing.Queue,
    hand_type: str,
    mano_model_dir: Optional[str],
    log_level: int = logging.INFO,
) -> None:
    """Initializer run in each worker process."""
    global _WORKER_MANO_MODEL, _WORKER_HAND_TYPE

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(QueueHandler(log_queue))
    root.setLevel(log_level)

    _WORKER_HAND_TYPE = hand_type
    if hand_type == "mano" and mano_model_dir:
        _WORKER_MANO_MODEL = MANOHandModel(mano_model_dir)
    else:
        _WORKER_MANO_MODEL = None


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


def process_clip(
    clip_path: str,
    undistort: bool,
    output_dir: str,
    images_dir: Optional[str] = None,
) -> str:
    """Process a single clip tar file. Deletes the tar on success. Returns clip name."""
    log = logging.getLogger(__name__)
    mano_model = _WORKER_MANO_MODEL
    hand_type = _WORKER_HAND_TYPE

    with open(clip_path, "rb") as f:
        tar_bytes = f.read()
    log.debug(f"[worker] read {len(tar_bytes)/1e6:.1f} MB from {os.path.basename(clip_path)}")
    tar = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r")

    clip_name = os.path.basename(clip_path).split(".tar")[0]
    clip_output_path = os.path.join(output_dir, clip_name)
    os.makedirs(clip_output_path, exist_ok=True)

    hand_shape: Optional[HandShapeCollection] = clip_util.load_hand_shape(tar)
    total_frames = clip_util.get_number_of_frames(tar)
    log.info(f"Processing {clip_name} ({total_frames} frames)")

    stream_image_dirs: Dict[str, str] = {}

    for frame_id in range(total_frames):
        frame_key = f"{frame_id:06d}"

        cameras, _ = clip_util.load_cameras(tar, frame_key)
        image_streams = sorted(cameras.keys(), key=lambda x: int(x.split("-")[0]))

        hands: Optional[Dict[str, Any]] = clip_util.load_hand_annotations(tar, frame_key)
        objects: Optional[Dict[str, Any]] = clip_util.load_object_annotations(tar, frame_key)

        hand_meshes = {}
        if hand_shape is not None and hands is not None:
            hand_meshes = clip_util.get_hand_meshes(hands, hand_shape, hand_type, mano_model)

        merged_object_masks: Dict[str, Optional[np.ndarray]] = {}
        stream_output_sizes: Dict[str, Optional[tuple]] = {}

        for stream_id in image_streams:
            stream_key = str(stream_id)
            camera_model = cameras[stream_id]

            if undistort:
                camera_model = clip_util.convert_to_pinhole_camera(camera_model)

            image = clip_util.load_image(tar, frame_key, stream_key)
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            image = np.rot90(image, k=3)
            image = resize_image(image, OUTPUT_SIZE, is_mask=False)
            stream_output_sizes[stream_key] = (image.shape[1], image.shape[0])

            if images_dir is not None:
                stream_output_dir = stream_image_dirs.get(stream_key)
                if stream_output_dir is None:
                    stream_output_dir = os.path.join(images_dir, f"stream{stream_key}", clip_name)
                    os.makedirs(stream_output_dir, exist_ok=True)
                    stream_image_dirs[stream_key] = stream_output_dir
                imageio.imwrite(os.path.join(stream_output_dir, f"{frame_key}.png"), image)

            if objects is not None:
                for instance_list in objects.values():
                    for instance in instance_list:
                        mask_rle = instance.get("masks_amodal", {}).get(stream_key)
                        if mask_rle is None:
                            continue
                        mask = clip_util.decode_binary_mask_rle(mask_rle)
                        existing = merged_object_masks.get(stream_key)
                        if existing is None:
                            merged_object_masks[stream_key] = mask
                        else:
                            np.bitwise_or(existing, mask, out=existing)

            for hand_side, hand_mesh in hand_meshes.items():
                _, mask, _ = rasterizer.rasterize_mesh(
                    verts=hand_mesh.vertices,
                    faces=hand_mesh.faces,
                    vert_normals=hand_mesh.vertex_normals,
                    camera=camera_model,
                )
                mask_out = np.rot90(mask.astype(np.uint8) * 255, k=3)
                mask_out = resize_image(mask_out, stream_output_sizes[stream_key], is_mask=True)
                mask_path = os.path.join(
                    clip_output_path,
                    f"{frame_key}_mask_stream{stream_key}_hand_{hand_side.name}.png",
                )
                imageio.imwrite(mask_path, mask_out)

        for stream_key, merged in merged_object_masks.items():
            merged_out = resize_image(
                np.rot90(merged.astype(np.uint8) * 255, k=3),
                stream_output_sizes.get(stream_key),
                is_mask=True,
            )
            mask_path = os.path.join(
                clip_output_path,
                f"{frame_key}_mask_stream{stream_key}_object.png",
            )
            imageio.imwrite(mask_path, merged_out)

        if (frame_id + 1) % 25 == 0 or (frame_id + 1) == total_frames:
            log.info(f"{clip_name}: {frame_id + 1}/{total_frames} frames")

    tar.close()

    # Delete the tar now that processing is complete.
    try:
        os.remove(clip_path)
        log.info(f"Deleted {clip_path}")
    except OSError as e:
        log.warning(f"Could not delete {clip_path}: {e}")

    return clip_name


def get_clip_index(filename: str) -> int:
    return int(filename.split(".tar")[0].split("clip-")[1])


def already_processed(clip_name: str, output_dir: str) -> bool:
    """True if the clip's output directory exists and contains at least one file."""
    clip_out = os.path.join(output_dir, clip_name)
    return os.path.isdir(clip_out) and bool(os.listdir(clip_out))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process HOT3D clips into masks and images (multi-node safe).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--clip-start", type=int, default=191,
                        help="First clip index to process (inclusive). Default: 191 (resumes after pre-processed 0-190).")
    parser.add_argument("--clip-end", type=int, default=-1,
                        help="Last clip index to process (inclusive). -1 = no upper bound.")
    parser.add_argument("--tar-dir", type=str, default="diffsynth/data/tar_recv",
                        help="Directory for staging downloaded tars. Use a separate dir per node.")
    parser.add_argument("--output-dir", type=str, default="diffsynth/data/training_masks",
                        help="Directory for output masks (shared across nodes).")
    parser.add_argument("--images-dir", type=str, default="diffsynth/data/training_images",
                        help="Directory for output images (shared across nodes). Empty string to skip.")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of parallel processing workers.")
    parser.add_argument("--max-stored-clips", type=int, default=4,
                        help="Max tars allowed on disk in tar-dir at once (download throttle).")
    parser.add_argument("--mano-model-dir", type=str,
                        default="./diffsynth/data/mano/models/",
                        help="Path to MANO model directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()

    clips_dir     = args.tar_dir
    output_dir    = args.output_dir
    images_dir    = args.images_dir if args.images_dir else None
    mano_model_dir = args.mano_model_dir
    hand_type      = "mano"
    clip_start     = args.clip_start
    clip_end       = args.clip_end
    num_workers    = args.num_workers
    max_stored_clips = args.max_stored_clips
    undistort      = False  # kept False throughout; not exposed as CLI arg

    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    if images_dir:
        os.makedirs(images_dir, exist_ok=True)

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

    # --- Pre-populate already-done set from output_dir ---
    # Any clip whose output subdir exists and is non-empty is considered done.
    processed_clips: set[str] = set()
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if already_processed(name, output_dir):
                processed_clips.add(f"{name}.tar")
    log.info(f"Already processed: {len(processed_clips)} clips (will skip)")

    # --- Clean up stale tars that are already processed ---
    # These were left over from previous interrupted runs.
    stale = [
        p for p in os.listdir(clips_dir)
        if p.endswith(".tar") and p in processed_clips
    ]
    for p in stale:
        stale_path = os.path.join(clips_dir, p)
        try:
            os.remove(stale_path)
            log.info(f"Cleaned up stale tar: {stale_path}")
        except OSError as e:
            log.warning(f"Could not remove stale tar {stale_path}: {e}")

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
                # --- Download loop: fill up to max_stored_clips ---
                while more_available and (clip_end < 0 or download_index <= clip_end):
                    on_disk = len([p for p in os.listdir(clips_dir) if p.endswith(".tar")])
                    if on_disk >= max_stored_clips:
                        log.debug(f"[download] disk full ({on_disk}/{max_stored_clips}), pausing")
                        break

                    # Skip clips already processed (avoids re-downloading)
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

                # --- Submit ready tars to workers ---
                in_flight = set(futures.values())
                pending_files = [
                    p for p in os.listdir(clips_dir)
                    if p.endswith(".tar")
                    and p not in processed_clips
                    and p not in in_flight
                    and get_clip_index(p) >= clip_start
                    and (clip_end < 0 or get_clip_index(p) <= clip_end)
                ]
                log.debug(f"[schedule] in_flight={len(in_flight)} pending={len(pending_files)} "
                          f"processed={len(processed_clips)}")
                for clip in pending_files:
                    future = executor.submit(
                        process_clip,
                        os.path.join(clips_dir, clip),
                        undistort,
                        output_dir,
                        images_dir,
                    )
                    futures[future] = clip
                    log.info(f"Submitted {clip}")

                # --- Collect completed futures ---
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
