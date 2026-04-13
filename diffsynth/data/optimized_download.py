#!/usr/bin/env python3
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

TAR_DIR = Path("diffsynth/data/tar_recv")

# Worker-process globals populated by the initializer. Avoids rebuilding
# MANOHandModel once per clip (it was being reconstructed on every call).
_WORKER_MANO_MODEL: Optional[MANOHandModel] = None
_WORKER_HAND_TYPE: Optional[str] = None


def setup_worker(log_queue: multiprocessing.Queue, hand_type: str, mano_model_dir: Optional[str], log_level: int = logging.INFO) -> None:
    """Initializer run in each worker process.

    - Routes logs to the shared queue.
    - Builds the MANO model once per worker and stashes it as a module global,
      so subsequent process_clip calls reuse it.
    """
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
    # PIL copies non-contiguous arrays internally; do it explicitly so we
    # avoid a hidden allocation path and make the intent clear.
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return np.array(Image.fromarray(arr).resize(size, resample=resample))


def delete_tar(clip_name: str) -> None:
    log = logging.getLogger(__name__)
    tar_path = TAR_DIR / f"{clip_name}.tar"
    if tar_path.exists():
        tar_path.unlink()
        log.info(f"Deleted {tar_path}")


def process_clip(
    clip_path: str,
    undistort: bool,
    output_dir: str,
    images_dir: Optional[str] = None,
) -> str:
    log = logging.getLogger(__name__)
    mano_model = _WORKER_MANO_MODEL
    hand_type = _WORKER_HAND_TYPE
    log.debug(f"[worker] start clip_path={clip_path} mano_loaded={mano_model is not None}")

    # Read the whole tar into memory once. For clips that fit comfortably in
    # RAM this eliminates repeated disk seeks across per-frame load_* calls.
    # tarfile over a BytesIO does all seeks in-memory. If your clips are too
    # large for this, replace with: open(clip_path, "rb", buffering=1<<20).
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

    # Cache per-clip output dirs per stream so we only makedirs once per stream
    # instead of once per frame per stream.
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
                            # In-place OR avoids allocating a new array each merge.
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
                    f"{frame_key}_mask_stream{stream_key}_hand_{hand_side.name}.png"
                )
                imageio.imwrite(mask_path, mask_out)

        for stream_key, merged in merged_object_masks.items():
            mask_path = os.path.join(
                clip_output_path,
                f"{frame_key}_mask_stream{stream_key}_object.png"
            )
            merged_out = resize_image(
                np.rot90(merged.astype(np.uint8) * 255, k=3),
                stream_output_sizes.get(stream_key),
                is_mask=True,
            )
            imageio.imwrite(mask_path, merged_out)

        if (frame_id + 1) % 25 == 0 or (frame_id + 1) == total_frames:
            log.info(f"{clip_name}: {frame_id + 1}/{total_frames} frames")

    tar.close()
    delete_tar(clip_name=clip_name)
    return clip_name


def get_clip_index(filename: str) -> int:
    return int(filename.split(".tar")[0].split("clip-")[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Process hand/object clips into masks and images.")
    parser.add_argument("--clip-start", type=int, default=0, help="First clip index to process (inclusive).")
    parser.add_argument("--clip-end", type=int, default=-1, help="Last clip index to process (inclusive). -1 means no upper bound.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()

    clips_dir = "./diffsynth/data/tar_recv/"
    mano_model_dir = "./diffsynth/data/mano/models/"
    output_dir = "./diffsynth/data/training_masks/"
    images_dir = "./diffsynth/data/training_images/"
    hand_type = "mano"
    clip_start = args.clip_start
    clip_end = args.clip_end
    undistort = False
    num_workers = 20
    max_stored_clips = 24

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    log_level = logging.DEBUG if args.debug else logging.INFO

    # --- Logging setup: single listener in the parent, queue-fed from workers ---
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

    download_index = clip_start
    more_available = True
    processed_clips: set[str] = set()

    log.info(f"Starting pipeline (max {max_stored_clips} clips on disk, {num_workers} workers)")

    try:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=setup_worker,
            initargs=(log_queue, hand_type, mano_model_dir, log_level),
        ) as executor:
            futures: dict = {}

            while more_available or futures:
                while more_available and (clip_end < 0 or download_index <= clip_end):
                    on_disk = len([p for p in os.listdir(clips_dir) if p.endswith(".tar")])
                    log.debug(f"[download] on_disk={on_disk} max={max_stored_clips} next_idx={download_index} more_available={more_available}")
                    if on_disk >= max_stored_clips:
                        log.debug(f"[download] disk full ({on_disk}/{max_stored_clips}), pausing downloads")
                        break
                    log.info(f"Downloading clip {download_index}")
                    more_available = tar_hf_import(download_index)
                    log.debug(f"[download] tar_hf_import({download_index}) -> more_available={more_available}")
                    if more_available:
                        download_index += 1

                # Build the "in-flight" set once instead of rebuilding it per candidate.
                in_flight = set(futures.values())

                pending_files = [
                    p for p in os.listdir(clips_dir)
                    if p.endswith(".tar")
                    and p not in processed_clips
                    and p not in in_flight
                    and get_clip_index(p) >= clip_start
                    and (clip_end < 0 or get_clip_index(p) <= clip_end)
                ]
                log.debug(f"[schedule] in_flight={len(in_flight)} pending={len(pending_files)} processed={len(processed_clips)}")
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

                done = [f for f in futures if f.done()]
                if done:
                    log.debug(f"[schedule] {len(done)} future(s) completed this tick")
                for future in done:
                    clip = futures.pop(future)
                    try:
                        clip_name = future.result()
                        processed_clips.add(clip)
                        log.info(f"Done {clip_name}")
                    except Exception as exc:
                        log.exception(f"ERROR processing {clip}: {exc}")

                if futures:
                    wait(futures, timeout=2, return_when=FIRST_COMPLETED)

        log.info("All clips processed.")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()