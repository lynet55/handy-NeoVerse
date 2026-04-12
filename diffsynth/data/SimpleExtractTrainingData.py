#!/usr/bin/env python3
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
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
from tqdm import tqdm


# Target size for saved images and masks. Set to None to skip resizing.
OUTPUT_SIZE: Optional[tuple] = (280, 280)  # (width, height)

TAR_DIR = Path("diffsynth/data/tar_recv")


def resize_image(arr: np.ndarray, size: Optional[tuple], is_mask: bool = False) -> np.ndarray:
    """Resize a numpy image array to `size` (width, height).
    Uses NEAREST resampling for masks to preserve binary values,
    and LANCZOS for RGB images to retain quality.
    """
    if size is None:
        return arr
    resample = Image.NEAREST if is_mask else Image.LANCZOS
    return np.array(Image.fromarray(arr).resize(size, resample=resample))


def delete_tar(clip_name: str) -> None:
    tar_path = TAR_DIR / f"{clip_name}.tar"
    if tar_path.exists():
        tar_path.unlink()
        print(f"Deleted: {tar_path}")


def process_clip(
    clip_path: str,
    hand_type: str,
    mano_model_dir: Optional[str],
    undistort: bool,
    output_dir: str,
    images_dir: Optional[str] = None,
) -> str:
    """Process a single clip. Returns the clip name on success.

    Note: MANOHandModel is instantiated per-process to avoid pickling issues
    when using ProcessPoolExecutor.
    """
    mano_model = MANOHandModel(mano_model_dir) if hand_type == "mano" and mano_model_dir else None

    tar = tarfile.open(clip_path, mode="r")
    clip_name = os.path.basename(clip_path).split(".tar")[0]
    clip_output_path = os.path.join(output_dir, clip_name)
    os.makedirs(clip_output_path, exist_ok=True)

    hand_shape: Optional[HandShapeCollection] = clip_util.load_hand_shape(tar)

    print(f"Processing clip {clip_name}")
    for frame_id in tqdm(range(clip_util.get_number_of_frames(tar)), desc=clip_name, leave=False):
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
                stream_output_dir = os.path.join(images_dir, f"stream{stream_key}", clip_name)
                os.makedirs(stream_output_dir, exist_ok=True)
                imageio.imwrite(os.path.join(stream_output_dir, f"{frame_key}.png"), image)

            if objects is not None:
                for instance_list in objects.values():
                    for instance in instance_list:
                        mask_rle = instance.get("masks_amodal", {}).get(stream_key)
                        if mask_rle is None:
                            continue
                        mask = clip_util.decode_binary_mask_rle(mask_rle)
                        if stream_key not in merged_object_masks:
                            merged_object_masks[stream_key] = mask
                        else:
                            merged_object_masks[stream_key] = merged_object_masks[stream_key] | mask

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

    delete_tar(clip_name=clip_name)
    return clip_name


def get_clip_index(filename: str) -> int:
    return int(filename.split(".tar")[0].split("clip-")[1])


def main() -> None:
    clips_dir = "./diffsynth/data/tar_recv/"
    mano_model_dir = "./diffsynth/data/mano/models/"
    output_dir = "./diffsynth/data/training_masks/"
    images_dir = "./diffsynth/data/training_images/"
    hand_type = "mano"
    clip_start = 0
    clip_end = -1
    undistort = False
    num_workers = 2       # Parallel processing workers
    max_stored_clips = 4  # Max clips on disk at any one time (download + processing buffer)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    download_index = clip_start
    more_available = True
    processed_clips: set[str] = set()

    print(f"Starting pipeline (max {max_stored_clips} clips on disk at a time)...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures: dict = {}

        while more_available or futures:
            # Fill up to max_stored_clips by downloading as slots free up
            while more_available and (clip_end < 0 or download_index <= clip_end):
                on_disk = len([p for p in os.listdir(clips_dir) if p.endswith(".tar")])
                if on_disk >= max_stored_clips:
                    break
                print(f"Downloading clip {download_index}...")
                more_available = tar_hf_import(download_index)
                if more_available:
                    download_index += 1

            # Submit any downloaded-but-not-yet-submitted clips for processing
            pending_files = [
                p for p in os.listdir(clips_dir)
                if p.endswith(".tar")
                and p not in processed_clips
                and p not in {futures[f] for f in futures}
                and get_clip_index(p) >= clip_start
                and (clip_end < 0 or get_clip_index(p) <= clip_end)
            ]
            for clip in pending_files:
                future = executor.submit(
                    process_clip,
                    os.path.join(clips_dir, clip),
                    hand_type,
                    mano_model_dir,
                    undistort,
                    output_dir,
                    images_dir,
                )
                futures[future] = clip
                print(f"Submitted: {clip}")

            # Collect any completed futures to free up disk slots for new downloads
            done = [f for f in futures if f.done()]
            for future in done:
                clip = futures.pop(future)
                try:
                    clip_name = future.result()
                    processed_clips.add(clip)
                    print(f"Done: {clip_name}")
                except Exception as exc:
                    print(f"ERROR processing {clip}: {exc}")

            if futures:
                # Block until at least one finishes, freeing a disk slot for the next download
                wait(futures, timeout=2, return_when=FIRST_COMPLETED)

    print("All clips processed.")


if __name__ == "__main__":
    main()
