#!/usr/bin/env python3
import argparse
import os
import tarfile
from typing import Any, Dict, Optional

import clip_util
import imageio
import numpy as np
from PIL import Image

# Target size for saved images and masks. Set to None to skip resizing.
OUTPUT_SIZE: Optional[tuple] = (280, 280)  # (width, height)

# Directory where sentinel files are written to signal a clip is fully processed.
# The dataset watches this directory before ingesting a clip.
SENTINEL_DIR = "./diffsynth/data/ready_sentinels/"


def resize_image(arr: np.ndarray, size: Optional[tuple], is_mask: bool = False) -> np.ndarray:
    """Resize a numpy image array to `size` (width, height).
    Uses NEAREST resampling for masks to preserve binary values,
    and LANCZOS for RGB images to retain quality.
    """
    if size is None:
        return arr
    resample = Image.NEAREST if is_mask else Image.LANCZOS
    return np.array(Image.fromarray(arr).resize(size, resample=resample))


from tar_hf_import import tar_hf_import
from hand_tracking_toolkit import rasterizer
from hand_tracking_toolkit.dataset import HandShapeCollection
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
from tqdm import tqdm


def mark_clip_ready(clip_name: str, sentinel_dir: str) -> None:
    """Write a zero-byte sentinel file so the dataset knows this clip is complete."""
    os.makedirs(sentinel_dir, exist_ok=True)
    sentinel_path = os.path.join(sentinel_dir, f"{clip_name}.ready")
    open(sentinel_path, "w").close()
    print(f"Sentinel written: {sentinel_path}")


def process_clip(
    clip_path: str,
    hand_type: str,
    mano_model: Optional[MANOHandModel],
    undistort: bool,
    output_dir: str,
    images_dir: Optional[str] = None,
    sentinel_dir: Optional[str] = None,
) -> None:
    tar = tarfile.open(clip_path, mode="r")

    clip_name = os.path.basename(clip_path).split(".tar")[0]
    clip_output_path = os.path.join(output_dir, clip_name)
    os.makedirs(clip_output_path, exist_ok=True)

    hand_shape: Optional[HandShapeCollection] = clip_util.load_hand_shape(tar)

    print(f"Processing clip {clip_name}")
    for frame_id in tqdm(range(clip_util.get_number_of_frames(tar))):
        frame_key = f"{frame_id:06d}"

        cameras, _ = clip_util.load_cameras(tar, frame_key)
        image_streams = sorted(cameras.keys(), key=lambda x: int(x.split("-")[0]))

        hands: Optional[Dict[str, Any]] = clip_util.load_hand_annotations(tar, frame_key)
        objects: Optional[Dict[str, Any]] = clip_util.load_object_annotations(tar, frame_key)

        hand_meshes = {}
        if hand_shape is not None and hands is not None:
            hand_meshes = clip_util.get_hand_meshes(hands, hand_shape, hand_type, mano_model)

        merged_object_masks: Dict[str, Optional[np.ndarray]] = {}
        # Tracks the final (w, h) to use for masks, keyed by stream_key.
        # Derived from the actual image so masks are always resized identically.
        stream_output_sizes: Dict[str, Optional[tuple]] = {}

        for stream_id in image_streams:
            stream_key = str(stream_id)
            camera_model = cameras[stream_id]

            if undistort:
                camera_model_orig = camera_model
                camera_model = clip_util.convert_to_pinhole_camera(camera_model)

            # Load image for every stream so we always know the true output size.
            # The rotated+resized image dimensions are the ground truth for mask alignment.
            image = clip_util.load_image(tar, frame_key, stream_key)
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            image = np.rot90(image, k=3)
            image = resize_image(image, OUTPUT_SIZE, is_mask=False)
            # image.shape is (H, W, C); PIL resize wants (W, H)
            stream_output_sizes[stream_key] = (image.shape[1], image.shape[0])

            if images_dir is not None:
                stream_output_dir = os.path.join(images_dir, f"stream{stream_key}", clip_name)
                os.makedirs(stream_output_dir, exist_ok=True)
                imageio.imwrite(os.path.join(stream_output_dir, f"{frame_key}.png"), image)

            # Object masks — merge all instances into one mask per stream
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

            # Hand masks — resized to exactly match the image for this stream
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

        # Write merged object masks — each resized to match its stream's image
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

    # All frames written — signal to the dataset that this clip is safe to ingest.
    if sentinel_dir is not None:
        mark_clip_ready(clip_name, sentinel_dir)


def main() -> None:
    clips_dir = "./diffsynth/data/tar_recv/"
    mano_model_dir = "./diffsynth/data/mano/models/"
    output_dir = "./diffsynth/data/training_masks/"
    images_dir = "./diffsynth/data/training_images/"
    hand_type = "mano"
    clip_start = 0
    clip_end = -1
    undistort = False

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(SENTINEL_DIR, exist_ok=True)

    mano_model = MANOHandModel(mano_model_dir) if hand_type == "mano" else None

    processed_clips = set()
    download_index = 1286
    more_available = True

    print("Starting download+process loop...")
    while True:
        if more_available:
            print(f"Downloading clip {download_index}...")
            more_available = tar_hf_import(download_index)
            if more_available:
                download_index += 1

        all_clips = sorted(p for p in os.listdir(clips_dir) if p.endswith(".tar"))
        pending = [
            p for p in all_clips
            if p not in processed_clips
            and int(p.split(".tar")[0].split("clip-")[1]) >= clip_start
            and (clip_end < 0 or int(p.split(".tar")[0].split("clip-")[1]) <= clip_end)
        ]

        if pending:
            for clip in tqdm(pending, desc="Processing clips"):
                process_clip(
                    clip_path=os.path.join(clips_dir, clip),
                    hand_type=hand_type,
                    mano_model=mano_model,
                    undistort=undistort,
                    output_dir=output_dir,
                    images_dir=images_dir,
                    sentinel_dir=SENTINEL_DIR,
                )
                processed_clips.add(clip)
        elif not more_available:
            print("All clips processed.")
            done_path = os.path.join(SENTINEL_DIR, "ALL_DONE")
            open(done_path, "w").close()
            print(f"Global DONE sentinel written: {done_path}")
            break


if __name__ == "__main__":
    main()
