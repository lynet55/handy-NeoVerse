#!/usr/bin/env python3
import argparse
import os
import tarfile
from typing import Any, Dict, Optional

import clip_util
import imageio
import numpy as np

from tar_hf_import import tar_hf_import
from hand_tracking_toolkit import rasterizer
from hand_tracking_toolkit.dataset import HandShapeCollection
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
from tqdm import tqdm


def process_clip(
    clip_path: str,
    hand_type: str,
    mano_model: Optional[MANOHandModel],
    undistort: bool,
    output_dir: str,
    images_dir: Optional[str] = None,
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

        for stream_id in image_streams:
            stream_key = str(stream_id)
            camera_model = cameras[stream_id]

            if undistort:
                camera_model_orig = camera_model
                camera_model = clip_util.convert_to_pinhole_camera(camera_model)

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

            # Hand masks
            for hand_side, hand_mesh in hand_meshes.items():
                _, mask, _ = rasterizer.rasterize_mesh(
                    verts=hand_mesh.vertices,
                    faces=hand_mesh.faces,
                    vert_normals=hand_mesh.vertex_normals,
                    camera=camera_model,
                )

                mask_out = np.rot90(mask.astype(np.uint8) * 255, k=3)
                mask_path = os.path.join(
                    clip_output_path,
                    f"{frame_key}_mask_stream{stream_key}_hand_{hand_side.name}.png"
                )
                imageio.imwrite(mask_path, mask_out)

            # Image extraction
            if images_dir is not None:
                stream_output_dir = os.path.join(images_dir, f"stream{stream_key}", clip_name)
                os.makedirs(stream_output_dir, exist_ok=True)
                image = clip_util.load_image(tar, frame_key, stream_key)
                if image.ndim == 2:
                    image = np.stack([image, image, image], axis=-1)
                image = np.rot90(image, k=3)
                imageio.imwrite(os.path.join(stream_output_dir, f"{frame_key}.png"), image)

        # Write merged object masks (one per stream)
        for stream_key, merged in merged_object_masks.items():
            mask_path = os.path.join(
                clip_output_path,
                f"{frame_key}_mask_stream{stream_key}_object.png"
            )
            imageio.imwrite(mask_path, np.rot90(merged.astype(np.uint8) * 255, k=3))


def main() -> None:
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--clips_dir", type=str, required=True, help="Path to folder with clips.")
    #parser.add_argument("--mano_model_dir", type=str, default="", help="Path to MANO model folder.")
    #parser.add_argument("--undistort", action="store_true", help="Whether to undistort images.")
    #parser.add_argument("--hand_type", type=str, default="umetrack", choices=["umetrack", "mano"])
    #parser.add_argument("--clip_start", type=int, default=0)
    #parser.add_argument("--clip_end", type=int, default=-1)
    #parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder.")
    #parser.add_argument("--images_dir", type=str, default="", help="If set, extract raw images into this folder.")
    #args = parser.parse_args()

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

    mano_model = MANOHandModel(mano_model_dir) if hand_type == "mano" else None
    for i in range(20): 
        tar_hf_import(i)

    clips = sorted(p for p in os.listdir(clips_dir) if p.endswith(".tar"))
    print("Processing clips...")
    for clip in tqdm(clips):
        clip_id = int(clip.split(".tar")[0].split("clip-")[1])
        if clip_id < clip_start or (clip_end >= 0 and clip_id > clip_end):
            continue

        process_clip(
            clip_path=os.path.join(clips_dir, clip),
            hand_type=hand_type,
            mano_model=mano_model,
            undistort=undistort,
            output_dir=output_dir,
            images_dir=images_dir,
        )


if __name__ == "__main__":
    main()
