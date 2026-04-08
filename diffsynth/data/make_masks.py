#!/usr/bin/env python3
import argparse
import os
import tarfile
from typing import Any, Dict, Optional

import clip_util
import imageio
import numpy as np

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

        for stream_id in image_streams:
            stream_key = str(stream_id)
            camera_model = cameras[stream_id]

            if undistort:
                camera_model_orig = camera_model
                camera_model = clip_util.convert_to_pinhole_camera(camera_model)

            # Object masks — decoded directly from pre-computed RLE in the annotation JSON
            if objects is not None:
                for instance_list in objects.values():
                    for inst_idx, instance in enumerate(instance_list):
                        bop_id = int(instance["object_bop_id"])
                        mask_rle = instance.get("masks_amodal", {}).get(stream_key)
                        if mask_rle is None:
                            continue
                        mask = clip_util.decode_binary_mask_rle(mask_rle)
                        mask_out = (mask.astype(np.uint8) * 255)
                        mask_path = os.path.join(
                            clip_output_path,
                            f"{frame_key}_mask_stream{stream_key}_obj_bop{bop_id}_inst{inst_idx}.png"
                        )
                        imageio.imwrite(mask_path, mask_out)

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", type=str, required=True, help="Path to folder with clips.")
    parser.add_argument("--mano_model_dir", type=str, default="", help="Path to MANO model folder.")
    parser.add_argument("--undistort", action="store_true", help="Whether to undistort images.")
    parser.add_argument("--hand_type", type=str, default="umetrack", choices=["umetrack", "mano"])
    parser.add_argument("--clip_start", type=int, default=0)
    parser.add_argument("--clip_end", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mano_model = MANOHandModel(args.mano_model_dir) if args.hand_type == "mano" else None

    clips = sorted(p for p in os.listdir(args.clips_dir) if p.endswith(".tar"))
    print("Processing clips...")
    for clip in tqdm(clips):
        clip_id = int(clip.split(".tar")[0].split("clip-")[1])
        if clip_id < args.clip_start or (args.clip_end >= 0 and clip_id > args.clip_end):
            continue

        process_clip(
            clip_path=os.path.join(args.clips_dir, clip),
            hand_type=args.hand_type,
            mano_model=mano_model,
            undistort=args.undistort,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
