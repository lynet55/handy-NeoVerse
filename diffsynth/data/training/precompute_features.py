"""Pre-extract frozen backbone features for all dataset samples.

Saves token_list tensors to disk so training can skip the backbone entirely.
This is the most impactful optimisation — the backbone accounts for ~85% of
per-step time and its outputs never change during training.

Usage:
    python -m diffsynth.data.training.precompute_features \
        --data_root diffsynth/data/training_data \
        --output_dir diffsynth/data/training_features \
        --model_path models/NeoVerse/reconstructor.ckpt

Storage estimate:
    Each frame produces 4 intermediate token tensors of shape
    [1, N, 2048] in bfloat16 (~8 MB total per frame).
    Adjust --frame_stride to control total cache size.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.models.model_manager import ModelManager

STREAMS = ["stream1201-1", "stream1201-2"]


def dbg(msg: str):
    print(f"[PRE {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute backbone features")
    parser.add_argument("--data_root", default="diffsynth/data/training_data")
    parser.add_argument("--output_dir", default="diffsynth/data/training_features")
    parser.add_argument("--model_path", default="models/NeoVerse/reconstructor.ckpt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frame_stride", type=int, default=5,
                        help="Keep every Nth frame (1 = all)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load model
    dbg(f"Loading model from {args.model_path} ...")
    manager = ModelManager()
    manager.load_model(args.model_path, device=args.device, torch_dtype=torch.bfloat16)
    reconstructor: WorldMirror = manager.fetch_model("reconstructor")
    reconstructor.eval()
    for p in reconstructor.parameters():
        p.requires_grad = False
    backbone = reconstructor.visual_geometry_transformer
    dbg("Model loaded.")

    # Iterate over clips
    clip_paths = sorted(data_root.glob("clip-*.npz"))
    dbg(f"Found {len(clip_paths)} clips, stride={args.frame_stride}")

    total_frames = 0
    t0 = time.time()

    for ci, npz_path in enumerate(clip_paths):
        clip_name = npz_path.stem
        npz = np.load(str(npz_path), mmap_mode="r")
        n_frames = next(npz[k].shape[0] for k in npz.files if k.startswith("images_"))

        for stream in STREAMS:
            img_key = f"images_{stream}"
            if img_key not in npz.files:
                continue

            for frame_idx in range(0, n_frames, args.frame_stride):
                out_file = out_root / f"{clip_name}_{stream}_f{frame_idx}.pt"
                if out_file.exists():
                    continue

                # Load single frame -> [1, 1, 3, H, W]
                image = torch.tensor(
                    npz[img_key][frame_idx], dtype=torch.float32,
                ).permute(2, 0, 1) / 255.0
                imgs = image.unsqueeze(0).unsqueeze(0).to(args.device)

                with torch.no_grad(), torch.amp.autocast(args.device, dtype=torch.bfloat16):
                    token_list, patch_start_idx, _, _ = backbone(imgs, use_motion=False)

                # Save as list of CPU bfloat16 tensors + metadata
                data = {
                    "token_list": [t.cpu() for t in token_list],
                    "patch_start_idx": patch_start_idx,
                    "clip_name": clip_name,
                    "stream": stream,
                    "frame_idx": frame_idx,
                }
                torch.save(data, out_file)
                total_frames += 1

        if (ci + 1) % 10 == 0 or ci == len(clip_paths) - 1:
            elapsed = time.time() - t0
            dbg(f"  clip {ci+1}/{len(clip_paths)} | {total_frames} frames | {elapsed:.0f}s")

    elapsed = time.time() - t0
    dbg(f"Done. {total_frames} frames extracted in {elapsed:.0f}s")
    dbg(f"Saved to {out_root}")


if __name__ == "__main__":
    main()
