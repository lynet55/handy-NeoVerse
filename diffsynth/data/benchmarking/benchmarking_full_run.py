import numpy as np
import torch
import os
import math
import json

from datetime import datetime
from torch.utils.data import DataLoader
from dataclasses import dataclass

from diffsynth.auxiliary_models.worldmirror.models.heads.dense_head import DPTHead
from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.models.model_manager import ModelManager

from diffsynth.data.benchmarking.benchmarking_dataset import BenchmarkingDataset
from diffsynth.data.benchmarking.benchmark_full import BenchmarkEvaluator


# TODO: Copied TrainConfig and removed some lines
@dataclass
class TestConfig:
    # Model Architecture
    img_shape = (280,280)
    patch_size: int = 14
    embed_dim: int = 1024

    num_classes: int = 4

    # Dataloader
    batch_size: int = 2
    num_workers: int = 2
    pin_memory: bool = True

    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoint
    save_model_path_prefix: str = "models/NeoVerse/hand_seg_model_opt"
    resume_from: str = "latest"

    # Output
    results_path: str = "benchmark_results.json"

def resolve_resume_path(cfg: TestConfig):
    if cfg.resume_from is None:
        return None
    if cfg.resume_from == "latest":
        return f"{cfg.save_model_path_prefix}_latest.ckpt"
    if cfg.resume_from == "best":
        return f"{cfg.save_model_path_prefix}_best.ckpt"
    return cfg.resume_from

def load_model(cfg: TestConfig) -> WorldMirror:
    model = WorldMirror(
        img_size=cfg.img_shape[0],
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
    ).to(cfg.device)

    path = resolve_resume_path(cfg)
    if path is None or not os.path.exists(path):
        print(f"[load_model] No checkpoint found at '{path}', using random weights.")
        return model

    ckpt = torch.load(path, map_location=cfg.device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"[load_model] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[load_model] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"[load_model] Loaded from '{path}'")
    return model

def build_views(images: torch.Tensor, device: str) -> dict:
    images = images.to(device)
    B, S = images.shape[:2]
    return {
        "img": images,
        "timestamp": torch.arange(S, dtype=torch.float32, device=device).unsqueeze(0).expand(B,-1),
        "is_static": torch.ones(B, S, dtype=torch.bool, device=device),
    }

def test():
    cfg = TestConfig()

    # TODO: Is this how to do this?
    model = load_model(cfg)
    model.eval()

    # TODO: Make sure data root is correct and use dataset that makes one clip one sample
    test_dataset = BenchmarkingDataset( 
        data_root = "diffsynth/data/test_data",
        streams=["stream1201-1", "stream1201-2"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )

    benchmarker = BenchmarkEvaluator(
        device=cfg.device,
        num_classes=cfg.num_classes,
        run_seg=True, 
        run_render=True,
    )

    with torch.no_grad():
        for batch_idx, (images, gt_mask, clip_names, streams) in enumerate(test_loader):

            views = build_views(images, cfg.device)
            preds = model(views)
            # TODO: Check all shapes of masks

            seg_logits = preds["seg_labels"]
            B, S, H, W, C = seg_logits.shape

            pred_mask = seg_logits.reshape(B*S,H,W,C).permute(0,3,1,2).contiguous()
            gt_flat = gt_mask.reshape(B*S, cfg.num_classes, H, W).to(cfg.device)

            # TODO: Correct way to get rendered images?
            rendered_rgb, rendered_depths, rendered_alphas = model.gs_renderer.rasterizer.forward(
                render_splats=preds["splats"],
                render_viewmats=preds["rendered_extrinsics"],
                render_Ks=preds["rendered_intrinsics"],
                render_timestamps=preds["rendered_timestamps"],
                sh_degree=model.gs_renderer.sh_degree,
                width=W,
                height=H,
            )

            # rendered_colors has shape [B, S, H, W, 3]. 
            # Permute to [B, S, 3, H, W] to match gt_rgb shape
            pred_rgb = rendered_rgb.permute(0,1,4,2,3).contiguous()
            gt_rgb = images.to(cfg.device)

            benchmarker.update(
                pred_mask=pred_mask, 
                gt_mask=gt_mask, 
                pred_rgb=pred_rgb.reshape(-1,3,H,W), 
                gt_rgb=gt_rgb.reshape(-1,3,H,W),
            )

            if (batch_idx + 1) % 10 == 0:
                print(f" [{batch_idx} batches / {(batch_idx + 1) * cfg.batch_size} clips]"
                      f"last: {clip_names[0]} / {streams[0]}")


        results = benchmarker.compute()

        class_names = ["right_hand", "left_hand", "object", "background"]
        print("\n=== Benchmark Results ===")
        print(f"  mIoU:           {results['mIoU']:.4f}")
        print(f"  Pixel Accuracy: {results['Pixel_Accuracy']:.4f}")
        print(f"  Boundary F1:    {results['Boundary_F1']:.4f}")
        print("  Per-class IoU:")
        for name, iou in zip(class_names, results["IoU_Per_class"]):
            print(f"    {name:>15s}: {iou:.4f}")
            print("  Per-class Pixel Accuracy:")
        for name, acc in zip(class_names, results["Pixel_Acc_per_class"]):
            print(f"    {name:>15s}: {acc:.4f}")
       
       # TODO: Correct?
        with open(cfg.results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to '{cfg.results_path}'")

if __name__ == "__main__":
    test()
