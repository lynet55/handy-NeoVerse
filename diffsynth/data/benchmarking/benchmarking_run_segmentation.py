import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass



from diffsynth.data.benchmarking.benchmarking_metrics_segmentation import evaluate as benchmarking_evaluate
from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.models.model_manager import ModelManager
from diffsynth.data.SimpleHandObjectSegmentationDataset import HandObjectSegmentationDataset

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Test-set evaluation for hand/object segmentation.")

    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of segmentation classes (default: 4)")
    parser.add_argument("--data_root", type=str, default="diffsynth/data/test_data",
                        help="Path to test dataset root")
    parser.add_argument("--checkpoint", type=str, default="best",
                        choices=["best", "latest"],
                        help="Which checkpoint to load: 'best' or 'latest'")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from TestConfig")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device from TestConfig (e.g. 'cpu', 'cuda')")
    parser.add_argument("--bf1_tolerance", type=int, default=2,
                        help="Pixel tolerance for Boundary F1 (default: 2)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_test_dataset(model, test_loader, cfg, num_classes, bf1_tolerance):
    model.head.eval()
    all_batch_metrics = []

    for batch in test_loader:
        images, gt_mask, _, _ = batch
        images = images.to(cfg.device, non_blocking=True)
        gt_mask = gt_mask.to(cfg.device, non_blocking=True)
        
        # Inference
        classifications, recon_rgb = model.forward(images)
        # Check if permute is actually needed (standard is B,C,H,W)
        if classifications.shape[1] != num_classes:
            classifications = classifications.permute(0, 3, 1, 2)

        # Vectorized call: processes the whole batch at once on GPU
        batch_results = benchmarking_evaluate(
            classifications, 
            gt_mask,
            num_classes=num_classes,
            bf1_tolerance=bf1_tolerance,
            pred_rgb=recon_rgb,
            gt_rgb=images
        )
        all_batch_metrics.append(batch_results)

    # Aggregate across batches
    aggregated = {}
    for key in all_batch_metrics[0].keys():
        values = [m[key] for m in all_batch_metrics]
        if isinstance(values[0], list):
            # Handles per-class lists
            aggregated[key] = np.mean(values, axis=0).tolist()
        else:
            aggregated[key] = float(np.mean(values))

    return aggregated, all_batch_metrics

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_class_names(num_classes):
    """
    Default names for known configs, generic fallback for anything else.
    Add new cases here when you extend to more classes.
    """
    known = {
        4: ["right_hand", "left_hand", "object", "background"],
        3: ["hand", "object", "background"],
        2: ["foreground", "background"],
    }
    return known.get(num_classes, [f"class_{i}" for i in range(num_classes)])


def report(aggregated, class_names):
    dbg("=== Test Results ===")
    dbg(f"  mIoU:           {aggregated['miou']:.4f}")
    dbg(f"  Pixel Accuracy: {aggregated['pixel_accuracy']:.4f}")
    dbg(f"  BF1:            {aggregated['bf1']:.4f}")
    dbg("  Per-class:")
    for c, name in enumerate(class_names):
        dbg(f"    {name:>11}:  "
            f"IoU={aggregated['iou_per_class'][c]:.4f}  "
            f"PA={aggregated['pixel_accuracy_per_class'][c]:.4f}  "
            f"BF1={aggregated['bf1_per_class'][c]:.4f}")


def save_results(aggregated, per_sample, args):
    """
    Saves aggregated metrics and per-sample metrics to JSON.
    Filename includes timestamp and key config for traceability.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = (f"eval_{timestamp}"
            f"_classes{args.num_classes}"
            f"_ckpt{args.checkpoint}")

    results = {
        "meta": {
            "timestamp":    timestamp,
            "num_classes":  args.num_classes,
            "checkpoint":   args.checkpoint,
            "data_root":    args.data_root,
            "bf1_tolerance": args.bf1_tolerance,
            "n_samples":    len(per_sample),
        },
        "aggregated": aggregated,
        "per_sample":  per_sample,
    }

    out_path = Path("eval_results") / f"{stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    dbg(f"Results saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Other helpers
# ---------------------------------------------------------------------------
def dbg(msg: str):
    print(f"[DBG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class TestConfig:
    # Model Architecture
    img_shape = (280,280)
    patch_size: int = 14
    embed_dim: int = 1024
    token_dim: int = 2048
    num_classes: int = 4
    patch_start_idx: int = 5

    # Training Hyperparameters
    batch_size: int = 50
    epochs: int = 10
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    lr_min_factor: float = 0.01  # cosine annealing decays to lr * this factor
    class_weights = torch.tensor([10.0, 10.0, 5.0, 1.0])

    optimizer = torch.optim.AdamW
    grad_clip_norm = 1.0
    # Dataset
    frame_stride: int = 3  # sample every Nth frame (1 = all frames, original behaviour)
    val_fraction: float = 0.1  # fraction of clips held out for validation

    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Reconstructor
    reconstruction_model_path = "models/NeoVerse/reconstructor.ckpt"
    save_model_path_prefix = "models/NeoVerse/hand_seg_model_opt"
    low_vram = False

    # Resume: path to a head checkpoint produced by save_checkpoint /
    # save_step_checkpoint. If None, train from scratch. If the literal
    # string "latest", auto-resolves to <prefix>_latest.ckpt. If "best",
    # resolves to <prefix>_best.ckpt.
    resume_from: str = "latest"

    # Run id: shared between log_dir and per-epoch checkpoint filenames so
    # the TensorBoard run and its checkpoints can be cross-referenced. Two
    # separate default_factory lambdas would generate two timestamps a few
    # microseconds apart — to keep them identical we generate one timestamp
    # in __post_init__ and stamp both fields from it.
    run_id: str = ""
    log_dir: str = ""

    def __post_init__(self):
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not self.log_dir:
            self.log_dir = f"runs/neoverse_seg_opt_{self.run_id}"

    # DataLoader
    num_workers: int = 2
    pin_memory: bool = True

class LeanReconstructor:
    """Loads WorldMirror but only exercises backbone + hand_pred_head.

    The original training pipeline called ``reconstructor(views)`` which:
      * runs the ViT backbone,
      * runs ``prepare_contexts`` (backbone a *second* time when enable_gs),
      * runs every enabled prediction head (depth, pts, normals, GS, cam),
      * runs the Gaussian-splat rasterizer.

    Only ``hand_pred_head`` is trainable and only its output enters the loss.
    This wrapper skips all the rest.
    """

    def __init__(self, cfg: TestConfig):
        self.cfg = cfg

        dbg(f"Loading reconstructor from {cfg.reconstruction_model_path} ...")
        model_manager = ModelManager()
        model_manager.load_model(
            cfg.reconstruction_model_path,
            device=cfg.device,
            torch_dtype=torch.bfloat16,
        )
        self.reconstructor: WorldMirror = model_manager.fetch_model("reconstructor")
        dbg("Reconstructor loaded.")

        # Freeze everything, unfreeze hand_pred_head
        n_train, n_total = 0, 0
        for name, param in self.reconstructor.named_parameters():
            n_total += 1
            if "hand_pred_head" in name:
                param.requires_grad = True
                n_train += 1
            else:
                param.requires_grad = False
        dbg(f"Trainable: {n_train}/{n_total} params.")
        if n_train == 0:
            raise RuntimeError("No parameters matched 'hand_pred_head'.")

        # Keep references for the two components we actually use
        self.backbone = self.reconstructor.visual_geometry_transformer
        self.head = self.reconstructor.hand_pred_head

        # bfloat16 has 7 mantissa bits (eps ~8e-3 at magnitude 1), so Adam
        # updates of order `lr` round to zero and the head silently freezes.
        # Cast head to fp32 so optimizer state and updates stay precise.
        self.head = self.head.float()
        for p in self.head.parameters():
            p.requires_grad = True
        dbg("Head cast to float32 for training.")

        # The hand_pred_head is freshly initialized (not in reconstructor.ckpt)
        # and the upstream DPT features are unbounded, so default Kaiming init
        # on the final projection produces logits with |x| ~1000 — softmax
        # then saturates and per-pixel CE blows up to hundreds. Scaling the
        # final conv down at init makes first-step outputs O(1) instead of
        # O(1000), so training escapes the "shrink the logits" phase that
        # otherwise eats the first ~200 steps before real learning starts.
        with torch.no_grad():
            final_proj = self.head.scratch.output_conv2[2]
            final_proj.weight.mul_(0.01)
            if final_proj.bias is not None:
                final_proj.bias.zero_()
            dbg(f"Final projection scaled: weight×0.01, bias=0 "
                f"(was producing |logits|~1000 at init).")

    @torch.no_grad()
    def _extract_features(self, imgs: torch.Tensor):
        """Run the frozen backbone once and return (token_list, patch_start_idx).

        Args:
            imgs: [B, S, 3, H, W] in [0, 1]
        """
        with torch.amp.autocast(self.cfg.device, dtype=torch.bfloat16):
            token_list, patch_start_idx, _, _ = self.backbone(
                imgs, use_motion=False,
            )
        return token_list, patch_start_idx

    def forward(self, images: torch.Tensor):
        """Backbone (frozen, no_grad) -> hand_pred_head (trainable).

        Args:
            images: [B, 3, H, W] batch of frames from DataLoader.

        Returns:
            classifications: [B, H, W, num_classes] logits (channels-last, as
                produced by activate_head). Caller applies .permute(0,3,1,2) to
                get [B, num_classes, H, W] for CrossEntropyLoss.
        """
        # Treat the batch dimension as the sequence dimension S expected by the backbone.
        imgs = images.unsqueeze(0).to(self.cfg.device, non_blocking=True)

        # 1. Frozen backbone — single pass, no graph built
        token_list, patch_start_idx = self._extract_features(imgs)

        # 2. Trainable head in fp32 — feed fp32 tokens + images so gradients
        #    accumulate at full precision.
        token_list_fp32 = [t.float() for t in token_list]
        classifications, _ = self.head(
            token_list_fp32, images=imgs.float(), patch_start_idx=patch_start_idx,
        )

        # head returns [B=1, S, H, W, num_classes] (channels-last) → squeeze batch dim
        return classifications.squeeze(0)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    args = parse_args()

    cfg = TestConfig()
    cfg.num_classes = args.num_classes
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    cfg.resume_from = args.checkpoint

    class_names = build_class_names(args.num_classes)
    dbg(f"num_classes={args.num_classes}  classes={class_names}")
    dbg(f"device={cfg.device}  checkpoint={args.checkpoint}  bf1_tolerance={args.bf1_tolerance}")

    # ---- model ----
    model = LeanReconstructor(cfg)
    ckpt_path = f"{cfg.save_model_path_prefix}_{args.checkpoint}.ckpt"
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    model.head.load_state_dict(ckpt["model_state_dict"], strict=True)
    dbg(f"Loaded {ckpt_path}  (epoch={ckpt.get('epoch')}  "
        f"val_loss={ckpt.get('val_loss', float('nan')):.4f})")

    # ---- dataset ----
    test_dataset = HandObjectSegmentationDataset(
        data_root=args.data_root,
        streams=["stream1201-1", "stream1201-2"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=False,
    )
    dbg(f"Test samples: {len(test_dataset)}")

    # ---- run ----
    aggregated, per_sample = evaluate_test_dataset(
        model, test_loader, cfg,
        num_classes=args.num_classes,
        bf1_tolerance=args.bf1_tolerance,
    )

    report(aggregated, class_names)
    save_results(aggregated, per_sample, args)
    return aggregated, per_sample


if __name__ == "__main__":
    run()
