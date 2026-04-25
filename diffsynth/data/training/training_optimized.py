"""Optimized training script for hand_pred_head segmentation.

Key optimizations over training_with_debug.py:
  1. Runs frozen backbone ONCE per step (original runs it twice via prepare_contexts)
  2. Skips all frozen heads (depth, pts, normals, GS, camera) — only runs hand_pred_head
  3. Skips rasterizer entirely (outputs were unused in loss)
  4. Frame subsampling via configurable stride (consecutive frames are near-duplicates)
  5. Cosine-annealing LR schedule for continued refinement after initial convergence
"""

import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass

from diffsynth.auxiliary_models.worldmirror.models.heads.dense_head import DPTHead
from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.models.model_manager import ModelManager

from datetime import datetime
from dataclasses import dataclass, field

from ..SimpleHandObjectSegmentationDataset import HandObjectSegmentationDataset


def dbg(msg: str):
    print(f"[DBG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Model Architecture
    img_shape = (280, 280)
    patch_size: int = 14
    embed_dim: int = 1024
    token_dim: int = 2048
    num_classes: int = 2
    patch_start_idx: int = 5

    # Training Hyperparameters
    batch_size: int = 10
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    lr_min_factor: float = 0.01  # cosine annealing decays to lr * this factor
    class_weights = torch.tensor([0.03, 0.97])

    optimizer = torch.optim.AdamW

    # Dataset
    val_fraction: float = 0.1  # fraction of clips held out for validation

    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Reconstructor
    reconstruction_model_path = "models/NeoVerse/reconstructor.ckpt"
    save_model_path_prefix = "models/NeoVerse/hand_seg_model_opt"
    low_vram = False

    # Logging
    log_dir: str = field(
        default_factory=lambda: "runs/neoverse_seg_opt_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # DataLoader
    num_workers: int = 2
    pin_memory: bool = True


# ---------------------------------------------------------------------------
# Lean reconstructor wrapper — backbone-once, head-only forward
# ---------------------------------------------------------------------------

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

    def __init__(self, cfg: TrainConfig):
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
            if "gs_head.seg_conv" in name:
                param.requires_grad = True
                n_train += 1
            else:
                param.requires_grad = False
        dbg(f"Trainable: {n_train}/{n_total} params.")
        if n_train == 0:
            raise RuntimeError("No parameters matched 'hand_pred_head'.")

        # Keep references for the two components we actually use
        self.backbone = self.reconstructor.visual_geometry_transformer
        self.head = self.reconstructor.gs_head
        self.trainable_part = self.reconstructor.gs_head.seg_conv

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
            images: [B, S, 3, H, W] batch of frames from DataLoader.

        Returns:
            classifications: [B, S, H, W, num_classes] logits (channels-last, as
                produced by activate_head). Caller applies .permute(0,3,1,2) to
                get [B, num_classes, H, W] for CrossEntropyLoss.
        """
        # Treat the batch dimension as the sequence dimension S expected by the backbone.
        imgs = images.to(self.cfg.device, non_blocking=True)
        B, S = imgs.shape[:2]

        views = {
            "img":       imgs,
            "is_target": torch.zeros((B, S), dtype=torch.bool,  device=self.cfg.device),
            "is_static": torch.zeros((B, S), dtype=torch.bool,  device=self.cfg.device),
            "timestamp": torch.arange(S, dtype=torch.int64, device=self.cfg.device).unsqueeze(0).expand(B, -1),
        }


        # 1. Frozen backbone — single pass, no graph built
        token_list, patch_start_idx = self._extract_features(imgs)
        
        with torch.no_grad():
            with torch.amp.autocast(self.cfg.device, dtype=torch.bfloat16):
                context_preds = self.reconstructor.prepare_contexts(
                        views=views,
                        cond_flags=[0,0,0],
                        is_inference=False,
                        use_motion=False,
                )
        context_token_list = context_preds.get("token_list", token_list)


        # 2. Trainable head — gradients flow here
        with torch.amp.autocast(self.cfg.device, dtype=torch.bfloat16):
            _, _, _, seg_logits = self.reconstructor.gs_head(
                context_token_list, images=imgs, patch_start_idx=patch_start_idx,
            )

        # head returns [B, S, H, W, num_classes] (channels-last)
        return seg_logits.float()


# ---------------------------------------------------------------------------
# Metrics & checkpointing
# ---------------------------------------------------------------------------

def get_criterion(cfg):
    return nn.CrossEntropyLoss(weight=cfg.class_weights.to(cfg.device))


@torch.no_grad()
def compute_miou(pred_logits, gt_mask, num_classes=2):
    pred = pred_logits.argmax(dim=1)
    gt = gt_mask
    intersection = torch.zeros(num_classes, device=pred.device)
    union = torch.zeros(num_classes, device=pred.device)
    for c in range(num_classes):
        p = (pred == c)
        g = (gt == c)
        intersection[c] = (p & g).sum()
        union[c] = (p | g).sum()
    iou = intersection / (union + 1e-6)
    return iou.mean().item(), iou


def save_step_checkpoint(head, optimizer, epoch, global_step, loss, miou, cfg):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "mIoU": miou,
    }
    path = f"{cfg.save_model_path_prefix}_step{global_step}.ckpt"
    torch.save(ckpt, path)
    dbg(f"  -> step checkpoint global_step={global_step} (loss={loss:.4f}, mIoU={miou:.4f})")


def save_checkpoint(head, optimizer, epoch, train_loss, avg_miou, val_loss, cfg, best_val_loss):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "mIoU": avg_miou,
    }
    torch.save(ckpt, f"{cfg.save_model_path_prefix}_latest.ckpt")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(ckpt, f"{cfg.save_model_path_prefix}_best.ckpt")
        dbg(f"  -> new best val (val_loss={best_val_loss:.4f})")
    return best_val_loss


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, criterion, cfg, class_names):
    model.head.eval()
    total_loss, total_miou, n_steps = 0.0, 0.0, 0
    per_class = torch.zeros(cfg.num_classes, device=cfg.device)
    for batch in val_loader:
        images, gt_mask, _, _ = batch
        classifications = model.forward(images)
        classifications = classifications.permute(0,1,4,2,3)
        B,S,C,H,W = classifications.shape
        classifications = classifications.view(B*S,C,H,W)
        gt_mask = gt_mask.to(cfg.device).view(B*S,H,W)
        loss = criterion(classifications, gt_mask)
        miou, pc = compute_miou(classifications, gt_mask, cfg.num_classes)
        total_loss += loss.item()
        total_miou += miou
        per_class += pc
        n_steps += 1
    model.head.train()
    if n_steps == 0:
        return None, None, None
    return total_loss / n_steps, total_miou / n_steps, per_class / n_steps


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    dbg("=== train() [optimized] ===")
    cfg = TrainConfig()
    dbg(f"Config: device={cfg.device}, batch={cfg.batch_size}, epochs={cfg.epochs},"
        f"lr={cfg.learning_rate}")

    if cfg.device == "cuda":
        dbg(f"CUDA: {torch.cuda.get_device_name(0)} | torch={torch.__version__}")

    # ---- model ----
    model = LeanReconstructor(cfg)
    model.head.train()

    # ---- dataset split (clip-level 90/10) ----
    t0 = time.time()
    all_clip_names = sorted(p.stem for p in Path("diffsynth/data/training_data").glob("clip-*.npz"))
    n_val = max(1, int(len(all_clip_names) * cfg.val_fraction))
    val_clip_names = set(all_clip_names[-n_val:])
    train_clip_names = set(all_clip_names[:-n_val])
    dbg(f"Clips: {len(all_clip_names)} total → {len(train_clip_names)} train / {len(val_clip_names)} val")

    train_dataset = HandObjectSegmentationDataset(
        data_root="diffsynth/data/training_data_amodal",
        clip_names=train_clip_names,
    )
    val_dataset = HandObjectSegmentationDataset(
        data_root="diffsynth/data/training_data_amodal",
        clip_names=val_clip_names,
    )
    dbg(f"Samples: {len(train_dataset)} train / {len(val_dataset)} val  "
        f"(built in {time.time()-t0:.1f}s)")

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )

    # ---- optimiser + scheduler ----
    criterion = get_criterion(cfg=cfg)
    trainable = [p for p in model.head.parameters() if p.requires_grad]
    dbg(f"Optimizer: {len(trainable)} trainable tensors")
    optimizer = cfg.optimizer(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg.learning_rate * cfg.lr_min_factor,
    )
    dbg(f"Scheduler: CosineAnnealingLR over {total_steps} steps "
        f"(~{steps_per_epoch}/epoch), eta_min={cfg.learning_rate * cfg.lr_min_factor:.1e}")

    # ---- logging ----
    writer = SummaryWriter(log_dir=cfg.log_dir, flush_secs=10)
    os.makedirs(os.path.dirname(cfg.save_model_path_prefix), exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0
    class_names = ["background", "foreground"]

    # ---- epochs ----
    for epoch in range(cfg.epochs):
        dbg(f"=== Epoch {epoch+1}/{cfg.epochs} ===")
        epoch_loss = 0.0
        epoch_miou = 0.0
        epoch_per_class = torch.zeros(cfg.num_classes, device=cfg.device)
        t_epoch = time.time()

        loader_iter = iter(train_loader)
        step = -1
        while True:
            try:
                t_fetch = time.time()
                batch = next(loader_iter)
                fetch_time = time.time() - t_fetch
            except StopIteration:
                break
            step += 1

            images, gt_mask, _, _ = batch

            t_fwd = time.time()
            classifications = model.forward(images)
            classifications = classifications.permute(0,1,4,2,3)
            B,S,C,H,W = classifications.shape
            classifications = classifications.view(B*S,C,H,W)
            gt_mask = gt_mask.to(cfg.device).view(B*S,H,W)

            loss = criterion(classifications, gt_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()  # implicit GPU sync
            step_time = time.time() - t_fwd

            epoch_loss += loss_val
            global_step += 1

            miou, per_class = compute_miou(classifications.detach(), gt_mask, cfg.num_classes)
            epoch_miou += miou
            epoch_per_class += per_class

            # TensorBoard
            writer.add_scalar("train/loss_step", loss_val, global_step)
            writer.add_scalar("train/mIoU_step", miou, global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            for c, name in enumerate(class_names):
                writer.add_scalar(f"train/IoU_{name}_step", per_class[c].item(), global_step)
            writer.add_scalar("train/step_time", step_time, global_step)
            writer.add_scalar("train/fetch_time", fetch_time, global_step)

            if step < 5 or step % 10 == 0:
                dbg(f"  step {step}: loss={loss_val:.4f} mIoU={miou:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"fetch={fetch_time:.2f}s fwd+bwd={step_time:.2f}s")

            # Early checkpoint
            if epoch == 0 and step == 5:
                save_step_checkpoint(model.head, optimizer, epoch, global_step, loss_val, miou, cfg)
            elif global_step > 0 and global_step % 10_000 == 0:
                save_step_checkpoint(model.head, optimizer, epoch, global_step, loss_val, miou, cfg)

        if step < 0:
            dbg("WARNING: epoch produced 0 steps.")
            continue

        n_steps = step + 1
        avg_loss = epoch_loss / n_steps
        avg_miou = epoch_miou / n_steps
        epoch_per_class /= n_steps
        elapsed = time.time() - t_epoch

        # ---- validation ----
        val_loss, val_miou, val_per_class = evaluate(model, val_loader, criterion, cfg, class_names)

        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        writer.add_scalar("train/mIoU_epoch", avg_miou, epoch)
        writer.add_scalar("train/epoch_time_s", elapsed, epoch)
        for c, name in enumerate(class_names):
            writer.add_scalar(f"train/IoU_{name}", epoch_per_class[c].item(), epoch)
        if val_loss is not None:
            writer.add_scalar("val/loss_epoch", val_loss, epoch)
            writer.add_scalar("val/mIoU_epoch", val_miou, epoch)
            for c, name in enumerate(class_names):
                writer.add_scalar(f"val/IoU_{name}", val_per_class[c].item(), epoch)
        writer.flush()

        val_str = f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f}" if val_loss is not None else "val=n/a"
        dbg(f"Epoch {epoch+1}/{cfg.epochs}  loss={avg_loss:.4f}  mIoU={avg_miou:.4f}  {val_str}  ({elapsed:.0f}s)")

        effective_val_loss = val_loss if val_loss is not None else avg_loss
        best_val_loss = save_checkpoint(
            model.head, optimizer, epoch, avg_loss, avg_miou, effective_val_loss, cfg, best_val_loss
        )

    writer.close()
    dbg("=== Training complete ===")


if __name__ == "__main__":
    train()
