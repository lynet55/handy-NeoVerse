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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass

from diffsynth.auxiliary_models.worldmirror.models.heads.dense_head import DPTHead
from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.models.model_manager import ModelManager

from ..SimpleHandObjectSegmentationDataset import HandObjectSegmentationDataset, ClipStreamSampler


def dbg(msg: str):
    print(f"[DBG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Model Architecture
    img_shape = (560, 336)
    patch_size: int = 14
    embed_dim: int = 1024
    token_dim: int = 2048
    num_classes: int = 4
    patch_start_idx: int = 5

    # Training Hyperparameters
    batch_size: int = 10
    epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    lr_min_factor: float = 0.01  # cosine annealing decays to lr * this factor

    optimizer = torch.optim.AdamW

    # Dataset
    frame_stride: int = 5  # sample every Nth frame (1 = all frames, original behaviour)

    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Reconstructor
    reconstruction_model_path = "models/NeoVerse/reconstructor.ckpt"
    save_model_path_prefix = "models/NeoVerse/hand_seg_model_opt"
    low_vram = False

    # Logging
    log_dir: str = "runs/neoverse_seg_opt"

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
            images: [S, 3, H, W] single-sample batch from dataset.

        Returns:
            classifications: [S, num_classes, H, W] logits.
        """
        imgs = images.unsqueeze(0).to(self.cfg.device, non_blocking=True)

        # 1. Frozen backbone — single pass, no graph built
        token_list, patch_start_idx = self._extract_features(imgs)

        # 2. Trainable head — gradients flow here
        with torch.amp.autocast(self.cfg.device, dtype=torch.bfloat16):
            classifications, _ = self.head(
                token_list, images=imgs, patch_start_idx=patch_start_idx,
            )

        # classifications shape: [B, S, num_classes, H, W] → squeeze batch
        return classifications.squeeze(0).float()


# ---------------------------------------------------------------------------
# Subsampled dataset wrapper
# ---------------------------------------------------------------------------

class StridedHandObjectDataset(HandObjectSegmentationDataset):
    """HandObjectSegmentationDataset with frame sub-sampling.

    Consecutive video frames are near-identical.  A stride of N keeps every
    Nth frame, cutting dataset size (and redundancy) proportionally.
    """

    def __init__(self, data_root: str, frame_stride: int = 1, streams=None):
        self._frame_stride = frame_stride
        super().__init__(data_root=data_root, streams=streams)

    def _build_index(self) -> None:
        from pathlib import Path

        for npz_path in sorted(Path(self.data_root).glob("clip-*.npz")):
            clip_name = npz_path.stem
            npz = np.load(str(npz_path), mmap_mode="r")
            n_frames = next(
                npz[k].shape[0] for k in npz.files if k.startswith("images_")
            )
            for stream in self.streams:
                if f"images_{stream}" not in npz.files:
                    continue
                for frame_idx in range(0, n_frames, self._frame_stride):
                    self.samples.append(
                        {"clip_name": clip_name, "stream": stream, "frame_idx": frame_idx}
                    )


# ---------------------------------------------------------------------------
# Metrics & checkpointing (unchanged from original)
# ---------------------------------------------------------------------------

def get_criterion():
    return nn.CrossEntropyLoss()


@torch.no_grad()
def compute_miou(pred_logits, gt_mask, num_classes=4):
    pred = pred_logits.argmax(dim=1)
    gt = gt_mask.argmax(dim=1)
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


def save_checkpoint(head, optimizer, epoch, avg_loss, avg_miou, cfg, best_loss):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "mIoU": avg_miou,
    }
    torch.save(ckpt, f"{cfg.save_model_path_prefix}_latest.ckpt")
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(ckpt, f"{cfg.save_model_path_prefix}_best.ckpt")
        dbg(f"  -> new best (loss={best_loss:.4f})")
    if (epoch + 1) % 50 == 0:
        torch.save(ckpt, f"{cfg.save_model_path_prefix}_epoch{epoch + 1}.ckpt")
    return best_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    dbg("=== train() [optimized] ===")
    cfg = TrainConfig()
    dbg(f"Config: device={cfg.device}, batch={cfg.batch_size}, epochs={cfg.epochs}, "
        f"stride={cfg.frame_stride}, lr={cfg.learning_rate}")

    if cfg.device == "cuda":
        dbg(f"CUDA: {torch.cuda.get_device_name(0)} | torch={torch.__version__}")

    # ---- model ----
    model = LeanReconstructor(cfg)

    # ---- dataset ----
    t0 = time.time()
    train_dataset = StridedHandObjectDataset(
        data_root="diffsynth/data/training_data",
        frame_stride=cfg.frame_stride,
    )
    dbg(f"Dataset: {len(train_dataset)} samples (stride={cfg.frame_stride}) built in {time.time()-t0:.1f}s")

    train_sampler = ClipStreamSampler(train_dataset, shuffle_clips=True)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )

    # ---- optimiser + scheduler ----
    criterion = get_criterion()
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
    best_loss = float("inf")
    global_step = 0
    class_names = ["right_hand", "left_hand", "object", "background"]

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
            classifications = model.forward(images).permute(0,3,1,2)[0]

            if cfg.device == "cuda":
                gt_mask = gt_mask.to("cuda", non_blocking=True)

            loss = criterion(classifications, gt_mask.argmax(dim=1).long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if cfg.device == "cuda":
                torch.cuda.synchronize()
            step_time = time.time() - t_fwd

            loss_val = loss.item()
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

        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        writer.add_scalar("train/mIoU_epoch", avg_miou, epoch)
        writer.add_scalar("train/epoch_time_s", elapsed, epoch)
        for c, name in enumerate(class_names):
            writer.add_scalar(f"train/IoU_{name}", epoch_per_class[c].item(), epoch)
        writer.flush()

        dbg(f"Epoch {epoch+1}/{cfg.epochs}  loss={avg_loss:.4f}  mIoU={avg_miou:.4f}  ({elapsed:.0f}s)")
        best_loss = save_checkpoint(model.head, optimizer, epoch, avg_loss, avg_miou, cfg, best_loss)

    writer.close()
    dbg("=== Training complete ===")


if __name__ == "__main__":
    train()
