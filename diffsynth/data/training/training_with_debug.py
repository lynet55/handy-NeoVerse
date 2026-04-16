import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as TF
from dataclasses import dataclass
from typing import List
import numpy as np
from PIL import Image
import os
import sys
import time

from diffsynth.auxiliary_models.worldmirror.models.heads.dense_head import DPTHead
from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.models.model_manager import ModelManager
from diffsynth.utils.auxiliary import homo_matrix_inverse

from .SimpleHandObjectSegmentationDataset import HandObjectSegmentationDataset, ClipStreamSampler


def dbg(msg: str):
    """Flushed debug print so we can see exactly where execution stops."""
    print(f"[DBG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    num_frames: int = 4
    epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    optimizer = torch.optim.AdamW

    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Reconstructor
    reconstruction_model_path = "models/NeoVerse/reconstructor.ckpt"
    save_model_path_prefix = "models/NeoVerse/hand_seg_model"
    low_vram = False
    scene_type = "static_scene"

    # Logging
    log_dir: str = "runs/neoverse_seg"

    # DataLoader
    # Start with 0 to confirm no DataLoader-fork hang. Bump to (cpus_per_task - 1)
    # once the first epoch completes cleanly.
    num_workers: int = 3
    pin_memory: bool = True


class NeoVerseReconstructor:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        try:
            dbg(f"Loading reconstructor from {cfg.reconstruction_model_path} (device: {cfg.device})...")
            model_manager = ModelManager()
            model_manager.load_model(cfg.reconstruction_model_path, device=cfg.device, torch_dtype=torch.bfloat16)
            reconstructor = model_manager.fetch_model("reconstructor")
            dbg("Reconstructor fetched from ModelManager.")
            self.reconstructor: WorldMirror = reconstructor
            self.vram_management_enabled = cfg.low_vram

            dbg("Setting requires_grad on parameters...")
            n_train, n_total = 0, 0
            for name, param in self.reconstructor.named_parameters():
                n_total += 1
                if "hand_pred_head" in name:
                    param.requires_grad = True
                    n_train += 1
                else:
                    param.requires_grad = False
            dbg(f"requires_grad set: {n_train}/{n_total} params trainable.")
            if n_train == 0:
                dbg("WARNING: no parameters matched 'hand_pred_head' — optimizer will be empty.")
        except ImportError:
            dbg("Reconstructor Import/Instantiation failed.")
            exit(1)

    def reconstruct(self, images: torch.Tensor):
        """Run reconstructor forward + rasterizer; return rgb, alpha, classification logits."""
        S = len(images)
        static = self.cfg.scene_type == "static_scene"
        views = {
            "img": images.unsqueeze(0).to(self.cfg.device, non_blocking=True),
            "is_target": torch.zeros((1, S), dtype=torch.bool, device=self.cfg.device),
            "is_static": torch.ones((1, S), dtype=torch.bool, device=self.cfg.device) if static
                         else torch.zeros((1, S), dtype=torch.bool, device=self.cfg.device),
            "timestamp": torch.zeros((1, S), dtype=torch.int64, device=self.cfg.device) if static
                         else torch.arange(S, dtype=torch.int64, device=self.cfg.device).unsqueeze(0),
        }

        if self.cfg.low_vram:
            self.reconstructor.to(self.cfg.device)

        try:
            with torch.amp.autocast(self.cfg.device, dtype=torch.bfloat16):
                predictions = self.reconstructor(views, is_inference=False, use_motion=False)
        finally:
            if self.cfg.low_vram:
                self.reconstructor.to("cpu")

        gaussians = predictions["splats"]
        input_c2w = predictions["rendered_extrinsics"][0]
        input_intrs = predictions["rendered_intrinsics"][0]
        input_timestamps = predictions["rendered_timestamps"][0]
        classifications = predictions["seg_labels"]
        input_w2c = homo_matrix_inverse(input_c2w)

        target_rgb, _, target_alphas = self.reconstructor.gs_renderer.rasterizer.forward(
            gaussians,
            render_viewmats=[input_w2c],
            render_Ks=[input_intrs],
            render_timestamps=[input_timestamps],
            sh_degree=0, width=self.cfg.img_shape[0], height=self.cfg.img_shape[1],
        )
        return target_rgb, target_alphas, classifications.permute(0, 1, 4, 2, 3)[0]


def get_criterion():
    return nn.CrossEntropyLoss()


@torch.no_grad()
def compute_miou(pred_logits, gt_mask, num_classes=4):
    """Compute per-class IoU and mIoU from a single batch."""
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


def save_checkpoint(reconstructor, optimizer, epoch, avg_loss, avg_miou, cfg, best_loss):
    """Save latest, best, and periodic (every 50 epochs) checkpoints. Returns updated best_loss."""
    ckpt = {
        "epoch": epoch,
        "model_state_dict": {
            k: v for k, v in reconstructor.state_dict().items()
            if "hand_pred_head" in k
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "mIoU": avg_miou,
    }

    torch.save(ckpt, f"{cfg.save_model_path_prefix}_latest.ckpt")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(ckpt, f"{cfg.save_model_path_prefix}_best.ckpt")
        dbg(f"  -> saved best checkpoint (loss={best_loss:.4f})")

    if (epoch + 1) % 50 == 0:
        torch.save(ckpt, f"{cfg.save_model_path_prefix}_epoch{epoch + 1}.ckpt")
        dbg(f"  -> saved periodic checkpoint at epoch {epoch + 1}")

    return best_loss


def train():
    dbg("=== train() entered ===")
    cfg = TrainConfig()
    dbg(f"Config: device={cfg.device}, num_workers={cfg.num_workers}, batch_size={cfg.batch_size}, epochs={cfg.epochs}")

    if cfg.device == "cuda":
        dbg(f"CUDA device: {torch.cuda.get_device_name(0)} | "
            f"torch={torch.__version__} cuda={torch.version.cuda}")

    dbg("Constructing NeoVerseReconstructor...")
    worldmirror = NeoVerseReconstructor(cfg)
    dbg("Reconstructor wrapper ready.")

    dbg("Constructing HandObjectSegmentationDataset...")
    t0 = time.time()
    train_dataset = HandObjectSegmentationDataset(data_root='diffsynth/data/training_data')
    dbg(f"Dataset built in {time.time()-t0:.2f}s, len={len(train_dataset)}")

    dbg("Constructing ClipStreamSampler...")
    train_sampler = ClipStreamSampler(train_dataset, shuffle_clips=True)
    dbg("Sampler built.")

    dbg(f"Constructing DataLoader (num_workers={cfg.num_workers}, pin_memory={cfg.pin_memory})...")
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )
    dbg("DataLoader built.")

    criterion = get_criterion()

    trainable = [p for p in worldmirror.reconstructor.parameters() if p.requires_grad]
    dbg(f"Optimizer will manage {len(trainable)} trainable tensors.")
    optimizer = cfg.optimizer(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # TensorBoard + checkpointing setup
    writer = SummaryWriter(log_dir=cfg.log_dir, flush_secs=10)
    save_dir = os.path.dirname(cfg.save_model_path_prefix)
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    global_step = 0
    class_names = ["right_hand", "left_hand", "object", "background"]

    for epoch in range(cfg.epochs):
        dbg(f"=== Epoch {epoch+1}/{cfg.epochs} starting ===")
        epoch_loss = 0.0
        epoch_miou = 0.0
        epoch_per_class = torch.zeros(cfg.num_classes, device=cfg.device)
        t0 = time.time()

        dbg("About to request first batch from DataLoader (workers spawn here)...")
        loader_iter = iter(train_loader)
        dbg("iter(train_loader) returned.")

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
            pred_rgb, pred_alpha, classifications = worldmirror.reconstruct(images)
            if cfg.device == "cuda":
                gt_mask = gt_mask.to("cuda", non_blocking=True)
            loss = criterion(classifications, gt_mask.argmax(dim=1).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cfg.device == "cuda":
                torch.cuda.synchronize()
            step_time = time.time() - t_fwd

            epoch_loss += loss.item()
            global_step += 1
            writer.add_scalar("train/loss_step", loss.item(), global_step)

            miou, per_class = compute_miou(classifications.detach(), gt_mask, cfg.num_classes)
            epoch_miou += miou
            epoch_per_class += per_class

            # Print every step early on, then every 10 once stable
            if step < 5 or step % 10 == 0:
                dbg(f"  step {step}: loss={loss.item():.4f} mIoU={miou:.4f} "
                    f"fetch={fetch_time:.2f}s step={step_time:.2f}s")

        if step < 0:
            dbg("WARNING: epoch produced 0 steps. Dataset/sampler may be empty.")
            continue

        avg_loss = epoch_loss / (step + 1)
        avg_miou = epoch_miou / (step + 1)
        epoch_per_class /= (step + 1)
        elapsed = time.time() - t0

        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        writer.add_scalar("train/mIoU_epoch", avg_miou, epoch)
        writer.add_scalar("train/epoch_time_s", elapsed, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        for c, name in enumerate(class_names):
            writer.add_scalar(f"train/IoU_{name}", epoch_per_class[c].item(), epoch)
        writer.flush()

        dbg(f"Epoch {epoch + 1}/{cfg.epochs}  loss: {avg_loss:.4f}  mIoU: {avg_miou:.4f}  ({elapsed:.0f}s)")

        best_loss = save_checkpoint(
            worldmirror.reconstructor, optimizer, epoch, avg_loss, avg_miou, cfg, best_loss
        )

    writer.close()
    dbg("=== Training complete ===")


if __name__ == "__main__":
    train()