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
import time

from diffsynth.auxiliary_models.worldmirror.models.heads.dense_head import DPTHead
from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.models.model_manager import ModelManager
from diffsynth.utils.auxiliary import homo_matrix_inverse

from ..SimpleHandObjectSegmentationDataset import HandObjectSegmentationDataset, ClipStreamSampler

@dataclass
class TrainConfig:
    # Model Architecture
    img_shape = (560, 336)
    patch_size: int = 14
    embed_dim: int = 1024
    token_dim: int = 2048  # 2 * embed_dim
    num_classes: int = 4  # classes 0-3
    patch_start_idx: int = 5  # 1 camera token + 4 register tokens

    # Training Hyperparameters
    batch_size: int = 10
    num_frames: int = 4
    epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    # Optimizer
    optimizer = torch.optim.AdamW

    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    #neoverse
    reconstruction_model_path = "models/NeoVerse/reconstructor.ckpt"
    save_model_path_prefix = "models/NeoVerse/hand_seg_model"
    low_vram = False
    scene_type = "static_scene"

    # Logging
    log_dir: str = "runs/neoverse_seg"


class NeoVerseReconstructor:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        try:
            print(f"Loading reconstructor from {cfg.reconstruction_model_path} (device: {cfg.device})...")
            model_manager = ModelManager()
            model_manager.load_model(cfg.reconstruction_model_path, device=cfg.device, torch_dtype=torch.bfloat16)
            reconstructor = model_manager.fetch_model("reconstructor")
            print("Reconstructor loaded.")
            self.reconstructor: WorldMirror = reconstructor
            self.vram_management_enabled = cfg.low_vram
            for name, param in self.reconstructor.named_parameters(): 
                if "hand_pred_head" in name: 
                    param.requires_grad = True
                else: 
                    param.requires_grad = False
        except ImportError:
            print("Reconstructor Import/Instansiation failed.")
            exit(1)

    def reconstruct(self, images: torch.Tensor):
        """
        Run VGGT backbone on image and return intermediate token list + patch_start_idx.

        Args:
            image: [C, H, W] single input image tensor (matches Hot3DClipsDataset output)
        Returns:
            token_list: List[Tensor] — 4 intermediate token tensors, each [1, S, patches, 2*embed_dim]
            patch_start_idx: int — index where patch tokens begin
            images: [1, S, C, H, W] — the views image tensor (needed by DPT head)
        """
        
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
        input_c2w = predictions["rendered_extrinsics"][0]   # [S, 4, 4]
        input_intrs = predictions["rendered_intrinsics"][0] # [S, 3, 3]
        input_timestamps = predictions["rendered_timestamps"][0]  # [S]
        classifications = predictions["seg_labels"]
        # --- Point cloud GLB ---
        input_w2c = homo_matrix_inverse(input_c2w)  # [S, 4, 4]

        target_rgb, _, target_alphas, _ = self.reconstructor.gs_renderer.rasterizer.forward(
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
    pred = pred_logits.argmax(dim=1)  # [B, H, W]
    gt = gt_mask.argmax(dim=1)        # [B, H, W]
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
    """Save latest, best, and periodic (every 50 epochs) checkpoints.
    Returns updated best_loss."""
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
        print(f"  -> saved best checkpoint (loss={best_loss:.4f})")

    if (epoch + 1) % 50 == 0:
        torch.save(ckpt, f"{cfg.save_model_path_prefix}_epoch{epoch + 1}.ckpt")
        print(f"  -> saved periodic checkpoint at epoch {epoch + 1}")

    return best_loss


def train():
    cfg = TrainConfig()
    worldmirror = NeoVerseReconstructor(cfg)

    train_dataset = HandObjectSegmentationDataset(data_root='diffsynth/data/training_data')
    train_sampler = ClipStreamSampler(train_dataset, shuffle_clips=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg.batch_size, num_workers=4)

    criterion = get_criterion()

    optimizer = cfg.optimizer(
        filter(lambda p: p.requires_grad, worldmirror.reconstructor.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # --- TensorBoard + checkpointing setup ---
    writer = SummaryWriter(log_dir=cfg.log_dir)
    save_dir = os.path.dirname(cfg.save_model_path_prefix)
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    global_step = 0
    class_names = ["right_hand", "left_hand", "object", "background"]

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        epoch_miou = 0.0
        t0 = time.time()

        for step, (images, gt_mask, _, _) in enumerate(train_loader):
            pred_rgb, pred_alpha, classifications = worldmirror.reconstruct(images)
            if cfg.device == "cuda":
                gt_mask = gt_mask.to("cuda", non_blocking=True)
            loss = criterion(classifications, gt_mask.argmax(dim=1).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # per-step scalar
            writer.add_scalar("train/loss_step", loss.item(), global_step)

            # batch mIoU (cheap, already computed)
            miou, per_class = compute_miou(classifications.detach(), gt_mask, cfg.num_classes)
            epoch_miou += miou

        # --- epoch-level logging ---
        avg_loss = epoch_loss / (step + 1)
        avg_miou = epoch_miou / (step + 1)
        elapsed = time.time() - t0

        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        writer.add_scalar("train/mIoU_epoch", avg_miou, epoch)
        writer.add_scalar("train/epoch_time_s", elapsed, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        for c, name in enumerate(class_names):
            writer.add_scalar(f"train/IoU_{name}", per_class[c].item(), epoch)

        print(f"Epoch {epoch + 1}/{cfg.epochs}  loss: {avg_loss:.4f}  mIoU: {avg_miou:.4f}  ({elapsed:.0f}s)")

        best_loss = save_checkpoint(
            worldmirror.reconstructor, optimizer, epoch, avg_loss, avg_miou, cfg, best_loss
        )

    writer.close()

if __name__ == "__main__":
    train()
