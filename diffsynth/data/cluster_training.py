import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from dataclasses import dataclass
from typing import List
import numpy as np
from PIL import Image
import os

from diffsynth.auxiliary_models.worldmirror.models.heads.dense_head import DPTHead
from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.models.model_manager import ModelManager
from diffsynth.utils.auxiliary import homo_matrix_inverse

from .SimpleHandObjectSegmentationDataset import HandObjectSegmentationDataset, ClipStreamSampler

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
    batch_size: int = 10 #smoke test
    num_frames: int = 4
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Optimizer
    optimizer = torch.optim.AdamW

    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    #neoverse
    reconstruction_model_path = "models/NeoVerse/reconstructor.ckpt"
    save_model_path_prefix = "model/NeoVerse/reconstructor"
    save_model_path_suffix = "/.ckpt"
    low_vram = False
    scene_type = "static_scene"


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

        target_rgb, _ , target_alphas = self.reconstructor.gs_renderer.rasterizer.forward(
        gaussians,
        render_viewmats=[input_w2c],
        render_Ks=[input_intrs],
        render_timestamps=[input_timestamps],
        sh_degree=0, width=self.cfg.img_shape[0], height=self.cfg.img_shape[1],
        )
        return target_rgb, target_alphas, classifications.permute(0, 1, 4, 2, 3)[0]




def get_criterion():
    return nn.CrossEntropyLoss()


def train():
    cfg = TrainConfig()
    worldmirror = NeoVerseReconstructor(cfg)

    dataset = HandObjectSegmentationDataset(image_root='diffsynth/data/training_images/',
                                            mask_root='diffsynth/data/training_masks/'
                                            )
    
    sampler = ClipStreamSampler(dataset, shuffle_clips=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=10, num_workers=0)

    criterion = get_criterion()

    optimizer = cfg.optimizer(
        filter(lambda p: p.requires_grad, worldmirror.reconstructor.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for step, (images, gt_mask) in enumerate(dataloader):
            pred_rgb, pred_alpha, classifications = worldmirror.reconstruct(images)
            if cfg.device == "cuda":
                gt_mask = gt_mask.to("cuda", non_blocking=True)
            loss = criterion(classifications, gt_mask.argmax(dim=1).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{cfg.epochs}  loss: {epoch_loss / (step + 1):.4f}")

if __name__ == "__main__":
    train()
