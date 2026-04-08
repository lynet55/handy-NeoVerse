import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List
import numpy as np
from PIL import Image
import os

from data import Hot3DClipsDataset
from diffsynth.auxiliary_models.worldmirror.models.heads.dense_head import DPTHead

@dataclass
class TrainConfig:
    # Model Architecture
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    token_dim: int = 2048  # 2 * embed_dim
    num_classes: int = 4  # classes 0-3
    patch_start_idx: int = 5  # 1 camera token + 4 register tokens

    # Training Hyperparameters
    batch_size: int = 2
    num_frames: int = 4
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Optimizer
    optimizer: torch.optim.Optimizer = torch.optim.AdamW

    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    #neoverse
    reconstruction_model_path = "models/NeoVerse/reconstructor.ckpt"
    low_vram = True
    scene_type = "Static scene"


class NeoVerseReconstructor:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        try:
            from diffsynth.pipelines import WanVideoNeoVersePipeline

            self.pipe = WanVideoNeoVersePipeline.from_pretrained(
                local_model_path="models",
                reconstructor_path=cfg.reconstruction_model_path,
                lora_path="models/NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
                lora_alpha=1.0,
                device=cfg.device,
                torch_dtype=torch.bfloat16,
                enable_vram_management=cfg.low_vram,
            )
        except ImportError:
            print("WanVideoNeoVersePipeline Import/Instansiation failed.")
            self.pipe = None

    def reconstruct(self, image: torch.Tensor):
        """
        Run VGGT backbone on image and return intermediate token list + patch_start_idx.

        Args:
            image: [C, H, W] single input image tensor (matches Hot3DClipsDataset output)
        Returns:
            token_list: List[Tensor] — 4 intermediate token tensors, each [1, S, patches, 2*embed_dim]
            patch_start_idx: int — index where patch tokens begin
            images: [1, S, C, H, W] — the views image tensor (needed by DPT head)
        """
        if self.pipe is None:
            raise RuntimeError("NeoVerse pipeline not available.")

        device = image.device
        pil_image = TF.to_pil_image(image.cpu())

        state = {"images": [pil_image], "scene_type": self.cfg.scene_type}
        pil_images = state["images"]
        static_flag = self.cfg.scene_type == "Static scene"
        S = len(pil_images)

        views = {
            "img": torch.stack([TF.to_tensor(img)[None] for img in pil_images], dim=1).to(device),
            "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
        }
        if static_flag:
            views["is_static"] = torch.ones((1, S), dtype=torch.bool, device=device)
            views["timestamp"] = torch.zeros((1, S), dtype=torch.int64, device=device)
        else:
            views["is_static"] = torch.zeros((1, S), dtype=torch.bool, device=device)
            views["timestamp"] = torch.arange(0, S, dtype=torch.int64, device=device).unsqueeze(0)

        # Low-VRAM: load reconstructor to GPU before use
        if self.pipe.vram_management_enabled:
            self.pipe.reconstructor.to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.pipe.torch_dtype):
            # Run just the backbone to get token_list
            imgs = views["img"]
            token_list, patch_start_idx, _, _ = self.pipe.reconstructor.visual_geometry_transformer(
                imgs, use_motion=False
            )

        # Low-VRAM: offload reconstructor back to CPU
        if self.pipe.vram_management_enabled:
            self.pipe.reconstructor.cpu()
            torch.cuda.empty_cache()
        return token_list, patch_start_idx, imgs



def get_criterion():
    return nn.CrossEntropyLoss()


def train():
    cfg = TrainConfig()

    neoverse = NeoVerseReconstructor(cfg)
    hand_pred_head = DPTHead(
        dim_in= 2*cfg.embed_dim,
        output_dim=cfg.num_classes,  # 4 classes, all channels are logits
        patch_size=cfg.patch_size,
        activation="linear+none",  # raw logits (no activation), no confidence split
    )

    # HOT3D DataLoaders
    train_dataset = Hot3DClipsDataset(
        input_dir="dataset/train/input",
        ground_truth_dir="dataset/train/gt",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    hand_pred_head.to(cfg.device)
    hand_pred_head.train()

    criterion = get_criterion()
    optimizer = cfg.optimizer(
        hand_pred_head.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):

            image = batch["image"].to(cfg.device)            # [B, C, H, W]
            gt_mask = batch["ground_truth_mask"].to(cfg.device)  # [B, H, W] long, values 0-3

            # Run VGGT backbone (frozen, no grad)
            # Process one image at a time since reconstruct expects [C, H, W]
            token_lists = []
            imgs_list = []
            for i in range(image.shape[0]):
                token_list, patch_start_idx, imgs = neoverse.reconstruct(image[i])
                token_lists.append(token_list)
                imgs_list.append(imgs)

            # Stack batch: each token_list is List[Tensor] of 4 levels
            batched_tokens = [
                torch.cat([tl[lvl] for tl in token_lists], dim=0)
                for lvl in range(len(token_lists[0]))
            ]
            batched_imgs = torch.cat(imgs_list, dim=0)  # [B, S, C, H, W]

            optimizer.zero_grad()
            logits = hand_pred_head(batched_tokens, batched_imgs, patch_start_idx)
            # logits: [B, S, num_classes, H, W] — squeeze S=1 for single-frame
            logits = logits[:, 0]  # [B, num_classes, H, W]

            loss = criterion(logits, gt_mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{cfg.epochs}  loss: {epoch_loss / (step + 1):.4f}")
