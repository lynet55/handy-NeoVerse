import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# from torch.nn import functional as F  # BUG: wrong F — need torchvision, not torch.nn
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List
import numpy as np
from PIL import Image
import os

from data import Hot3DClipsDataset


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
    reconstruction_model_path = ""
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

        # state = {"images": [pil_image], "scene_type": self.cfg.SCENE_TYPE}  # BUG: SCENE_TYPE → scene_type
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


# ---------------------------------------------------------------------------
# Old ClassificationHead (pooled classifier — commented out, kept for reference)
# ---------------------------------------------------------------------------
# class ClassificationHead(nn.Module):
#     def __init__(self, cfg: TrainConfig):
#         super().__init__()
#         self.patch_start_idx = cfg.patch_start_idx
#         self.norm = nn.LayerNorm(cfg.token_dim)
#         self.head = nn.Sequential(
#             nn.Linear(cfg.token_dim, cfg.token_dim // 2),
#             nn.GELU(),
#             nn.Linear(cfg.token_dim // 2, cfg.num_classes),
#         )
#
#     def forward(self, token_list: "List[torch.Tensor]"):
#         tokens = token_list[-1][:, :, self.patch_start_idx:]
#
#         # Normalize then pool over patches
#         tokens = self.norm(tokens)
#         pooled = tokens.mean(dim=2)  # [B, S, token_dim]
#
#         # Two-layer projection to logits: [B, S, num_classes]
#         return self.head(pooled)


class SegmentationHead(nn.Module):
    """
    DPT-style dense segmentation head.
    Takes VGGT token_list and produces per-pixel class logits [B, S, num_classes, H, W].
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        dim_in = cfg.token_dim          # 2 * embed_dim = 2048
        patch_size = cfg.patch_size     # 14
        num_classes = cfg.num_classes   # 4
        features = 256
        out_channels = [256, 512, 1024, 1024]

        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim_in)

        # Project each intermediate token level to its channel count
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, kernel_size=1) for oc in out_channels
        ])

        # Resize layers matching DPT multi-scale fusion
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        # Scratch: 1x1 convs to unify channel counts
        self.layer1_rn = nn.Conv2d(out_channels[0], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False)

        # Refinement blocks (bottom-up fusion)
        self.refinenet4 = self._make_fusion_block(features, has_residual=False)
        self.refinenet3 = self._make_fusion_block(features)
        self.refinenet2 = self._make_fusion_block(features)
        self.refinenet1 = self._make_fusion_block(features)

        # Final conv → raw logits (no activation — CrossEntropyLoss expects raw logits)
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(
        self,
        token_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> torch.Tensor:
        """
        Args:
            token_list: 4 intermediate token tensors from VGGT, each [B, S, N, 2*embed_dim]
            images: [B, S, C, H, W] input images (used only for shape)
            patch_start_idx: index where patch tokens start in token dim 2

        Returns:
            logits: [B, S, num_classes, H, W] raw class logits
        """
        B, S, _, H, W = images.shape
        ph = H // self.patch_size
        pw = W // self.patch_size

        # Multi-scale feature extraction
        feats = []
        for proj, resize, tokens in zip(self.projects, self.resize_layers, token_list):
            patch_tokens = tokens[:, :, patch_start_idx:]                         # [B, S, patches, C]
            patch_tokens = patch_tokens.reshape(B * S, -1, patch_tokens.shape[-1])  # [B*S, patches, C]
            patch_tokens = self.norm(patch_tokens)
            feat = patch_tokens.permute(0, 2, 1).reshape(B * S, -1, ph, pw)       # [B*S, C, ph, pw]
            feat = proj(feat)
            feat = resize(feat)
            feats.append(feat)

        # Bottom-up refinement fusion
        layer_1_rn = self.layer1_rn(feats[0])
        layer_2_rn = self.layer2_rn(feats[1])
        layer_3_rn = self.layer3_rn(feats[2])
        layer_4_rn = self.layer4_rn(feats[3])

        out = self.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        out = self.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        out = self.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        out = self.refinenet1(out, layer_1_rn)

        # Upsample to input resolution
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)

        # Raw logits — no softmax (CrossEntropyLoss handles that)
        logits = self.output_conv(out)                                # [B*S, num_classes, H, W]
        logits = logits.reshape(B, S, *logits.shape[1:])              # [B, S, num_classes, H, W]
        return logits

    @staticmethod
    def _make_fusion_block(features, has_residual=True):
        return _FusionBlock(features, has_residual=has_residual)


class _ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class _FusionBlock(nn.Module):
    def __init__(self, features, has_residual=True):
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.res_unit1 = _ResidualConvUnit(features)
        self.res_unit2 = _ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1)

    def forward(self, x, residual=None, size=None):
        if self.has_residual and residual is not None:
            x = x + self.res_unit1(residual)
        x = self.res_unit2(x)
        if size is not None:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        x = self.out_conv(x)
        return x


# ---------------------------------------------------------------------------
# Old loss function (commented out)
# ---------------------------------------------------------------------------
# def compute_loss(logit, gt_mask):  # BUG: args unused, returned unparameterized loss
#     return nn.CrossEntropyLoss()

def get_criterion():
    return nn.CrossEntropyLoss()


def train():
    cfg = TrainConfig()

    neoverse = NeoVerseReconstructor(cfg)
    segmentation_model = SegmentationHead(cfg)

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

    segmentation_model.to(cfg.device)
    segmentation_model.train()

    criterion = get_criterion()
    optimizer = cfg.optimizer(
        segmentation_model.parameters(),
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
            logits = segmentation_model(batched_tokens, batched_imgs, patch_start_idx)
            # logits: [B, S, num_classes, H, W] — squeeze S=1 for single-frame
            logits = logits[:, 0]  # [B, num_classes, H, W]

            loss = criterion(logits, gt_mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{cfg.epochs}  loss: {epoch_loss / (step + 1):.4f}")
