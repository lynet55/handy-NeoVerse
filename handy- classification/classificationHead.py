import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from dataclasses import dataclass
from typing import List

from PIL import Image
from torchvision.transforms import functional as TF

from helpers import setup_streaming_dataloader

@dataclass
class TrainConfig:
    # Model Architecture
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    token_dim: int = 2048  # 2 * embed_dim
    num_classes: int = 10
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


def compute_loss():
    return nn.CrossEntropyLoss()


class ClassificationHead(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.patch_start_idx = cfg.patch_start_idx
        self.norm = nn.LayerNorm(cfg.token_dim)
        self.head = nn.Sequential(
            nn.Linear(cfg.token_dim, cfg.token_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.token_dim // 2, cfg.num_classes),
        )

    def forward(self, token_list: List[torch.Tensor]):
        tokens = token_list[-1][:, :, self.patch_start_idx:]

        # Normalize then pool over patches
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=2)  # [B, S, token_dim]

        # Two-layer projection to logits: [B, S, num_classes]
        return self.head(pooled)


def train():
    cfg = TrainConfig()
    dataloader = setup_streaming_dataloader(batch_size=cfg.batch_size)

    try:
        from diffsynth.auxiliary_models import WorldMirror
        classification_model = WorldMirror(cfg)
    except ImportError:
        print("ClassificationHead Import failed.")
        classification_model = ClassificationHead(cfg)

    classification_model.to(cfg.device)
    classification_model.train()

    criterion = compute_loss()
    optimizer = cfg.optimizer(
        classification_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            images = batch["pixel_values"].to(cfg.device)
            ground_truth = batch["segmented_image"].to(cfg.device)

            optimizer.zero_grad()
            logits = classification_model(images)
            loss = criterion(logits, ground_truth)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{cfg.epochs}  loss: {epoch_loss / (step + 1):.4f}")
