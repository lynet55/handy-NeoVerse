import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from typing import List, Optional

# --- 1. Configuration ---
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
    
    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Model Definition ---
class VGGTClassifier(nn.Module):
    """
    Classification head for a frozen VGGT backbone.
    Expects token_list: List[4] of tensors [B, S, N, 2048]
    """
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.patch_start_idx = cfg.patch_start_idx
        self.norm = nn.LayerNorm(cfg.token_dim)
        
        self.head = nn.Sequential(
            nn.Linear(cfg.token_dim, cfg.token_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.token_dim // 4, cfg.num_classes),
        )

    def forward(self, token_list: List[torch.Tensor]):
        # We use the last layer's tokens: token_list[-1]
        # Skip camera and register tokens (indices 0 to patch_start_idx-1)
        tokens = token_list[-1][:, :, self.patch_start_idx:]
        
        # Apply normalization and Global Average Pooling over patches (dim=2)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=2)  # Result: [B, S, 2048]
        
        # Final projection to logits: [B, S, num_classes]
        return self.head(pooled)

# --- 3. Mock Dataset (Replace with your actual data) ---
class MockVideoDataset(Dataset):
    def __init__(self, cfg: TrainConfig, length=10):
        self.cfg = cfg
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Returns (Frames, Labels)
        # Frames: [S, C, H, W], Labels: [S]
        frames = torch.rand(self.cfg.num_frames, 3, self.cfg.img_size, self.cfg.img_size)
        labels = torch.randint(0, self.cfg.num_classes, (self.cfg.num_frames,))
        return frames, labels

# --- 4. Main Training Function ---
def train():
    # Initialize Config
    cfg = TrainConfig()
    print(f"Starting training on device: {cfg.device}")

    # A. Initialize Frozen Backbone
    # Note: Ensure VisualGeometryTransformer is imported/defined in your environment
    try:
        from diffsynth.auxiliary_models.worldmirror.models.models.visual_transformer import VisualGeometryTransformer
        backbone = VisualGeometryTransformer(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim
        ).to(cfg.device)
    except ImportError:
        print("VGGT Import failed. Using a mock backbone for demonstration.")
        backbone = lambda x: ([torch.randn(cfg.batch_size, cfg.num_frames, 100, cfg.token_dim).to(cfg.device)], cfg.patch_start_idx)

    # Set backbone to eval and freeze parameters
    if isinstance(backbone, nn.Module):
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False

    # B. Initialize Trainable Classifier
    model = VGGTClassifier(cfg).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # C. Prepare Data
    dataset = MockVideoDataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # D. Training Loop
    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
            
            optimizer.zero_grad()

            # Forward Pass: Backbone (No gradients)
            with torch.no_grad():
                # VGGT expects [B, S, C, H, W]
                token_list, _ = backbone(imgs)

            # Forward Pass: Classification Head
            logits = model(token_list) # [B, S, num_classes]

            # Calculate Loss: Flatten B and S for CrossEntropy compatibility
            # Input: [N, C], Target: [N]
            loss = criterion(
                logits.view(-1, cfg.num_classes), 
                labels.view(-1)
            )

            # Backward and Step
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1}/{cfg.epochs}] | Avg Loss: {avg_loss:.4f}")

    # E. Save Model
    torch.save(model.state_dict(), "vggt_classification_head.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()