import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from dataclasses import dataclass
import os

from data import Hot3DClipsDataset

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

    def reconstruct(self, image: torch.Tensor) -> torch.Tensor:
        
        """
            Placeholder for NeoVerse reconstruction logic.
            Args:
            image: [B, C, H, W] input egocentric image tensors
            Returns:
            [B, S, token_dim] reconstructed token sequence for classification head.
        """
        if self.pipe is None:
            raise RuntimeError("NeoVerse pipeline not available.")

        device = image.device
        pil_image = F.to_pil_image(image.cpu())

        state = {"images": [pil_image], "scene_type": self.cfg.SCENE_TYPE}
        pil_images = state["images"]

        views = {
            "img": torch.stack([F.to_tensor(img)[None] for img in pil_images], dim=1).to(device),
            "is_target": torch.zeros((1, 1), dtype=torch.bool, device=device),
        }

        with torch.amp.autocast("cuda", dtype=self.pipe.torch_dtype):
            predictions = self.pipe.reconstructor(views, is_inference=True, use_motion=False)
        
        # Low-VRAM: offload reconstructor back to CPU
        if self.pipe.vram_management_enabled:
            self.pipe.reconstructor.cpu()
            torch.cuda.empty_cache()

        gaussians = predictions["splats"]
        input_intrs = predictions["rendered_intrinsics"][0]        # [S, 3, 3]
        input_cam2world = predictions["rendered_extrinsics"][0]     # [S, 4, 4]
        input_timestamps = predictions["rendered_timestamps"][0]    # [S]

        # points, colors, frame_indices = extract_point_cloud(predictions)

        state["source_views"] = views
        state["gaussians"] = gaussians
        state["input_intrs"] = input_intrs
        state["input_cam2world"] = input_cam2world
        state["input_timestamps"] = input_timestamps
        state["points"] = points
        state["colors"] = colors
        state["frame_indices"] = frame_indices
        state["height"] = pil_images[0].size[1]
        state["width"] = pil_images[0].size[0]

        # # Build GLB: 11-frame point cloud, all S cameras shown
        # scene = build_scene_glb(points, colors, frame_indices, input_cam2world.cpu().numpy())
        # glb_path = _export_scene(scene)


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
    

def compute_loss(logit, gt_mask):
    return nn.CrossEntropyLoss()


def train():
    cfg = TrainConfig()

    neoverse = NeoVerseReconstructor(cfg)
    classification_model = ClassificationHead(cfg)

    #HOT3D DataLoaders
    train_dataset = Hot3DClipsDataset(
        input_dir="dataset/train/input",
        ground_truth_dir="dataset/train/gt",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

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
        for batch, label in enumerate(train_loader):

            image = batch["egocentric_image"].to(cfg.device)
            gt_mask = batch["segmented_image"].to(cfg.device)
            neoverse_reconstruction = neoverse.reconstruct(image)

            optimizer.zero_grad()
            logits = classification_model(neoverse_reconstruction)
            loss = criterion(logits, gt_mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{cfg.epochs}  loss: {epoch_loss / (step + 1):.4f}")
