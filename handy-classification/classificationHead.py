import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# from torch.nn import functional as F  # BUG: wrong F — need torchvision, not torch.nn
from torchvision.transforms import functional as F
from dataclasses import dataclass
from typing import List
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
            Reconstruct scene from input image and render via GS rasterizer.
            Args:
                image: [C, H, W] single input image tensor (matches Hot3DClipsDataset output)
            Returns:
                rendered_rgb: [1, S, H, W, 3] rendered RGB from gaussian splatting
        """
        if self.pipe is None:
            raise RuntimeError("NeoVerse pipeline not available.")

        device = image.device
        # image is [C, H, W] from dataset — convert to PIL for pipeline
        pil_image = F.to_pil_image(image.cpu())

        # state = {"images": [pil_image], "scene_type": self.cfg.SCENE_TYPE}  # BUG: SCENE_TYPE → scene_type
        state = {"images": [pil_image], "scene_type": self.cfg.scene_type}
        pil_images = state["images"]
        static_flag = self.cfg.scene_type == "Static scene"
        S = len(pil_images)

        views = {
            "img": torch.stack([F.to_tensor(img)[None] for img in pil_images], dim=1).to(device),
            # "is_target": torch.zeros((1, 1), dtype=torch.bool, device=device),
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
            predictions = self.pipe.reconstructor(views, is_inference=True, use_motion=False)

        # Low-VRAM: offload reconstructor back to CPU
        if self.pipe.vram_management_enabled:
            self.pipe.reconstructor.cpu()
            torch.cuda.empty_cache()

        gaussians = predictions["splats"]
        input_intrs = predictions["rendered_intrinsics"][0]        # [S, 3, 3]
        input_cam2world = predictions["rendered_extrinsics"][0]     # [S, 4, 4]
        input_timestamps = predictions["rendered_timestamps"][0]    # [S]

        # # points, colors, frame_indices = extract_point_cloud(predictions)

        # state["source_views"] = views
        # state["gaussians"] = gaussians
        # state["input_intrs"] = input_intrs
        # state["input_cam2world"] = input_cam2world
        # state["input_timestamps"] = input_timestamps
        # state["points"] = points   # BUG: referenced before assignment (extract_point_cloud was commented out)
        # state["colors"] = colors
        # state["frame_indices"] = frame_indices
        # state["height"] = pil_images[0].size[1]
        # state["width"] = pil_images[0].size[0]

        # # Build GLB: 11-frame point cloud, all S cameras shown
        # scene = build_scene_glb(points, colors, frame_indices, input_cam2world.cpu().numpy())
        # glb_path = _export_scene(scene)

        # --- Render via GS rasterizer (see app.py:291-297) ---
        from diffsynth.utils.auxiliary import homo_matrix_inverse
        H, W = pil_images[0].size[1], pil_images[0].size[0]
        target_world2cam = homo_matrix_inverse(input_cam2world)

        with torch.no_grad():
            rendered_rgb, rendered_depth, rendered_alpha = (
                self.pipe.reconstructor.gs_renderer.rasterizer.forward(
                    gaussians,
                    render_viewmats=[target_world2cam],
                    render_Ks=[input_intrs],
                    render_timestamps=[input_timestamps],
                    sh_degree=0,
                    width=W,
                    height=H,
                )
            )

        return rendered_rgb  # [1, S, H, W, 3]


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

    def forward(self, token_list: "List[torch.Tensor]"):
        tokens = token_list[-1][:, :, self.patch_start_idx:]

        # Normalize then pool over patches
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=2)  # [B, S, token_dim]

        # Two-layer projection to logits: [B, S, num_classes]
        return self.head(pooled)
    

# def compute_loss(logit, gt_mask):  # BUG: args unused, returns unparameterized loss
#     return nn.CrossEntropyLoss()

def get_criterion():
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

    # criterion = compute_loss()  # BUG: compute_loss expected 2 args, returned unused CE
    criterion = get_criterion()
    optimizer = cfg.optimizer(
        classification_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        # for batch, label in enumerate(train_loader):  # BUG: enumerate yields (int, dict), names swapped
        for step, batch in enumerate(train_loader):

            # image = batch["egocentric_image"].to(cfg.device)  # BUG: wrong key, dataset uses "image"
            # gt_mask = batch["segmented_image"].to(cfg.device)  # BUG: wrong key, dataset uses "ground_truth_image"
            image = batch["image"].to(cfg.device)
            gt_mask = batch["ground_truth_image"].to(cfg.device)
            neoverse_reconstruction = neoverse.reconstruct(image)

            optimizer.zero_grad()
            logits = classification_model(neoverse_reconstruction)
            loss = criterion(logits, gt_mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # print(f"Epoch {epoch + 1}/{cfg.epochs}  loss: {epoch_loss / (step + 1):.4f}")  # BUG: 'step' was undefined
        print(f"Epoch {epoch + 1}/{cfg.epochs}  loss: {epoch_loss / (step + 1):.4f}")
