import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from torchvision.transforms import functional as TF

from helpers import setup_streaming_dataloader


def segment_HOI(frame: np.ndarray, hand_boxes: pd.DataFrame, timestamp_ns: int) -> np.ndarray:
    """Produce a ground-truth segmentation mask for a single frame.

    Args:
        frame: HxWx3 uint8 RGB image.
        hand_boxes: rows from box2d_hands.csv filtered to this timestamp.
        timestamp_ns: frame timestamp in nanoseconds.

    Returns:
        HxW uint8 label mask (0 = background, 1 = hand, …).
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for _, row in hand_boxes.iterrows():
        x1, x2 = int(row["x_min[pixel]"]), int(row["x_max[pixel]"])
        y1, y2 = int(row["y_min[pixel]"]), int(row["y_max[pixel]"])
        mask[y1:y2, x1:x2] = 1  # hand region
    return mask


class Hot3DAriaDataset(Dataset):
    """Dataset for HOT3D Aria recordings.

    Discovers all recordings under `data_root` by looking for directories
    matching the pattern `*_ground_truth` paired with a `*_preview_rgb.mp4`
    video. Each valid (timestamp, stream) entry that passes the QA mask is
    treated as one sample.

    Args:
        data_root: path that contains one or more recording bundles.
        img_size: spatial size to resize frames to (square).
        rgb_stream_id: stream ID used for the RGB camera in the CSVs.
        transform: optional torchvision transform applied to the RGB tensor.
    """

    RGB_STREAM = "214-1"

    def __init__(
        self,
        data_root: str,
        img_size: int = 518,
        rgb_stream_id: str = RGB_STREAM,
        transform=None,
    ):
        self.img_size = img_size
        self.rgb_stream_id = rgb_stream_id
        self.transform = transform

        self.samples: List[dict] = []  # one entry per valid frame
        self._videos: dict = {}        # recording_name -> cv2.VideoCapture

        for gt_dir in sorted(Path(data_root).glob("*_ground_truth")):
            recording_name = gt_dir.name.replace("_ground_truth", "")
            video_path = gt_dir.parent / f"{recording_name}_preview_rgb.mp4"
            if not video_path.exists():
                continue
            self._index_recording(gt_dir, video_path, recording_name)

    def _index_recording(self, gt_dir: Path, video_path: Path, recording_name: str):
        # Load per-frame annotations
        hands_df = pd.read_csv(gt_dir / "box2d_hands.csv")
        qa_df = pd.read_csv(gt_dir / "masks" / "mask_qa_pass.csv")

        # Keep only QA-passing RGB-stream timestamps
        valid_ts = set(
            qa_df[(qa_df["mask"] == True) & (qa_df["stream_id"] == self.rgb_stream_id)][
                "timestamp[ns]"
            ].tolist()
        )

        # Open the video to determine fps / total frames
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Map timestamps to frame indices using the timecode mapping if present
        tc_path = gt_dir / "timecode_devicetime_mapping.csv"
        if tc_path.exists():
            tc_df = pd.read_csv(tc_path)
            # Build a lookup: device_time_ns -> frame index (nearest)
            ts_sorted = sorted(valid_ts)
            tc_ns = tc_df.iloc[:, 0].values  # first col is timestamp
            tc_frames = (tc_df.iloc[:, 1].values * fps).astype(int)  # second col seconds
        else:
            tc_ns, tc_frames = None, None

        for ts in sorted(valid_ts):
            # Estimate frame index
            if tc_ns is not None:
                idx = int(np.interp(ts, tc_ns, tc_frames))
            else:
                # Fallback: assume timestamps start at first frame
                ts_list = sorted(valid_ts)
                idx = ts_list.index(ts)

            idx = max(0, min(idx, total_frames - 1))

            self.samples.append(
                {
                    "recording": recording_name,
                    "video_path": str(video_path),
                    "frame_idx": idx,
                    "timestamp_ns": ts,
                    "hand_boxes": hands_df[hands_df["timestamp[ns]"] == ts],
                }
            )

        self._videos[recording_name] = None  # opened lazily in __getitem__

    def _read_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # --- RGB frame (model input) ---
        frame = self._read_frame(sample["video_path"], sample["frame_idx"])
        frame_resized = cv2.resize(frame, (self.img_size, self.img_size))
        pixel_values = TF.to_tensor(Image.fromarray(frame_resized))  # [3, H, W]
        if self.transform is not None:
            pixel_values = self.transform(pixel_values)

        # --- Ground-truth segmentation mask ---
        gt_mask = segment_HOI(frame_resized, sample["hand_boxes"], sample["timestamp_ns"])
        segmented_image = torch.from_numpy(gt_mask).long()  # [H, W]

        return {
            "pixel_values": pixel_values,
            "segmented_image": segmented_image,
            "timestamp_ns": sample["timestamp_ns"],
        }

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
