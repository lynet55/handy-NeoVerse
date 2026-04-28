"""Optimized training script for hand_pred_head segmentation.

Key optimizations over training_with_debug.py:
  1. Runs frozen backbone ONCE per step (original runs it twice via prepare_contexts)
  2. Skips all frozen heads (depth, pts, normals, GS, camera) — only runs hand_pred_head
  3. Skips rasterizer entirely (outputs were unused in loss)
  4. Frame subsampling via configurable stride (consecutive frames are near-duplicates)
  5. Cosine-annealing LR schedule for continued refinement after initial convergence
"""

import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass


from diffsynth.auxiliary_models.worldmirror.models.heads.dense_head import DPTHead
from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.models.model_manager import ModelManager

from datetime import datetime
from dataclasses import dataclass, field

from diffsynth.data.SimpleHandObjectSegmentationDataset import HandObjectSegmentationDataset, ClipStreamSampler


def dbg(msg: str):
    print(f"[DBG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Model Architecture
    img_shape = (280,280)
    patch_size: int = 14
    embed_dim: int = 1024
    token_dim: int = 2048
    num_classes: int = 4
    patch_start_idx: int = 5

    # Training Hyperparameters
    batch_size: int = 50
    epochs: int = 10
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    lr_min_factor: float = 0.01  # cosine annealing decays to lr * this factor
    class_weights = torch.tensor([10.0, 10.0, 5.0, 1.0])

    optimizer = torch.optim.AdamW
    grad_clip_norm = 1.0
    # Dataset
    frame_stride: int = 3  # sample every Nth frame (1 = all frames, original behaviour)
    val_fraction: float = 0.1  # fraction of clips held out for validation

    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Reconstructor
    reconstruction_model_path = "models/NeoVerse/reconstructor.ckpt"
    save_model_path_prefix = "models/NeoVerse/hand_seg_model_opt"
    low_vram = False

    # Resume: path to a head checkpoint produced by save_checkpoint /
    # save_step_checkpoint. If None, train from scratch. If the literal
    # string "latest", auto-resolves to <prefix>_latest.ckpt. If "best",
    # resolves to <prefix>_best.ckpt.
    resume_from: str = "latest"

    # Run id: shared between log_dir and per-epoch checkpoint filenames so
    # the TensorBoard run and its checkpoints can be cross-referenced. Two
    # separate default_factory lambdas would generate two timestamps a few
    # microseconds apart — to keep them identical we generate one timestamp
    # in __post_init__ and stamp both fields from it.
    run_id: str = ""
    log_dir: str = ""

    def __post_init__(self):
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not self.log_dir:
            self.log_dir = f"runs/neoverse_seg_opt_{self.run_id}"

    # DataLoader
    num_workers: int = 2
    pin_memory: bool = True


# ---------------------------------------------------------------------------
# Lean reconstructor wrapper — backbone-once, head-only forward
# ---------------------------------------------------------------------------

class LeanReconstructor:
    """Loads WorldMirror but only exercises backbone + hand_pred_head.

    The original training pipeline called ``reconstructor(views)`` which:
      * runs the ViT backbone,
      * runs ``prepare_contexts`` (backbone a *second* time when enable_gs),
      * runs every enabled prediction head (depth, pts, normals, GS, cam),
      * runs the Gaussian-splat rasterizer.

    Only ``hand_pred_head`` is trainable and only its output enters the loss.
    This wrapper skips all the rest.
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        dbg(f"Loading reconstructor from {cfg.reconstruction_model_path} ...")
        model_manager = ModelManager()
        model_manager.load_model(
            cfg.reconstruction_model_path,
            device=cfg.device,
            torch_dtype=torch.bfloat16,
        )
        self.reconstructor: WorldMirror = model_manager.fetch_model("reconstructor")
        dbg("Reconstructor loaded.")

        # Freeze everything, unfreeze hand_pred_head
        n_train, n_total = 0, 0
        for name, param in self.reconstructor.named_parameters():
            n_total += 1
            if "hand_pred_head" in name:
                param.requires_grad = True
                n_train += 1
            else:
                param.requires_grad = False
        dbg(f"Trainable: {n_train}/{n_total} params.")
        if n_train == 0:
            raise RuntimeError("No parameters matched 'hand_pred_head'.")

        # Keep references for the two components we actually use
        self.backbone = self.reconstructor.visual_geometry_transformer
        self.head = self.reconstructor.hand_pred_head

        # bfloat16 has 7 mantissa bits (eps ~8e-3 at magnitude 1), so Adam
        # updates of order `lr` round to zero and the head silently freezes.
        # Cast head to fp32 so optimizer state and updates stay precise.
        self.head = self.head.float()
        for p in self.head.parameters():
            p.requires_grad = True
        dbg("Head cast to float32 for training.")

        # The hand_pred_head is freshly initialized (not in reconstructor.ckpt)
        # and the upstream DPT features are unbounded, so default Kaiming init
        # on the final projection produces logits with |x| ~1000 — softmax
        # then saturates and per-pixel CE blows up to hundreds. Scaling the
        # final conv down at init makes first-step outputs O(1) instead of
        # O(1000), so training escapes the "shrink the logits" phase that
        # otherwise eats the first ~200 steps before real learning starts.
        with torch.no_grad():
            final_proj = self.head.scratch.output_conv2[2]
            final_proj.weight.mul_(0.01)
            if final_proj.bias is not None:
                final_proj.bias.zero_()
            dbg(f"Final projection scaled: weight×0.01, bias=0 "
                f"(was producing |logits|~1000 at init).")

    @torch.no_grad()
    def _extract_features(self, imgs: torch.Tensor):
        """Run the frozen backbone once and return (token_list, patch_start_idx).

        Args:
            imgs: [B, S, 3, H, W] in [0, 1]
        """
        with torch.amp.autocast(self.cfg.device, dtype=torch.bfloat16):
            token_list, patch_start_idx, _, _ = self.backbone(
                imgs, use_motion=False,
            )
        return token_list, patch_start_idx

    def forward(self, images: torch.Tensor):
        """Backbone (frozen, no_grad) -> hand_pred_head (trainable).

        Args:
            images: [B, 3, H, W] batch of frames from DataLoader.

        Returns:
            classifications: [B, H, W, num_classes] logits (channels-last, as
                produced by activate_head). Caller applies .permute(0,3,1,2) to
                get [B, num_classes, H, W] for CrossEntropyLoss.
        """
        # Treat the batch dimension as the sequence dimension S expected by the backbone.
        imgs = images.unsqueeze(0).to(self.cfg.device, non_blocking=True)

        # 1. Frozen backbone — single pass, no graph built
        token_list, patch_start_idx = self._extract_features(imgs)

        # 2. Trainable head in fp32 — feed fp32 tokens + images so gradients
        #    accumulate at full precision.
        token_list_fp32 = [t.float() for t in token_list]
        classifications, _ = self.head(
            token_list_fp32, images=imgs.float(), patch_start_idx=patch_start_idx,
        )

        # head returns [B=1, S, H, W, num_classes] (channels-last) → squeeze batch dim
        return classifications.squeeze(0)


# ---------------------------------------------------------------------------
# Subsampled dataset wrapper
# ---------------------------------------------------------------------------

class StridedHandObjectDataset(HandObjectSegmentationDataset):
    """HandObjectSegmentationDataset with frame sub-sampling and optional clip filter.

    Consecutive video frames are near-identical.  A stride of N keeps every
    Nth frame, cutting dataset size (and redundancy) proportionally.

    clip_names: optional set of clip stems (e.g. {"clip-000001", ...}) to
        include.  If None, all clips in data_root are used.
    """

    def __init__(self, data_root: str, frame_stride: int = 1, streams=None, clip_names=None):
        self._frame_stride = frame_stride
        self._clip_names_filter = clip_names  # None → include all
        super().__init__(data_root=data_root, streams=streams)

    def _build_index(self) -> None:
        for npz_path in sorted(Path(self.data_root).glob("clip-*.npz")):
            clip_name = npz_path.stem
            if self._clip_names_filter is not None and clip_name not in self._clip_names_filter:
                continue
            npz = np.load(str(npz_path), mmap_mode="r")
            n_frames = next(
                npz[k].shape[0] for k in npz.files if k.startswith("images_")
            )
            for stream in self.streams:
                if f"images_{stream}" not in npz.files:
                    continue
                for frame_idx in range(0, n_frames, self._frame_stride):
                    self.samples.append(
                        {"clip_name": clip_name, "stream": stream, "frame_idx": frame_idx}
                    )


# ---------------------------------------------------------------------------
# Metrics & checkpointing
# ---------------------------------------------------------------------------

def get_criterion(cfg):
    # label_smoothing=0 — the pretrained head produces confident logits;
    # smoothing pushes them toward uniform and destroys the prior on the
    # first few steps. Re-enable later if overfitting.
    return nn.CrossEntropyLoss(weight=cfg.class_weights.to(cfg.device), label_smoothing=0.0)


@torch.no_grad()
def compute_miou(pred_logits, gt_mask, num_classes=4):
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


def save_step_checkpoint(head, optimizer, epoch, global_step, loss, miou, cfg, best_val_loss=None):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "mIoU": miou,
        "best_val_loss": best_val_loss,
    }
    path = f"{cfg.save_model_path_prefix}_step{global_step}.ckpt"
    torch.save(ckpt, path)
    dbg(f"  -> step checkpoint global_step={global_step} (loss={loss:.4f}, mIoU={miou:.4f})")


def save_checkpoint(head, optimizer, epoch, train_loss, avg_miou, val_loss, cfg, best_val_loss, global_step=None):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "mIoU": avg_miou,
        "best_val_loss": best_val_loss,  # written *before* update below
        "run_id": cfg.run_id,
    }
    # _latest.ckpt: overwritten every epoch, used as the default resume target.
    torch.save(ckpt, f"{cfg.save_model_path_prefix}_latest.ckpt")
    # Per-epoch archive: never overwritten. run_id keeps separate runs from
    # colliding (resumed runs get a fresh run_id, so their epoch files won't
    # clash with the original run's). Epoch number is 1-indexed in the
    # filename to match the human-facing "Epoch N/M" log lines.
    epoch_path = f"{cfg.save_model_path_prefix}_run{cfg.run_id}_epoch{epoch+1:03d}.ckpt"
    torch.save(ckpt, epoch_path)
    dbg(f"  -> saved {os.path.basename(epoch_path)}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        ckpt["best_val_loss"] = best_val_loss
        torch.save(ckpt, f"{cfg.save_model_path_prefix}_best.ckpt")
        dbg(f"  -> new best val (val_loss={best_val_loss:.4f})")
    return best_val_loss


def resolve_resume_path(cfg) -> str:
    """Map 'latest'/'best' shortcuts to actual filenames; pass through paths."""
    if cfg.resume_from is None:
        return None
    if cfg.resume_from == "latest":
        return f"{cfg.save_model_path_prefix}_latest.ckpt"
    if cfg.resume_from == "best":
        return f"{cfg.save_model_path_prefix}_best.ckpt"
    return cfg.resume_from


def load_checkpoint(model, optimizer, cfg, train_loader_len: int):
    """Restore head weights + optimizer state from a checkpoint.

    Returns (start_epoch, global_step, best_val_loss). If no resume is
    configured or the file is missing, returns (0, 0, +inf) so training
    starts fresh.

    Notes on what is and isn't restored:
      * head weights + AdamW state: yes (this is the point).
      * global_step: from checkpoint if present, else estimated as
        (epoch+1) * train_loader_len. Used to decide whether warmup
        is still active and to keep TB step monotonic.
      * scheduler state: not stored in the checkpoint format. The caller
        fast-forwards LinearLR/ExponentialLR by stepping them the right
        number of times — see train().
      * best_val_loss: from checkpoint if present, else seeded from the
        checkpoint's val_loss to avoid clobbering an existing _best.ckpt.
    """
    path = resolve_resume_path(cfg)
    if path is None:
        return 0, 0, float("inf")
    if not os.path.exists(path):
        dbg(f"WARNING: resume_from={path!r} does not exist — starting fresh.")
        return 0, 0, float("inf")

    dbg(f"Resuming from {path} ...")
    ckpt = torch.load(path, map_location=cfg.device, weights_only=False)

    # Head weights. strict=True — if it mismatches we want to know loudly,
    # not silently train half a network.
    missing, unexpected = model.head.load_state_dict(ckpt["model_state_dict"], strict=True)
    if missing or unexpected:
        # strict=True would have raised; this branch is defensive in case
        # someone flips it to False later.
        dbg(f"WARNING: head load missing={len(missing)} unexpected={len(unexpected)}")

    # Optimizer state. After load, move tensors to the right device — torch
    # restores them on whatever device they were saved from, which can be
    # wrong if you trained on a different GPU id.
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(cfg.device)

    # Epoch: checkpoints store the epoch that *just finished*. We start the
    # next one. (For step-checkpoints saved mid-epoch this is approximate —
    # we restart that epoch from the top, which is the safer default.)
    finished_epoch = int(ckpt.get("epoch", -1))
    start_epoch = finished_epoch + 1

    # Global step: prefer the saved value; fall back to a reasonable estimate.
    if ckpt.get("global_step") is not None:
        global_step = int(ckpt["global_step"])
    else:
        global_step = start_epoch * max(1, train_loader_len)
        dbg(f"  global_step missing from checkpoint — estimated {global_step} "
            f"as ({start_epoch}*{train_loader_len}).")

    # best_val_loss: prefer saved; else seed from val_loss; else +inf.
    if ckpt.get("best_val_loss") is not None and math.isfinite(float(ckpt["best_val_loss"])):
        best_val_loss = float(ckpt["best_val_loss"])
    elif ckpt.get("val_loss") is not None:
        best_val_loss = float(ckpt["val_loss"])
        dbg(f"  best_val_loss missing — seeded from ckpt val_loss={best_val_loss:.4f}.")
    else:
        best_val_loss = float("inf")

    dbg(f"Resumed: start_epoch={start_epoch}  global_step={global_step}  "
        f"best_val_loss={best_val_loss:.4f}")
    return start_epoch, global_step, best_val_loss


class DiceLoss:
    """Multi-class Dice loss that ignores classes absent from each sample.

    The naive per-class formulation hands out a perfect score (smooth/smooth)
    when both prediction and target are empty for a class, which rewards the
    "predict nothing for that class" collapse on absent classes. Here we mask
    those classes out and average only over classes actually present per sample.
    """

    def __init__(self, smooth: float = 1.0):
        self.smooth = smooth

    def __call__(self, pred, target):
        pred = F.softmax(pred, dim=1)  # logits -> probs
        b, c, *_ = pred.shape
        score_sum = pred.new_zeros(b)
        present_count = pred.new_zeros(b)
        for k in range(c):
            pred_k = pred[:, k]
            tgt_k = target[:, k]
            inter = (pred_k * tgt_k).sum(dim=(1, 2))
            union = pred_k.sum(dim=(1, 2)) + tgt_k.sum(dim=(1, 2))
            score = (2.0 * inter + self.smooth) / (union + self.smooth)
            present = (tgt_k.sum(dim=(1, 2)) > 0).float()
            score_sum += score * present
            present_count += present
        per_sample = score_sum / present_count.clamp(min=1.0)
        return 1.0 - per_sample.mean()
# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, criterion, cfg, class_names):
    model.head.eval()
    total_loss, total_miou, n_steps = 0.0, 0.0, 0
    per_class = torch.zeros(cfg.num_classes, device=cfg.device)
    K = cfg.num_classes
    confusion = torch.zeros(K, K, dtype=torch.long, device=cfg.device)
    for batch in val_loader:
        images, gt_mask, _, _ = batch
        classifications = model.forward(images).permute(0, 3, 1, 2)
        if cfg.device == "cuda":
            gt_mask = gt_mask.to("cuda", non_blocking=True)
        loss = criterion(classifications, gt_mask.argmax(dim=1).long())
        miou, pc = compute_miou(classifications, gt_mask, cfg.num_classes)
        # Accumulate confusion matrix (rows=GT, cols=pred).
        pred_flat = classifications.argmax(dim=1).reshape(-1)
        gt_flat = gt_mask.argmax(dim=1).reshape(-1)
        idx = gt_flat * K + pred_flat
        confusion += torch.bincount(idx, minlength=K * K).reshape(K, K)
        total_loss += loss.item()
        total_miou += miou
        per_class += pc
        n_steps += 1
    model.head.train()
    if n_steps == 0:
        return None, None, None, None
    return total_loss / n_steps, total_miou / n_steps, per_class / n_steps, confusion


def log_confusion_matrix(confusion: torch.Tensor, class_names):
    """Pretty-print row-normalized confusion (rows=GT, cols=pred)."""
    cm = confusion.detach().cpu().float()
    row_sum = cm.sum(dim=1, keepdim=True).clamp(min=1)
    cm_pct = (cm / row_sum) * 100.0
    name_w = max(6, max(len(n) for n in class_names))
    header = " " * (name_w + 6) + " ".join(f"{n[:10]:>10}" for n in class_names)
    dbg("Confusion (rows=GT, cols=pred, row-norm %):")
    dbg(header)
    for i, n in enumerate(class_names):
        row = " ".join(f"{cm_pct[i, j].item():>10.1f}" for j in range(len(class_names)))
        dbg(f"{n:>{name_w}} ({int(cm[i].sum().item()):>6}) | {row}")


@torch.no_grad()
def log_class_pixel_distribution(loader, num_classes, class_names, label):
    """One-shot full-loader pixel count per class. Expensive — call sparingly."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, gt, *_ in loader:
        counts += torch.bincount(gt.argmax(dim=1).reshape(-1), minlength=num_classes)
    total = counts.sum().clamp(min=1)
    pct = counts.float() / total * 100.0
    dbg(f"Class pixel distribution ({label}, total={int(total.item()):,} px):")
    for i, n in enumerate(class_names):
        dbg(f"  {n:>11}: {int(counts[i].item()):>12,}  ({pct[i].item():.2f}%)")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    dbg("=== train() [optimized] ===")
    cfg = TrainConfig()
    dbg(f"Config: device={cfg.device}, batch={cfg.batch_size}, epochs={cfg.epochs}, "
        f"stride={cfg.frame_stride}, lr={cfg.learning_rate}")
    if cfg.resume_from is not None:
        dbg(f"Resume requested: {cfg.resume_from!r}")

    if cfg.device == "cuda":
        dbg(f"CUDA: {torch.cuda.get_device_name(0)} | torch={torch.__version__}")

    # ---- model ----
    model = LeanReconstructor(cfg)
    model.head.train()

    # ---- dataset split (clip-level 90/10) ----
    t0 = time.time()
    all_clip_names = sorted(p.stem for p in Path("diffsynth/data/training_data").glob("clip-*.npz"))
    n_val = max(1, int(len(all_clip_names) * cfg.val_fraction))
    val_clip_names = set(all_clip_names[-n_val:])
    train_clip_names = set(all_clip_names[:-n_val])
    dbg(f"Clips: {len(all_clip_names)} total → {len(train_clip_names)} train / {len(val_clip_names)} val")

    train_dataset = StridedHandObjectDataset(
        data_root="diffsynth/data/training_data",
        frame_stride=cfg.frame_stride,
        clip_names=train_clip_names,
    )
    val_dataset = StridedHandObjectDataset(
        data_root="diffsynth/data/training_data",
        frame_stride=cfg.frame_stride,
        clip_names=val_clip_names,
    )
    dbg(f"Samples: {len(train_dataset)} train / {len(val_dataset)} val  "
        f"(built in {time.time()-t0:.1f}s)")

    train_sampler = ClipStreamSampler(train_dataset, shuffle_clips=True)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_sampler = ClipStreamSampler(val_dataset, shuffle_clips=False)
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )

    # ---- optimiser + scheduler ----
    # Modest fg upweighting (bg=1, fg=3). [10,10,10,1] previously caused first-
    # step loss spikes that destroyed the pretrained head before warmup could
    # protect it. Full-dataset frequency scan was skipped (rough prior is fine).
    print("class weights (bg=1):", cfg.class_weights.tolist())
    criterion = get_criterion(cfg=cfg)
    dice_loss = DiceLoss()
    trainable = [p for p in model.head.parameters() if p.requires_grad]
    dbg(f"Optimizer: {len(trainable)} trainable tensors")
    optimizer = cfg.optimizer(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # 100-step linear warmup from lr/100 → lr, then per-epoch ExponentialLR(0.8).
    # The warmup protects the pretrained head from first-step gradient damage
    # while the loss is still settling.
    warmup_steps = 100
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    epoch_decay = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.80, last_epoch=-1
    )

    # ---- resume (after optimizer is built so we can restore its state) ----
    # train_loader_len is approximate but only used as a fallback estimator
    # when the checkpoint pre-dates global_step being saved.
    try:
        train_loader_len = len(train_loader)
    except TypeError:
        # Sampler may not expose __len__; harmless fallback.
        train_loader_len = 0
    start_epoch, global_step, best_val_loss = load_checkpoint(
        model, optimizer, cfg, train_loader_len
    )

    # Fast-forward schedulers to match restored progress.
    # NOTE: PyTorch's LinearLR / ExponentialLR use multiplicative updates from
    # group["lr"] (the *current* lr), not from base_lrs. Replaying step()s
    # after load_checkpoint compounds the saved-lr × ramp-factors and silently
    # explodes lr (e.g. 6.4e-5 × 100 from warmup = 6.4e-3). Compute the target
    # lr directly from cfg.learning_rate, set it, then advance the schedulers'
    # last_epoch so future epoch boundaries behave.
    if start_epoch > 0 or global_step > 0:
        if global_step >= warmup_steps:
            target_lr = cfg.learning_rate * (0.80 ** start_epoch)
            warmup.last_epoch = warmup_steps  # mark warmup finished
            epoch_decay.last_epoch = start_epoch - 1  # next .step() = start_epoch
        else:
            # mid-warmup resume: lerp start_factor → 1 by global_step / warmup_steps
            frac = global_step / warmup_steps
            target_lr = cfg.learning_rate * (0.01 + 0.99 * frac)
            warmup.last_epoch = global_step
            epoch_decay.last_epoch = -1
        for pg in optimizer.param_groups:
            pg["lr"] = target_lr
        dbg(f"Schedulers fast-forwarded: target_lr={target_lr:.2e} "
            f"(was {optimizer.param_groups[0]['lr']:.2e} after ckpt restore)")

    # ---- logging ----
    writer = SummaryWriter(log_dir=cfg.log_dir, flush_secs=10)
    os.makedirs(os.path.dirname(cfg.save_model_path_prefix), exist_ok=True)
    class_names = ["right_hand", "left_hand", "object", "background"]

    # ---- epochs ----
    from collections import deque
    miou_window = deque(maxlen=20)
    loss_window = deque(maxlen=20)
    for epoch in range(start_epoch, cfg.epochs):
        dbg(f"=== Epoch {epoch+1}/{cfg.epochs} ===")
        epoch_loss = 0.0
        epoch_miou = 0.0
        epoch_per_class = torch.zeros(cfg.num_classes, device=cfg.device)
        t_epoch = time.time()

        loader_iter = iter(train_loader)
        step = -1
        while True:
            optimizer.zero_grad()
            try:
                t_fetch = time.time()
                batch = next(loader_iter)
                fetch_time = time.time() - t_fetch
            except StopIteration:
                break
            step += 1

            images, gt_mask, _, _ = batch

            t_fwd = time.time()
            classifications = model.forward(images).permute(0, 3, 1, 2)
            if cfg.device == "cuda":
                gt_mask = gt_mask.to("cuda", non_blocking=True)
            if step == 0 and epoch == start_epoch:
                with torch.no_grad():
                    logit_min = classifications.min().item()
                    logit_max = classifications.max().item()
                    logit_mean_abs = classifications.abs().mean().item()
                dbg(f"shape sanity: pred={tuple(classifications.shape)} "
                    f"gt={tuple(gt_mask.shape)} "
                    f"weight={tuple(cfg.class_weights.shape)} "
                    f"num_classes={cfg.num_classes}")
                dbg(f"logit stats: min={logit_min:.2f} max={logit_max:.2f} "
                    f"mean|x|={logit_mean_abs:.2f}")
            ce = criterion(classifications, gt_mask.argmax(dim=1).long())
            dl = dice_loss(classifications, gt_mask)
            loss = ce + 1.0 * dl

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg.grad_clip_norm)
            optimizer.step()
            if global_step < warmup_steps:
                warmup.step()

            loss_val = loss.item()  # implicit GPU sync
            step_time = time.time() - t_fwd

            epoch_loss += loss_val
            global_step += 1

            miou, per_class = compute_miou(classifications.detach(), gt_mask, cfg.num_classes)
            epoch_miou += miou
            epoch_per_class += per_class

            # TensorBoard
            writer.add_scalar("train/loss_step", loss_val, global_step)
            writer.add_scalar("train/mIoU_step", miou, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            for c, name in enumerate(class_names):
                writer.add_scalar(f"train/IoU_{name}_step", per_class[c].item(), global_step)
            writer.add_scalar("train/step_time", step_time, global_step)
            writer.add_scalar("train/fetch_time", fetch_time, global_step)

            miou_window.append(miou)
            loss_window.append(loss_val)
            if step < 5 or step % 10 == 0:
                pred_classes = classifications.argmax(dim=1).unique().tolist()
                ce_val, dl_val = ce.item(), dl.item()
                avg_miou = sum(miou_window) / len(miou_window)
                avg_loss = sum(loss_window) / len(loss_window)
                cur_lr = optimizer.param_groups[0]["lr"]
                pc = [f"{n}={per_class[i].item():.2f}"
                      for i, n in enumerate(class_names)]
                dbg(f"  step {step}: loss={loss_val:.3f} (ce={ce_val:.3f} dice={dl_val:.3f}) "
                    f"mIoU={miou:.4f} | avg(20) loss={avg_loss:.3f} mIoU={avg_miou:.4f} "
                    f"IoU[{' '.join(pc)}] "
                    f"lr={cur_lr:.2e} "
                    f"pred_classes={pred_classes} "
                    f"fetch={fetch_time:.2f}s fwd+bwd={step_time:.2f}s")

            # Early checkpoint
            # if epoch == 0 and step == 5:
            #     save_step_checkpoint(model.head, optimizer, epoch, global_step, loss_val, miou, cfg)
            # elif global_step > 0 and global_step % 10_000 == 0:
            #     save_step_checkpoint(model.head, optimizer, epoch, global_step, loss_val, miou, cfg)


        # Only decay between epochs once warmup has finished.
        if global_step >= warmup_steps:
            epoch_decay.step()
        if step < 0:
            dbg("WARNING: epoch produced 0 steps.")
            continue

        n_steps = step + 1
        avg_loss = epoch_loss / n_steps
        avg_miou = epoch_miou / n_steps
        epoch_per_class /= n_steps
        elapsed = time.time() - t_epoch

        # ---- validation ----
        val_loss, val_miou, val_per_class, val_confusion = evaluate(
            model, val_loader, criterion, cfg, class_names
        )

        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        writer.add_scalar("train/mIoU_epoch", avg_miou, epoch)
        writer.add_scalar("train/epoch_time_s", elapsed, epoch)
        for c, name in enumerate(class_names):
            writer.add_scalar(f"train/IoU_{name}", epoch_per_class[c].item(), epoch)
        if val_loss is not None:
            writer.add_scalar("val/loss_epoch", val_loss, epoch)
            writer.add_scalar("val/mIoU_epoch", val_miou, epoch)
            for c, name in enumerate(class_names):
                writer.add_scalar(f"val/IoU_{name}", val_per_class[c].item(), epoch)
        writer.flush()

        val_str = f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f}" if val_loss is not None else "val=n/a"
        dbg(f"Epoch {epoch+1}/{cfg.epochs}  loss={avg_loss:.4f}  mIoU={avg_miou:.4f}  {val_str}  ({elapsed:.0f}s)")

        # End-of-epoch diagnostics.
        if val_confusion is not None:
            log_confusion_matrix(val_confusion, class_names)
        # One-shot class pixel-frequency scan after epoch 1 (skip later epochs:
        # the distribution doesn't change and the full-loader pass is expensive).
        # if epoch == 0:
        #     log_class_pixel_distribution(
        #         train_loader, cfg.num_classes, class_names, "train"
        #     )

        effective_val_loss = val_loss if val_loss is not None else avg_loss
        best_val_loss = save_checkpoint(
            model.head, optimizer, epoch, avg_loss, avg_miou, effective_val_loss, cfg, best_val_loss,
            global_step=global_step,
        )

    writer.close()
    dbg("=== Training complete ===")


if __name__ == "__main__":
    train()