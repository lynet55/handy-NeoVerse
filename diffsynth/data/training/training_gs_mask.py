"""Training script for Gaussian mask parameters.

Approach: instead of a separate 2D prediction head, each Gaussian
carries mask logits as extra parameters (4 channels for 4 classes). These are
rendered through the rasterizer via alpha-compositing (same as RGB), producing
a 2D rendered mask that is supervised against ground-truth segmentation masks.

Gradient flow: GT mask loss → rendered 2D mask → rasterizer → 3D Gaussian mask logits

Only the gs_head (which predicts all Gaussian parameters including mask logits) is
trained. All other model components are frozen.
"""

import os
import time
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.models.model_manager import ModelManager
from diffsynth.data.SimpleHandObjectSegmentationDataset import HandObjectSegmentationDataset, ClipStreamSampler


def dbg(msg: str):
    print(f"[DBG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def homo_matrix_inverse(mat):
    """Invert a batch of 4x4 SE(3) matrices."""
    R = mat[..., :3, :3]
    t = mat[..., :3, 3:]
    R_inv = R.transpose(-1, -2)
    t_inv = -R_inv @ t
    inv = torch.zeros_like(mat)
    inv[..., :3, :3] = R_inv
    inv[..., :3, 3:] = t_inv
    inv[..., 3, 3] = 1.0
    return inv


@dataclass
class TrainConfig:
    img_shape: tuple = (280, 280)
    patch_size: int = 14
    embed_dim: int = 1024
    num_classes: int = 4

    batch_size: int = 4       # smaller than seg head training: full forward + rasterizer is heavier
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0

    frame_stride: int = 3
    val_fraction: float = 0.1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    reconstruction_model_path: str = "models/NeoVerse/reconstructor.ckpt"
    save_model_path_prefix: str = "models/NeoVerse/gs_mask_model"

    resume_from: str = "latest"

    # class weights: up-weight rare foreground classes
    class_weights: torch.Tensor = field(
        default_factory=lambda: torch.tensor([10.0, 10.0, 5.0, 1.0])
    )

    run_id: str = ""
    log_dir: str = ""
    num_workers: int = 2
    pin_memory: bool = True

    def __post_init__(self):
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not self.log_dir:
            self.log_dir = f"runs/neoverse_gs_mask_{self.run_id}"


# ---------------------------------------------------------------------------
# Model wrapper: freeze everything except gs_head (which now outputs mask logits)
# ---------------------------------------------------------------------------

class GsMaskReconstructor:
    """Runs the full WorldMirror forward pass + rasterization.

    Only gs_head is trainable. The rasterizer renders the per-Gaussian mask
    logits into a 2D mask image via alpha-compositing.
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

        # Freeze everything, then unfreeze only gs_renderer.gs_head
        n_train, n_total = 0, 0
        for name, param in self.reconstructor.named_parameters():
            n_total += 1
            if "gs_renderer.gs_head" in name or "gs_renderer.gs_head_dynamic" in name:
                param.requires_grad = True
                n_train += 1
            else:
                param.requires_grad = False
        dbg(f"Trainable: {n_train}/{n_total} params (gs_head only).")
        if n_train == 0:
            raise RuntimeError("No parameters matched 'gs_renderer.gs_head'.")

        # The checkpoint's gs_head.2 has shape [12, 256, 1, 1] (no mask channels).
        # load_state_dict(strict=False) silently skips mismatched shapes, which would
        # randomly reinitialise the whole final conv — losing pretrained geometry weights.
        # Fix: copy the pretrained 12 channels manually into channels 0:12 of the new [16, ...] layer.
        self._restore_pretrained_gs_head_weights(cfg.reconstruction_model_path, cfg.device)

        # Cast trainable parts to fp32 for numerical stability
        self.reconstructor.gs_renderer.gs_head.float()
        if hasattr(self.reconstructor.gs_renderer, "gs_head_dynamic"):
            self.reconstructor.gs_renderer.gs_head_dynamic.float()
        for p in self.reconstructor.gs_renderer.gs_head.parameters():
            p.requires_grad = True
        if hasattr(self.reconstructor.gs_renderer, "gs_head_dynamic"):
            for p in self.reconstructor.gs_renderer.gs_head_dynamic.parameters():
                p.requires_grad = True
        self.set_train_mode()
        dbg("gs_head cast to float32.")

    def _restore_pretrained_gs_head_weights(self, ckpt_path: str, device: str):
        """Copy pre-trained geometry channels into the expanded gs_head final conv.

        The checkpoint has gs_head.2.{weight,bias} with 12 output channels (quats/scales/
        opacities/sh/weights). Our new gs_head has 16 (12 + 4 mask logit channels).
        load_state_dict skips shape-mismatched tensors, so without this fix the entire
        final conv is randomly re-initialised, losing the pretrained geometry weights.
        We copy the pretrained 12 channels into positions 0:12 and leave 12:16 at their
        (small) random init values from GaussianSplatRenderer.__init__.
        """
        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Checkpoint may be a nested dict (e.g. {"state_dict": {...}}) or flat
        state = raw.get("state_dict", raw)

        for head_name in ["gs_renderer.gs_head", "gs_renderer.gs_head_dynamic"]:
            renderer = self.reconstructor.gs_renderer
            head = getattr(renderer, head_name.split(".")[-1], None)
            if head is None:
                continue
            final_conv = head[-1]  # nn.Conv2d

            w_key = f"{head_name}.2.weight"
            b_key = f"{head_name}.2.bias"

            if w_key not in state:
                dbg(f"  [{head_name}] key {w_key!r} not found in checkpoint — skipping restore.")
                continue

            pretrained_w = state[w_key].to(device=device, dtype=final_conv.weight.dtype)
            pretrained_b = state[b_key].to(device=device, dtype=final_conv.bias.dtype)
            n_pretrained = pretrained_w.shape[0]

            with torch.no_grad():
                final_conv.weight[:n_pretrained].copy_(pretrained_w)
                final_conv.bias[:n_pretrained].copy_(pretrained_b)
            dbg(f"  [{head_name}] restored {n_pretrained} pretrained channels into "
                f"[{final_conv.weight.shape[0]}-channel] final conv.")

    def set_train_mode(self):
        """Keep frozen modules deterministic while training only the GS mask head."""
        self.reconstructor.eval()
        self.reconstructor.gs_renderer.gs_head.train()
        if hasattr(self.reconstructor.gs_renderer, "gs_head_dynamic"):
            self.reconstructor.gs_renderer.gs_head_dynamic.train()

    def set_eval_mode(self):
        self.reconstructor.eval()
        self.reconstructor.gs_renderer.gs_head.eval()
        if hasattr(self.reconstructor.gs_renderer, "gs_head_dynamic"):
            self.reconstructor.gs_renderer.gs_head_dynamic.eval()

    def trainable_parameters(self):
        params = list(self.reconstructor.gs_renderer.gs_head.parameters())
        if hasattr(self.reconstructor.gs_renderer, "gs_head_dynamic"):
            params += list(self.reconstructor.gs_renderer.gs_head_dynamic.parameters())
        return params

    def forward(self, images: torch.Tensor):
        """Run full forward pass and rasterize mask channels.

        Args:
            images: [B, 3, H, W] — a batch of single frames.

        Returns:
            rendered_masks: [B, num_classes, H, W] rendered class logits.
        """
        B, C, H, W = images.shape
        # Treat each image in the batch as a 1-frame sequence
        cfg = self.cfg
        imgs = images.unsqueeze(1).to(cfg.device, non_blocking=True)  # [B, 1, 3, H, W]

        views = {
            "img": imgs,
            "is_target": torch.zeros((B, 1), dtype=torch.bool, device=cfg.device),
            "is_static": torch.ones((B, 1), dtype=torch.bool, device=cfg.device),
            "timestamp": torch.zeros((B, 1), dtype=torch.int64, device=cfg.device),
        }

        with torch.amp.autocast(cfg.device, dtype=torch.bfloat16):
            predictions = self.reconstructor(views, is_inference=False, use_motion=False)

        gaussians = predictions["splats"]
        input_c2w = predictions["rendered_extrinsics"]   # [B, 1, 4, 4]
        input_intrs = predictions["rendered_intrinsics"] # [B, 1, 3, 3]
        input_timestamps = predictions["rendered_timestamps"]  # [B, 1]

        # Rasterize per-batch: collect rendered masks [B, 1, H, W, C]
        batch_masks = []
        for b in range(B):
            w2c_b = homo_matrix_inverse(input_c2w[b])   # [1, 4, 4]
            _, _, _, rendered_masks = self.reconstructor.gs_renderer.rasterizer.forward(
                render_splats=[gaussians[b]],
                render_viewmats=[w2c_b],
                render_Ks=[input_intrs[b]],
                render_timestamps=[input_timestamps[b]],
                sh_degree=0,
                width=W,
                height=H,
            )
            if rendered_masks is None:
                raise RuntimeError(
                    "Rasterizer returned no rendered mask logits. Check that "
                    "GaussianSplatRenderer produces and propagates mask_logits."
                )
            if rendered_masks.ndim == 5 and rendered_masks.shape[0] == 1:
                rendered_masks = rendered_masks.squeeze(0)
            batch_masks.append(rendered_masks)  # [1, H, W, C]

        # [B, H, W, C] → [B, C, H, W] for loss
        rendered_masks = torch.cat(batch_masks, dim=0)          # [B, H, W, C]
        rendered_masks = rendered_masks.permute(0, 3, 1, 2)     # [B, C, H, W]
        return rendered_masks


# ---------------------------------------------------------------------------
# Dataset (same strided wrapper as training_25.py)
# ---------------------------------------------------------------------------

class StridedHandObjectDataset(HandObjectSegmentationDataset):
    def __init__(self, data_root: str, frame_stride: int = 1, streams=None, clip_names=None):
        self._frame_stride = frame_stride
        self._clip_names_filter = clip_names
        super().__init__(data_root=data_root, streams=streams)

    def _build_index(self) -> None:
        for npz_path in sorted(Path(self.data_root).glob("clip-*.npz")):
            clip_name = npz_path.stem
            if self._clip_names_filter is not None and clip_name not in self._clip_names_filter:
                continue
            npz = np.load(str(npz_path), mmap_mode="r")
            n_frames = next(npz[k].shape[0] for k in npz.files if k.startswith("images_"))
            for stream in self.streams:
                if f"images_{stream}" not in npz.files:
                    continue
                for frame_idx in range(0, n_frames, self._frame_stride):
                    self.samples.append(
                        {"clip_name": clip_name, "stream": stream, "frame_idx": frame_idx}
                    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def bce_loss_weighted(rendered_logits, gt, class_weights):
    """Per-class binary BCE-with-logits with class weighting.

    rendered_logits: [B, C, H, W] rendered mask logits from the rasterizer.
    gt: [B, C, H, W] one-hot floats.

    This is the "0/1 loss" the supervisor described: each Gaussian mask channel is treated
    as a binary foreground/background prediction per class.
    """
    w = class_weights.view(1, -1, 1, 1).to(rendered_logits.device)
    return F.binary_cross_entropy_with_logits(
        rendered_logits, gt.float(), weight=w.expand_as(rendered_logits)
    )


class DiceLoss:
    """Soft Dice loss for multi-class segmentation (ignores absent classes).

    pred is expected to be rendered logits. Softmax is applied across classes
    before computing Dice.
    """

    def __init__(self, smooth: float = 1.0):
        self.smooth = smooth

    def __call__(self, pred, target):
        pred = torch.softmax(pred, dim=1)
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
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_miou(pred, gt, num_classes=4):
    """pred: [B, C, H, W] probabilities or logits; gt: [B, C, H, W] one-hot."""
    pred_cls = pred.argmax(dim=1)
    gt_cls = gt.argmax(dim=1)
    intersection = torch.zeros(num_classes, device=pred.device)
    union = torch.zeros(num_classes, device=pred.device)
    for c in range(num_classes):
        p = (pred_cls == c)
        g = (gt_cls == c)
        intersection[c] = (p & g).sum()
        union[c] = (p | g).sum()
    iou = intersection / (union + 1e-6)
    return iou.mean().item(), iou


@torch.no_grad()
def compute_per_class_accuracy(pred, gt, num_classes=4):
    """Per-class pixel accuracy: fraction of GT-positive pixels predicted correctly."""
    pred_cls = pred.argmax(dim=1)
    gt_cls = gt.argmax(dim=1)
    acc = torch.zeros(num_classes, device=pred.device)
    for c in range(num_classes):
        g = (gt_cls == c)
        if g.sum() > 0:
            acc[c] = (pred_cls[g] == c).float().mean()
    return acc


def log_confusion_matrix(confusion: torch.Tensor, class_names):
    """Pretty-print row-normalised confusion matrix (rows=GT, cols=pred)."""
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


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, avg_miou, cfg, best_val_loss, global_step=None):
    params = {
        "gs_head": model.reconstructor.gs_renderer.gs_head.state_dict(),
    }
    if hasattr(model.reconstructor.gs_renderer, "gs_head_dynamic"):
        params["gs_head_dynamic"] = model.reconstructor.gs_renderer.gs_head_dynamic.state_dict()

    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": params,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "mIoU": avg_miou,
        "best_val_loss": best_val_loss,
        "run_id": cfg.run_id,
    }
    torch.save(ckpt, f"{cfg.save_model_path_prefix}_latest.ckpt")
    epoch_path = f"{cfg.save_model_path_prefix}_run{cfg.run_id}_epoch{epoch+1:03d}.ckpt"
    torch.save(ckpt, epoch_path)
    dbg(f"  -> saved {os.path.basename(epoch_path)}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        ckpt["best_val_loss"] = best_val_loss
        torch.save(ckpt, f"{cfg.save_model_path_prefix}_best.ckpt")
        dbg(f"  -> new best val (val_loss={best_val_loss:.4f})")
    return best_val_loss


def load_checkpoint(model, optimizer, cfg, train_loader_len: int):
    """Restore gs_head weights and optimizer state."""
    path_map = {"latest": f"{cfg.save_model_path_prefix}_latest.ckpt",
                "best":   f"{cfg.save_model_path_prefix}_best.ckpt"}
    path = path_map.get(cfg.resume_from, cfg.resume_from)
    if path is None or not os.path.exists(path):
        dbg(f"No checkpoint at {path!r} — starting fresh.")
        return 0, 0, float("inf")

    dbg(f"Resuming from {path} ...")
    ckpt = torch.load(path, map_location=cfg.device, weights_only=False)
    params = ckpt["model_state_dict"]
    model.reconstructor.gs_renderer.gs_head.load_state_dict(params["gs_head"], strict=True)
    if "gs_head_dynamic" in params and hasattr(model.reconstructor.gs_renderer, "gs_head_dynamic"):
        model.reconstructor.gs_renderer.gs_head_dynamic.load_state_dict(params["gs_head_dynamic"], strict=True)

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(cfg.device)

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    global_step = int(ckpt.get("global_step", start_epoch * max(1, train_loader_len)))
    best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
    dbg(f"Resumed: start_epoch={start_epoch}  global_step={global_step}  best_val_loss={best_val_loss:.4f}")
    return start_epoch, global_step, best_val_loss


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, dice_loss_fn, cfg, class_names):
    model.set_eval_mode()
    total_loss = 0.0
    total_miou = 0.0
    n_steps = 0
    per_class_iou = torch.zeros(cfg.num_classes, device=cfg.device)
    per_class_acc = torch.zeros(cfg.num_classes, device=cfg.device)
    K = cfg.num_classes
    confusion = torch.zeros(K, K, dtype=torch.long, device=cfg.device)

    for batch in val_loader:
        images, gt_mask, _, _ = batch
        rendered = model.forward(images)  # [B, C, H, W] logits
        gt_mask = gt_mask.to(cfg.device, non_blocking=True)
        if rendered.shape[-2:] != gt_mask.shape[-2:]:
            gt_mask = F.interpolate(gt_mask, size=rendered.shape[-2:], mode="nearest")

        bce = bce_loss_weighted(rendered, gt_mask, cfg.class_weights)
        dl = dice_loss_fn(rendered, gt_mask)
        loss = bce + dl

        miou, pc_iou = compute_miou(rendered, gt_mask, cfg.num_classes)
        pc_acc = compute_per_class_accuracy(rendered, gt_mask, cfg.num_classes)

        pred_flat = rendered.argmax(dim=1).reshape(-1)
        gt_flat = gt_mask.argmax(dim=1).reshape(-1)
        idx = gt_flat * K + pred_flat
        confusion += torch.bincount(idx, minlength=K * K).reshape(K, K)

        total_loss += loss.item()
        total_miou += miou
        per_class_iou += pc_iou
        per_class_acc += pc_acc
        n_steps += 1

    model.set_train_mode()
    if n_steps == 0:
        return None, None, None, None, None
    return (
        total_loss / n_steps,
        total_miou / n_steps,
        per_class_iou / n_steps,
        per_class_acc / n_steps,
        confusion,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    dbg("=== train() [gs_mask] ===")
    cfg = TrainConfig()
    dbg(f"device={cfg.device}, batch={cfg.batch_size}, epochs={cfg.epochs}, lr={cfg.learning_rate}")

    model = GsMaskReconstructor(cfg)
    model.set_train_mode()

    # Dataset split
    all_clips = sorted(p.stem for p in Path("diffsynth/data/training_data").glob("clip-*.npz"))
    n_val = max(1, int(len(all_clips) * cfg.val_fraction))
    val_clips = set(all_clips[-n_val:])
    train_clips = set(all_clips[:-n_val])
    dbg(f"Clips: {len(all_clips)} total → {len(train_clips)} train / {len(val_clips)} val")

    train_ds = StridedHandObjectDataset("diffsynth/data/training_data", cfg.frame_stride, clip_names=train_clips)
    val_ds   = StridedHandObjectDataset("diffsynth/data/training_data", cfg.frame_stride, clip_names=val_clips)
    dbg(f"Samples: {len(train_ds)} train / {len(val_ds)} val")

    train_loader = DataLoader(train_ds, sampler=ClipStreamSampler(train_ds, shuffle_clips=True),
                              batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                              pin_memory=cfg.pin_memory, persistent_workers=(cfg.num_workers > 0))
    val_loader   = DataLoader(val_ds, sampler=ClipStreamSampler(val_ds, shuffle_clips=False),
                              batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                              pin_memory=cfg.pin_memory, persistent_workers=(cfg.num_workers > 0))

    dice_loss_fn = DiceLoss()
    dbg(f"class weights (bg=1): {cfg.class_weights.tolist()}")

    # Optimizer + schedulers
    trainable = model.trainable_parameters()
    dbg(f"Optimizer: {len(trainable)} trainable tensors")
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    warmup_steps = 100
    warmup   = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    decay    = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.80)

    # Resume
    try:
        train_loader_len = len(train_loader)
    except TypeError:
        train_loader_len = 0
    start_epoch, global_step, best_val_loss = load_checkpoint(model, optimizer, cfg, train_loader_len)

    # Fast-forward schedulers
    if start_epoch > 0 or global_step > 0:
        if global_step >= warmup_steps:
            target_lr = cfg.learning_rate * (0.80 ** start_epoch)
            warmup.last_epoch = warmup_steps
            decay.last_epoch  = start_epoch - 1
        else:
            frac = global_step / warmup_steps
            target_lr = cfg.learning_rate * (0.01 + 0.99 * frac)
            warmup.last_epoch = global_step
            decay.last_epoch  = -1
        for pg in optimizer.param_groups:
            pg["lr"] = target_lr
        dbg(f"Schedulers fast-forwarded: target_lr={target_lr:.2e}")

    class_names = ["right_hand", "left_hand", "object", "background"]
    writer = SummaryWriter(log_dir=cfg.log_dir, flush_secs=10)
    os.makedirs(os.path.dirname(cfg.save_model_path_prefix), exist_ok=True)

    for epoch in range(start_epoch, cfg.epochs):
        dbg(f"=== Epoch {epoch+1}/{cfg.epochs} ===")
        epoch_loss = 0.0
        epoch_miou = 0.0
        epoch_per_class_iou = torch.zeros(cfg.num_classes, device=cfg.device)
        epoch_per_class_acc = torch.zeros(cfg.num_classes, device=cfg.device)
        t_epoch = time.time()

        miou_window = deque(maxlen=20)
        loss_window = deque(maxlen=20)

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
            gt_mask = gt_mask.to(cfg.device, non_blocking=True)

            t_fwd = time.time()
            rendered = model.forward(images)  # [B, C, H, W] logits

            if rendered.shape[-2:] != gt_mask.shape[-2:]:
                gt_mask = F.interpolate(gt_mask, size=rendered.shape[-2:], mode="nearest")

            # First-step sanity diagnostics
            if step == 0 and epoch == start_epoch:
                with torch.no_grad():
                    dbg(f"shape sanity: pred={tuple(rendered.shape)} gt={tuple(gt_mask.shape)} "
                        f"num_classes={cfg.num_classes}")
                    dbg(f"rendered stats: min={rendered.min().item():.3f} "
                        f"max={rendered.max().item():.3f} "
                        f"mean={rendered.mean().item():.3f}")
                    pred_counts = {name: (rendered.argmax(dim=1) == i).sum().item()
                                   for i, name in enumerate(class_names)}
                    dbg(f"pred class pixel counts: {pred_counts}")

            bce = bce_loss_weighted(rendered, gt_mask, cfg.class_weights)
            dl = dice_loss_fn(rendered, gt_mask)
            loss = bce + dl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg.grad_clip_norm)
            optimizer.step()
            if global_step < warmup_steps:
                warmup.step()

            loss_val = loss.item()
            step_time = time.time() - t_fwd
            epoch_loss += loss_val
            global_step += 1

            miou, per_class_iou = compute_miou(rendered.detach(), gt_mask, cfg.num_classes)
            per_class_acc = compute_per_class_accuracy(rendered.detach(), gt_mask, cfg.num_classes)
            epoch_miou += miou
            epoch_per_class_iou += per_class_iou
            epoch_per_class_acc += per_class_acc

            # TensorBoard — per step
            writer.add_scalar("train/loss_step", loss_val, global_step)
            writer.add_scalar("train/bce_step", bce.item(), global_step)
            writer.add_scalar("train/dice_step", dl.item(), global_step)
            writer.add_scalar("train/mIoU_step", miou, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/step_time", step_time, global_step)
            writer.add_scalar("train/fetch_time", fetch_time, global_step)
            for c, name in enumerate(class_names):
                writer.add_scalar(f"train/IoU_{name}_step", per_class_iou[c].item(), global_step)
                writer.add_scalar(f"train/Acc_{name}_step", per_class_acc[c].item(), global_step)

            miou_window.append(miou)
            loss_window.append(loss_val)
            if step < 5 or step % 10 == 0:
                avg_miou_w = sum(miou_window) / len(miou_window)
                avg_loss_w = sum(loss_window) / len(loss_window)
                pred_classes = rendered.argmax(dim=1).unique().tolist()
                pc_iou = [f"{n}={per_class_iou[i].item():.2f}" for i, n in enumerate(class_names)]
                pc_acc = [f"{n}={per_class_acc[i].item():.2f}" for i, n in enumerate(class_names)]
                dbg(
                    f"  step {step}: loss={loss_val:.3f} "
                    f"(bce={bce.item():.3f} dice={dl.item():.3f}) "
                    f"mIoU={miou:.4f} | avg(20) loss={avg_loss_w:.3f} mIoU={avg_miou_w:.4f} "
                    f"IoU[{' '.join(pc_iou)}] Acc[{' '.join(pc_acc)}] "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"pred_classes={pred_classes} "
                    f"fetch={fetch_time:.2f}s fwd+bwd={step_time:.2f}s"
                )

        if global_step >= warmup_steps:
            decay.step()
        if step < 0:
            dbg("WARNING: epoch produced 0 steps.")
            continue

        n_steps = step + 1
        avg_loss = epoch_loss / n_steps
        avg_miou = epoch_miou / n_steps
        epoch_per_class_iou /= n_steps
        epoch_per_class_acc /= n_steps
        elapsed = time.time() - t_epoch

        # Validation
        val_loss, val_miou, val_per_class_iou, val_per_class_acc, val_confusion = evaluate(
            model, val_loader, dice_loss_fn, cfg, class_names
        )

        # TensorBoard — per epoch (train)
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        writer.add_scalar("train/mIoU_epoch", avg_miou, epoch)
        writer.add_scalar("train/epoch_time_s", elapsed, epoch)
        for c, name in enumerate(class_names):
            writer.add_scalar(f"train/IoU_{name}", epoch_per_class_iou[c].item(), epoch)
            writer.add_scalar(f"train/Acc_{name}", epoch_per_class_acc[c].item(), epoch)

        # TensorBoard — per epoch (val)
        if val_loss is not None:
            writer.add_scalar("val/loss_epoch", val_loss, epoch)
            writer.add_scalar("val/mIoU_epoch", val_miou, epoch)
            for c, name in enumerate(class_names):
                writer.add_scalar(f"val/IoU_{name}", val_per_class_iou[c].item(), epoch)
                writer.add_scalar(f"val/Acc_{name}", val_per_class_acc[c].item(), epoch)

        writer.flush()

        val_str = (f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f}"
                   if val_loss is not None else "val=n/a")
        dbg(f"Epoch {epoch+1}/{cfg.epochs}  loss={avg_loss:.4f}  mIoU={avg_miou:.4f}  "
            f"{val_str}  ({elapsed:.0f}s)")

        if val_confusion is not None:
            log_confusion_matrix(val_confusion, class_names)

        effective_val = val_loss if val_loss is not None else avg_loss
        best_val_loss = save_checkpoint(
            model, optimizer, epoch, avg_loss, effective_val, avg_miou, cfg, best_val_loss, global_step
        )

    writer.close()
    dbg("=== Training complete ===")


if __name__ == "__main__":
    train()
