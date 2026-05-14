import numpy as np
import torch
import torch.nn.functional as F

# --- Metric for use in benchmarking of the segmentation masks ---
"""
    Metrics for benchmarking segmentation
    The metrics that are going to be used are:
    - mIoU
    - Boundary F1
    - Pixel accuracy
"""

# mIoU
@torch.no_grad()
def compute_mIoU(pred_mask, gt_mask, num_classes=4):
    pred = pred_mask.argmax(dim=1)
    gt = gt_mask.argmax(dim=1)
    intersection = torch.zeros(num_classes, device=pred.device)
    union = torch.zeros(num_classes, device=pred.device)
    for c in range(num_classes):
        p = (pred == c)
        g = (gt == c)
        intersection[c] = (p & g).sum()
        union[c] = (p | g).sum()
    iou = intersection / (union + 1e-6)
    return {
        "miou": iou.mean().item(),
        "iou_per_class": iou.tolist(),
    }

# Boundary F1
@torch.no_grad()
def boundary_f1(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int = 4, tolerance: int = 2) -> dict:
    """
    GPU-accelerated Boundary F1 calculation.
    pred_mask, gt_mask: (B, C, H, W) logit/probability tensors
    tolerance: pixel radius for correctness
    """
    device = pred_mask.device
    pred = pred_mask.argmax(dim=1).unsqueeze(1).float() # (B, 1, H, W)
    gt = gt_mask.argmax(dim=1).unsqueeze(1).float()     # (B, 1, H, W)

    bf1_per_class = []
    
    # Kernel size for tolerance-based dilation
    k_size = 2 * tolerance + 1

    for c in range(num_classes):
        p_c = (pred == c).float()
        g_c = (gt == c).float()

        # Get boundaries (B, 1, H, W)
        p_boundary = (p_c > 0.5) & ((1 - F.max_pool2d(1 - p_c, 3, 1, 1)) < 0.5)
        g_boundary = (g_c > 0.5) & ((1 - F.max_pool2d(1 - g_c, 3, 1, 1)) < 0.5)

        # Dilate for tolerance
        p_dil = F.max_pool2d(p_boundary.float(), kernel_size=k_size, stride=1, padding=tolerance) > 0.5
        g_dil = F.max_pool2d(g_boundary.float(), kernel_size=k_size, stride=1, padding=tolerance) > 0.5

        # Precision: % of predicted boundary pixels near a GT boundary pixel
        # Recall: % of GT boundary pixels near a predicted boundary pixel
        precision = (p_boundary & g_dil).sum() / (p_boundary.sum() + 1e-8)
        recall = (g_boundary & p_dil).sum() / (g_boundary.sum() + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        bf1_per_class.append(f1.item())
    
    return {
        "bf1": np.mean(bf1_per_class),
        "bf1_per_class": bf1_per_class,
    }


# Pixel accuracy
@torch.no_grad()
def pixel_accuracy(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int = 4) -> dict:
    pred = pred_mask.argmax(dim=1)
    gt = gt_mask.argmax(dim=1)

    overall_acc = (pred == gt).float().mean().item()

    acc_per_class = []
    for c in range(num_classes):
        gt_c = (gt == c)
        acc = (pred[gt_c] == c).float().mean().item() if gt_c.any() else 0.0
        acc_per_class.append(acc)
    return {
        "pixel_accuracy": overall_acc,
        "pixel_accuracy_per_class": acc_per_class,
    }


@torch.no_grad()
def evaluate(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int = 4, bf1_tolerance: int = 2, pred_rgb=None, gt_rgb=None) -> dict:
    """
    pred_mask, gt_mask: (B, C, H, W) logit tensors

    To add more mask metrics, see marked sections below.
    For rendering metrics (PSNR, SSIM, LPIPS), pass renders in and call eval_masked_render - see comment.
    """
    metrics = {}

    # --- Segmentation metrics ---
    metrics.update(compute_mIoU(pred_mask, gt_mask, num_classes))
    metrics.update(pixel_accuracy(pred_mask, gt_mask, num_classes))
    metrics.update(boundary_f1(pred_mask, gt_mask, num_classes, bf1_tolerance))

    # ADD MORE SEGMENTATION METRICS HERE
    
    return metrics

