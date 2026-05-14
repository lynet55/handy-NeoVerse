import numpy as np
import torch
import torch.nn.functional as F

# Rendering Metric Imports
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips

# ===================================================================== #
#                           SEGMENTATION METRICS                        #
# ===================================================================== #

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
    device = pred_mask.device
    pred = pred_mask.argmax(dim=1).unsqueeze(1).float() # (B, 1, H, W)
    gt = gt_mask.argmax(dim=1).unsqueeze(1).float()     # (B, 1, H, W)

    bf1_per_class = []
    k_size = 2 * tolerance + 1

    for c in range(num_classes):
        p_c = (pred == c).float()
        g_c = (gt == c).float()

        p_boundary = (p_c > 0.5) & ((1 - F.max_pool2d(1 - p_c, 3, 1, 1)) < 0.5)
        g_boundary = (g_c > 0.5) & ((1 - F.max_pool2d(1 - g_c, 3, 1, 1)) < 0.5)

        p_dil = F.max_pool2d(p_boundary.float(), kernel_size=k_size, stride=1, padding=tolerance) > 0.5
        g_dil = F.max_pool2d(g_boundary.float(), kernel_size=k_size, stride=1, padding=tolerance) > 0.5

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


# ===================================================================== #
#                            UNIFIED EVALUATOR                          #
# ===================================================================== #

class BenchmarkEvaluator:
    """
    A unified evaluation suite for 4D Gaussian Splatting and NeoVerse models.
    Handles both Segmentation metrics (mIoU, Pixel Accuracy, BF1) and 
    Rendering metrics (PSNR, SSIM, LPIPS).

    Attributes:
        run_seg (bool): If True, computes mask-based metrics.
        run_render (bool): If True, computes RGB-based metrics.
        num_classes (int): Number of segmentation categories.

    Inputs required for update():
        - pred_mask: [B, C, H, W] tensor of raw logits.
        - gt_mask:   [B, C, H, W] tensor of ground truth.
        - pred_rgb:  [B, 3, H, W] tensor, range [0, 1].
        - gt_rgb:    [B, 3, H, W] tensor, range [0, 1].

    Example Usage:
         evaluator = BenchmarkEvaluator(device="cuda", run_seg=True, run_render=True)
         for batch in dataloader:
             preds = model(batch)
             evaluator.update(
                 pred_mask=preds['logits'], 
                 gt_mask=batch['mask'],
                 pred_rgb=preds['render'], 
                 gt_rgb=batch['image']
             )
         results = evaluator.compute()
         # Save to file
         with open("metrics.json", "w") as f:
             json.dump(results, f, indent=4)

    Metric Interpretation:
        - PSNR (dB) ↑: Pixel-wise accuracy.
        - SSIM ↑: Structural/texture similarity.
        - LPIPS ↓: Perceptual distance (0.0 is perfect).
        - mIoU ↑: Mean Intersection over Union across classes.
        - BF1 ↑: Boundary F1 score for edge accuracy.
    """
    def __init__(self, device="cuda", num_classes=4, bf1_tolerance=2, run_seg=True, run_render=True):
        """
        Initializes the metric states.
        By setting run_seg and run_render, you load only the necessary tools into memory.
        """
        self.device = device
        self.num_classes = num_classes
        self.bf1_tolerance = bf1_tolerance
        
        self.run_seg = run_seg
        self.run_render = run_render

        # Render Metrics Setup
        if self.run_render:
            self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            # Standard VGG backbone for LPIPS. Loaded only once to save memory/compute
            self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
            
            # Since LPIPS isn't a torchmetric standard object, we store values to average later
            self.lpips_accumulator = []

        # Segmentation Metric Setup
        if self.run_seg:
            self.seg_accumulators = {
                    "miou": [], "iou_per_class": [], "bf1": [], 
                    "bf1_per_class": [], "pixel_accuracy": [], "pixel_acc_per_class": []
            }

    @torch.no_grad()
    def update(self, pred_mask=None, gt_mask=None, pred_rgb=None, gt_rgb=None):
        """
        Accumulates metrics for the current batch.
        
        Args:
            pred_mask (Tensor, optional): Model logits of shape (B, C, H, W).
            gt_mask (Tensor, optional): Target masks of shape (B, C, H, W).
            pred_rgb (Tensor, optional): Rendered images in range [0, 1] of shape (B, 3, H, W).
            gt_rgb (Tensor, optional): Ground truth images in range [0, 1] of shape (B, 3, H, W).
        """
        # --- Update Segmentation ---
        if self.run_seg:
            assert pred_mask is not None and gt_mask is not None, "Segmentation requires pred_mask and gt_mask"
            
            iou_res = compute_mIoU(pred_mask, gt_mask, self.num_classes)
            self.seg_accumulators["miou"].append(iou_res["miou"])
            self.seg_accumulators["iou_per_class"].append(iou_res["iou_per_class"])
            
            acc_res = pixel_accuracy(pred_mask, gt_mask, self.num_classes)
            self.seg_accumulators["pixel_accuracy"].append(acc_res["pixel_accuracy"])
            self.seg_accumulators["pixel_acc_per_class"].append(acc_res["pixel_accuracy_per_class"])

            bf1_res = boundary_f1(pred_mask, gt_mask, self.num_classes, self.bf1_tolerance)
            self.seg_accumulators["bf1"].append(bf1_res["bf1"])
            self.seg_accumulators["bf1_per_class"].append(bf1_res["bf1_per_class"])

        # --- Update Rendering ---
        if self.run_render:
            assert pred_rgb is not None and gt_rgb is not None, "Rendering requires pred_rgb and gt_rgb"
            
            # Torchmetrics update
            self.psnr_metric.update(pred_rgb, gt_rgb)
            self.ssim_metric.update(pred_rgb, gt_rgb)
            
            # LPIPS update (Scale [0, 1] -> [-1, 1])
            lp_preds = (pred_rgb * 2.0) - 1.0
            lp_target = (gt_rgb * 2.0) - 1.0
            dist = self.lpips_fn(lp_preds, lp_target)
            self.lpips_accumulator.append(dist.mean().item())

    def compute(self) -> dict:
        """
        Computes and returns the final averaged metrics across all seen batches.
        """
        final_metrics = {}

        if self.run_seg:
            final_metrics["mIoU"] = float(np.mean(self.seg_accumulators["miou"]))
            final_metrics["Pixel_Accuracy"] = float(np.mean(self.seg_accumulators["pixel_accuracy"]))
            final_metrics["Boundary_F1"] = float(np.mean(self.seg_accumulators["bf1"]))

            final_metrics["IoU_Per_class"] = np.mean(self.seg_accumulators["iou_per_class"], axis=0).tolist()
            final_metrics["Pixel_Acc_per_class"] = np.mean(self.seg_accumulators["pixel_acc_per_class"], axis=0).tolist()

        if self.run_render:
            final_metrics["PSNR"] = self.psnr_metric.compute().item()
            final_metrics["SSIM"] = self.ssim_metric.compute().item()
            final_metrics["LPIPS"] = float(np.mean(self.lpips_accumulator))

        return final_metrics

    def reset(self):
        """ Clears all accumulators for a new epoch/validation run. """
        if self.run_render:
            self.psnr_metric.reset()
            self.ssim_metric.reset()
            self.lpips_accumulator = []
        
        if self.run_seg:
            self.seg_accumulators = {k: [] for k in self.seg_accumulators}

# ===================================================================== #
#                           EXAMPLE USAGE                               #
# ===================================================================== #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate some dummy data
    B, C_seg, C_rgb, H, W = 2, 4, 3, 256, 256
    dummy_pred_mask = torch.randn(B, C_seg, H, W, device=device)
    dummy_gt_mask = torch.randn(B, C_seg, H, W, device=device)

    dummy_pred_rgb = torch.rand(B, C_rgb, H, W, device=device) # [0, 1]
    dummy_gt_rgb = torch.rand(B, C_rgb, H, W, device=device)   # [0, 1]

    print("--- 1. Evaluating Segmentation ONLY ---")
    evaluator_seg = BenchmarkEvaluator(device=device, run_seg=True, run_render=False)
    evaluator_seg.update(pred_mask=dummy_pred_mask, gt_mask=dummy_gt_mask)
    print(evaluator_seg.compute())

    print("\n--- 2. Evaluating Rendering ONLY ---")
    evaluator_ren = BenchmarkEvaluator(device=device, run_seg=False, run_render=True)
    evaluator_ren.update(pred_rgb=dummy_pred_rgb, gt_rgb=dummy_gt_rgb)
    print(evaluator_ren.compute())

    print("\n--- 3. Evaluating BOTH ---")
    evaluator_both = BenchmarkEvaluator(device=device, run_seg=True, run_render=True)
    evaluator_both.update(
        pred_mask=dummy_pred_mask, 
        gt_mask=dummy_gt_mask,
        pred_rgb=dummy_pred_rgb, 
        gt_rgb=dummy_gt_rgb
    )
    print(evaluator_both.compute())
