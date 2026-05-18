"""Reconstruction + segmentation benchmark for the NeoVerse reconstructor.

Mirrors the data path of ``training_25_04.py`` and the inference path of
``reconstruction_demo.py`` so the numbers we report match what the model
actually does in training and serving.

Two metric families, both on input viewpoints (no novel-view here):
  - Segmentation: mIoU, per-class IoU, pixel accuracy, boundary-F1.
    mIoU/IoU/accuracy come from a single accumulated confusion matrix —
    not per-batch averaging, which is biased when class frequencies vary.
  - Rendering: PSNR / SSIM / LPIPS via torchmetrics, on the splat re-render
    at the input cameras.

Aggregation: metrics are computed per (clip, stream) window, then averaged
across (clip, stream) groups so long clips don't dominate.

Output:
  outputs/benchmark/<run_id>/
    config.json
    per_clip.csv        # one row per (clip, stream)
    aggregate.json      # means + confusion matrix
    baselines.json      # majority-class mIoU
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.data.SimpleHandObjectSegmentationDataset import STREAMS
from diffsynth.models.model_manager import ModelManager
from diffsynth.utils.auxiliary import homo_matrix_inverse


CLASS_NAMES = ["right_hand", "left_hand", "object", "background"]


# ===================================================================== #
#                                 DATA                                  #
# ===================================================================== #


def get_val_clips(data_root: str, val_fraction: float) -> set[str]:
    """Reproduce the train/val split used by training_25_04.py.

    Both scripts must derive the split from the same sorted glob and the
    same fraction, otherwise the benchmark leaks training clips.
    """
    all_clips = sorted(p.stem for p in Path(data_root).glob("clip-*.npz"))
    if not all_clips:
        raise FileNotFoundError(f"No clip-*.npz under {data_root}")
    n_val = max(1, int(len(all_clips) * val_fraction))
    return set(all_clips[-n_val:])


class ClipWindowDataset(Dataset):
    """One sample = one window of consecutive frames from a single (clip, stream).

    The reconstructor consumes frames as a sequence S; we therefore pre-group
    frames into windows of size ``window_size``. With ``batch_size=1`` this
    yields ``[1, S, 3, H, W]`` images and ``[1, S, 4, H, W]`` one-hot masks —
    the exact shape the reconstructor and the demo expect.
    """

    def __init__(
        self,
        data_root: str,
        window_size: int,
        frame_stride: int,
        clip_names: set[str] | None,
        streams: list[str] | None = None,
    ):
        self.data_root = Path(data_root)
        self.window_size = window_size
        self.streams = streams or STREAMS
        self.windows: list[tuple[str, str, list[int]]] = []

        for npz_path in sorted(self.data_root.glob("clip-*.npz")):
            clip = npz_path.stem
            if clip_names is not None and clip not in clip_names:
                continue
            npz = np.load(str(npz_path), mmap_mode="r")
            n_frames = next(
                npz[k].shape[0] for k in npz.files if k.startswith("images_")
            )
            sampled = list(range(0, n_frames, frame_stride))
            for stream in self.streams:
                if f"images_{stream}" not in npz.files:
                    continue
                for i in range(0, len(sampled), window_size):
                    win = sampled[i : i + window_size]
                    if len(win) < 2:
                        continue
                    self.windows.append((clip, stream, win))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        clip, stream, frame_idxs = self.windows[idx]
        npz = np.load(str(self.data_root / f"{clip}.npz"), mmap_mode="r")

        images, masks = [], []
        for f in frame_idxs:
            img = torch.tensor(
                npz[f"images_{stream}"][f], dtype=torch.float32
            ).permute(2, 0, 1) / 255.0
            right = torch.tensor(npz[f"masks_{stream}_hand_RIGHT"][f] > 0)
            left = torch.tensor(npz[f"masks_{stream}_hand_LEFT"][f] > 0)
            obj = torch.tensor(npz[f"masks_{stream}_object"][f] > 0)
            bg = ~(right | left | obj)
            mask = torch.stack([right, left, obj, bg], dim=0).float()
            images.append(img)
            masks.append(mask)

        return (
            torch.stack(images),   # [S, 3, H, W]
            torch.stack(masks),    # [S, 4, H, W]
            clip,
            stream,
        )


# ===================================================================== #
#                              METRICS                                  #
# ===================================================================== #


def confusion_to_metrics(cm: torch.Tensor) -> dict:
    cm = cm.float()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / (tp + fp + fn + 1e-9)
    acc_per_class = tp / (cm.sum(1) + 1e-9)
    overall_acc = tp.sum() / (cm.sum() + 1e-9)
    return {
        "mIoU": iou.mean().item(),
        "IoU_per_class": iou.tolist(),
        "pixel_accuracy": overall_acc.item(),
        "pixel_accuracy_per_class": acc_per_class.tolist(),
    }


@torch.no_grad()
def boundary_f1_indices(
    pred_idx: torch.Tensor,
    gt_idx: torch.Tensor,
    num_classes: int,
    tolerance: int,
) -> list[float]:
    """Boundary F1 per class, computed on class-index tensors [N, H, W]."""
    pred = pred_idx.unsqueeze(1).float()
    gt = gt_idx.unsqueeze(1).float()
    k = 2 * tolerance + 1
    out = []
    for c in range(num_classes):
        p_c = (pred == c).float()
        g_c = (gt == c).float()
        p_b = (p_c > 0.5) & ((1 - F.max_pool2d(1 - p_c, 3, 1, 1)) < 0.5)
        g_b = (g_c > 0.5) & ((1 - F.max_pool2d(1 - g_c, 3, 1, 1)) < 0.5)
        p_dil = F.max_pool2d(p_b.float(), k, 1, tolerance) > 0.5
        g_dil = F.max_pool2d(g_b.float(), k, 1, tolerance) > 0.5
        prec = (p_b & g_dil).sum() / (p_b.sum() + 1e-8)
        rec = (g_b & p_dil).sum() / (g_b.sum() + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        out.append(f1.item())
    return out


# ===================================================================== #
#                            EVALUATOR                                  #
# ===================================================================== #


class BenchmarkEvaluator:
    """Runs the reconstructor over a dataloader and accumulates metrics.

    Segmentation:
      - Global confusion matrix for mIoU / per-class IoU / pixel accuracy.
      - Per-(clip, stream) confusion matrix for per-clip rows in the CSV.
      - BF1 averaged per-window then per-clip.

    Rendering:
      - Global torchmetrics PSNR/SSIM/LPIPS plus per-clip accumulators.
    """

    def __init__(
        self,
        reconstructor,
        dataloader,
        device: str = "cuda",
        num_classes: int = 4,
        bf1_tolerance: int = 2,
        run_seg: bool = True,
        run_render: bool = True,
        resolution: tuple[int, int] = (560, 336),
    ):
        self.reconstructor = reconstructor
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes
        self.bf1_tolerance = bf1_tolerance
        self.run_seg = run_seg
        self.run_render = run_render
        self.res_w, self.res_h = resolution

        if run_seg:
            self.cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
            self.per_clip_cm: dict[tuple[str, str], torch.Tensor] = defaultdict(
                lambda: torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
            )
            self.per_clip_bf1: dict[tuple[str, str], list[list[float]]] = defaultdict(list)

        if run_render:
            self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            self.lpips_m = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=True
            ).to(device)
            # Separate instance for per-window scores so .reset() doesn't
            # clobber the global accumulator. Sharing the VGG weights would
            # be nice but torchmetrics doesn't expose that cleanly.
            self.lpips_window = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=True
            ).to(device)
            self.per_clip_render: dict[tuple[str, str], dict] = defaultdict(
                lambda: {"psnr": [], "ssim": [], "lpips": []}
            )

        # Majority-class baseline (segmentation): pixel-frequency over GT.
        self.gt_pixel_counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def _step(
        self,
        images_seq: torch.Tensor,   # [1, S, 3, H, W]
        gt_mask: torch.Tensor,      # [S, 4, H, W]
        clip: str,
        stream: str,
    ):
        B, S = images_seq.shape[:2]
        views = {
            "img": images_seq,
            "is_target": torch.zeros((B, S), dtype=torch.bool, device=self.device),
            "is_static": torch.zeros((B, S), dtype=torch.bool, device=self.device),
            "timestamp": torch.arange(S, dtype=torch.int64, device=self.device)
            .unsqueeze(0)
            .expand(B, -1),
        }

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = self.reconstructor(views, is_inference=True, use_motion=False)

        key = (clip, stream)

        if self.run_seg:
            # predictions["seg_labels"]: [B, S, H, W, C] channels-last
            cls = predictions["seg_labels"][0]                # [S, H, W, C]
            pred_logits = cls.permute(0, 3, 1, 2).float()     # [S, C, H, W]
            pred_idx = pred_logits.argmax(dim=1)              # [S, H, W]
            gt_idx = gt_mask.argmax(dim=1)                    # [S, H, W]

            k = self.num_classes
            flat = gt_idx.reshape(-1) * k + pred_idx.reshape(-1)
            cm_step = torch.bincount(flat, minlength=k * k).reshape(k, k)
            self.cm += cm_step
            self.per_clip_cm[key] += cm_step

            self.gt_pixel_counts += torch.bincount(
                gt_idx.reshape(-1), minlength=k
            )

            bf1 = boundary_f1_indices(pred_idx, gt_idx, k, self.bf1_tolerance)
            self.per_clip_bf1[key].append(bf1)

        if self.run_render:
            gaussians = predictions["splats"]
            input_c2w = predictions["rendered_extrinsics"][0]
            input_intrs = predictions["rendered_intrinsics"][0]
            input_ts = predictions["rendered_timestamps"][0]
            input_w2c = homo_matrix_inverse(input_c2w)

            target_rgb, _, _ = self.reconstructor.gs_renderer.rasterizer.forward(
                gaussians,
                render_viewmats=[input_w2c],
                render_Ks=[input_intrs],
                render_timestamps=[input_ts],
                sh_degree=0,
                width=self.res_w,
                height=self.res_h,
                render_classes=[0, 1, 0, 1],
            )
            # target_rgb: [1, S, H, W, 3] in [0, 1]
            pred_rgb = target_rgb[0].permute(0, 3, 1, 2).clamp(0, 1).float()
            gt_rgb = images_seq[0].float()

            self.psnr.update(pred_rgb, gt_rgb)
            self.ssim.update(pred_rgb, gt_rgb)
            # torchmetrics LPIPS with normalize=True expects inputs in [0, 1].
            self.lpips_m.update(pred_rgb, gt_rgb)

            # Per-clip — recompute against window only.
            psnr_w = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            ssim_w = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            psnr_w.update(pred_rgb, gt_rgb)
            ssim_w.update(pred_rgb, gt_rgb)
            self.per_clip_render[key]["psnr"].append(psnr_w.compute().item())
            self.per_clip_render[key]["ssim"].append(ssim_w.compute().item())
            self.lpips_window.reset()
            self.lpips_window.update(pred_rgb, gt_rgb)
            self.per_clip_render[key]["lpips"].append(self.lpips_window.compute().item())

    @torch.no_grad()
    def eval(self):
        t0 = time.time()
        n_windows = 0
        for batch in self.dataloader:
            images, gt_mask, clip, stream = batch
            # batch_size=1 → strip the batch dim from the per-window tensors,
            # but keep [1, S, ...] for the reconstructor.
            images = images.to(self.device, non_blocking=True)        # [1, S, 3, H, W]
            gt_mask = gt_mask.to(self.device, non_blocking=True)[0]   # [S, 4, H, W]
            clip = clip[0] if isinstance(clip, (list, tuple)) else clip
            stream = stream[0] if isinstance(stream, (list, tuple)) else stream
            self._step(images, gt_mask, clip, stream)
            n_windows += 1
            if n_windows % 20 == 0:
                print(
                    f"  [{n_windows} windows, {time.time()-t0:.0f}s elapsed]",
                    flush=True,
                )
        print(f"Eval done: {n_windows} windows in {time.time()-t0:.0f}s", flush=True)
        return self.compute()

    def compute(self) -> dict:
        out: dict = {"per_clip": {}, "aggregate": {}, "baselines": {}}

        if self.run_seg:
            global_seg = confusion_to_metrics(self.cm.cpu())
            out["aggregate"].update({
                "mIoU": global_seg["mIoU"],
                "IoU_per_class": dict(zip(CLASS_NAMES, global_seg["IoU_per_class"])),
                "pixel_accuracy": global_seg["pixel_accuracy"],
                "pixel_accuracy_per_class": dict(
                    zip(CLASS_NAMES, global_seg["pixel_accuracy_per_class"])
                ),
            })

            # Majority-class baseline: predict the most frequent class for every pixel.
            freqs = self.gt_pixel_counts.float().cpu()
            total = freqs.sum().clamp(min=1)
            majority = int(freqs.argmax().item())
            maj_iou = (freqs[majority] / total).item()  # IoU = freq, others = 0
            out["baselines"]["majority_class"] = CLASS_NAMES[majority]
            out["baselines"]["majority_class_mIoU"] = maj_iou / self.num_classes
            out["baselines"]["majority_class_pixel_accuracy"] = (
                freqs[majority] / total
            ).item()

            # Per-(clip,stream) seg rows.
            for key, cm in self.per_clip_cm.items():
                row = confusion_to_metrics(cm.cpu())
                bf1s = np.array(self.per_clip_bf1[key])  # [n_windows, C]
                row["bf1"] = float(bf1s.mean())
                row["bf1_per_class"] = bf1s.mean(axis=0).tolist()
                out["per_clip"].setdefault(f"{key[0]}::{key[1]}", {}).update(row)

            # Mean BF1 across clips.
            all_bf1 = [r["bf1"] for r in out["per_clip"].values() if "bf1" in r]
            if all_bf1:
                out["aggregate"]["boundary_f1"] = float(np.mean(all_bf1))

            out["aggregate"]["confusion_matrix"] = self.cm.cpu().tolist()

        if self.run_render:
            out["aggregate"]["PSNR"] = self.psnr.compute().item()
            out["aggregate"]["SSIM"] = self.ssim.compute().item()
            out["aggregate"]["LPIPS"] = self.lpips_m.compute().item()

            # Per-clip rendering: mean across the clip's windows.
            for key, d in self.per_clip_render.items():
                row = out["per_clip"].setdefault(f"{key[0]}::{key[1]}", {})
                row["PSNR"] = float(np.mean(d["psnr"]))
                row["SSIM"] = float(np.mean(d["ssim"]))
                row["LPIPS"] = float(np.mean(d["lpips"]))

            # Macro-mean across clips (complements the torchmetrics global).
            clip_psnr = [
                r["PSNR"] for r in out["per_clip"].values() if "PSNR" in r
            ]
            clip_ssim = [
                r["SSIM"] for r in out["per_clip"].values() if "SSIM" in r
            ]
            clip_lpips = [
                r["LPIPS"] for r in out["per_clip"].values() if "LPIPS" in r
            ]
            if clip_psnr:
                out["aggregate"]["PSNR_macro"] = float(np.mean(clip_psnr))
                out["aggregate"]["PSNR_macro_std"] = float(np.std(clip_psnr))
            if clip_ssim:
                out["aggregate"]["SSIM_macro"] = float(np.mean(clip_ssim))
                out["aggregate"]["SSIM_macro_std"] = float(np.std(clip_ssim))
            if clip_lpips:
                out["aggregate"]["LPIPS_macro"] = float(np.mean(clip_lpips))
                out["aggregate"]["LPIPS_macro_std"] = float(np.std(clip_lpips))

        return out


# ===================================================================== #
#                                 CLI                                   #
# ===================================================================== #


def parse_args():
    p = argparse.ArgumentParser(description="NeoVerse reconstructor benchmark.")

    p.add_argument("--data-root", type=str, default="diffsynth/data/training_data",
                   help="Must match training_25_04.py to keep the val split consistent.")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--frame-stride", type=int, default=3)
    p.add_argument("--window-size", type=int, default=6,
                   help="Frames per reconstructor call (S). Matches the demo's default.")
    p.add_argument("--img-shape", type=int, nargs=2, default=(280, 280),
                   help="(H, W). Resolution for the rasterizer is taken from this.")
    p.add_argument("--num-classes", type=int, default=4)

    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--reconstruction-model-path", type=str,
                   default="models/NeoVerse/reconstructor.ckpt")
    p.add_argument("--hand-head-path", type=str,
                   default="models/NeoVerse/hand_seg_model_opt_run20260507-224244_epoch005.ckpt")

    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true", default=True)

    p.add_argument("--run-seg", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-render", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--output-dir", type=str, default="outputs/benchmark")
    p.add_argument("--run-id", type=str,
                   default=datetime.now().strftime("%Y%m%d-%H%M%S"))

    return p.parse_args()


def load_reconstructor(args):
    print(f"Loading reconstructor from {args.reconstruction_model_path} ...", flush=True)
    mm = ModelManager()
    mm.load_model(
        args.reconstruction_model_path,
        device=args.device,
        torch_dtype=torch.bfloat16,
    )
    reconstructor: WorldMirror = mm.fetch_model("reconstructor")

    print(f"Loading hand head from {args.hand_head_path} ...", flush=True)
    ckpt = torch.load(args.hand_head_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    if not any(k.startswith("hand_pred_head.") for k in sd.keys()):
        sd = {f"hand_pred_head.{k}": v for k, v in sd.items()}
    else:
        sd = {k: v for k, v in sd.items() if k.startswith("hand_pred_head.")}
    missing, unexpected = reconstructor.load_state_dict(sd, strict=False)
    head_keys = [k for k in missing if "hand_pred_head" in k]
    if head_keys:
        print(f"WARNING: {len(head_keys)} hand_pred_head keys missing from ckpt", flush=True)

    # fp32 head — matches training_25_04.py
    reconstructor.hand_pred_head.float()
    reconstructor.to(args.device).eval()
    return reconstructor


def write_outputs(results: dict, args, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(results["aggregate"], f, indent=2)

    with open(out_dir / "baselines.json", "w") as f:
        json.dump(results["baselines"], f, indent=2)

    per_clip = results["per_clip"]
    if per_clip:
        # Stable column order: derive from the first row's keys, then sort scalars first.
        all_keys = set()
        for row in per_clip.values():
            all_keys.update(row.keys())
        scalar_keys = sorted(
            k for k in all_keys
            if not isinstance(next(iter(per_clip.values())).get(k, None), list)
        )
        list_keys = sorted(k for k in all_keys if k not in scalar_keys)
        columns = ["clip_stream", *scalar_keys, *list_keys]

        with open(out_dir / "per_clip.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(columns)
            for name, row in sorted(per_clip.items()):
                vals = [name]
                for k in scalar_keys:
                    v = row.get(k, "")
                    vals.append(f"{v:.6f}" if isinstance(v, float) else v)
                for k in list_keys:
                    vals.append(json.dumps(row.get(k, [])))
                w.writerow(vals)


def main():
    args = parse_args()

    val_clips = get_val_clips(args.data_root, args.val_fraction)
    print(f"Val clips: {len(val_clips)} (data_root={args.data_root})", flush=True)

    dataset = ClipWindowDataset(
        data_root=args.data_root,
        window_size=args.window_size,
        frame_stride=args.frame_stride,
        clip_names=val_clips,
    )
    print(f"Windows: {len(dataset)} (window_size={args.window_size}, "
          f"frame_stride={args.frame_stride})", flush=True)
    if len(dataset) == 0:
        raise RuntimeError("No windows built — check data_root and val_fraction.")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
    )

    reconstructor = load_reconstructor(args)

    H, W = args.img_shape
    evaluator = BenchmarkEvaluator(
        reconstructor,
        loader,
        device=args.device,
        num_classes=args.num_classes,
        run_seg=args.run_seg,
        run_render=args.run_render,
        resolution=(W, H),
    )
    results = evaluator.eval()

    out_dir = Path(args.output_dir) / args.run_id
    write_outputs(results, args, out_dir)

    print("\n=== Aggregate ===")
    print(json.dumps(results["aggregate"], indent=2))
    print("\n=== Baselines ===")
    print(json.dumps(results["baselines"], indent=2))
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
