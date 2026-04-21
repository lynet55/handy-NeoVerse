"""
Reconstruct a single image using only the reconstructor.

Usage:
    python image_gen.py --image_name my_image.png
"""

import argparse
import gc
import os
import torch
from PIL import Image
from torchvision.transforms import functional as F

from diffsynth.models import ModelManager
from diffsynth.utils.auxiliary import load_video, homo_matrix_inverse

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
INPUT_DIR = os.path.join(RESULTS_DIR, "input")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "output")
RECONSTRUCTOR_PATH = "models/NeoVerse/reconstructor.ckpt"
RES_W, RES_H = 560, 336
LOW_VRAM = False

parser = argparse.ArgumentParser()
parser.add_argument("--image_name", required=True,
                    help="Name of the input image inside results/input")
args = parser.parse_args()

input_path = os.path.join(INPUT_DIR, args.image_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, args.image_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
load_device = "cpu" if LOW_VRAM else device

print(f"Loading reconstructor from {RECONSTRUCTOR_PATH} (device: {load_device})...")
model_manager = ModelManager()
model_manager.load_model(RECONSTRUCTOR_PATH, device=load_device, torch_dtype=torch.bfloat16)
reconstructor = model_manager.fetch_model("reconstructor")
print("Reconstructor loaded.")


@torch.no_grad()
def reconstruct_image(image_path, out_path):
    pil_images = load_video(image_path, num_frames=1, resolution=(RES_W, RES_H),
                            resize_mode="center_crop", static_scene=True)
    S = len(pil_images)

    views = {
        "img": torch.stack([F.to_tensor(img)[None] for img in pil_images], dim=1).to(device),
        "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
        "is_static": torch.ones((1, S), dtype=torch.bool, device=device),
        "timestamp": torch.zeros((1, S), dtype=torch.int64, device=device),
    }

    if LOW_VRAM:
        reconstructor.to(device)

    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = reconstructor(views, is_inference=True, use_motion=False)
    finally:
        if LOW_VRAM:
            reconstructor.to("cpu")
            torch.cuda.empty_cache()

    gaussians = predictions["splats"]
    input_c2w = predictions["rendered_extrinsics"][0]
    input_intrs = predictions["rendered_intrinsics"][0]
    input_timestamps = predictions["rendered_timestamps"][0]

    input_w2c = homo_matrix_inverse(input_c2w)

    target_rgb, _, _ = reconstructor.gs_renderer.rasterizer.forward(
        gaussians,
        render_viewmats=[input_w2c],
        render_Ks=[input_intrs],
        render_timestamps=[input_timestamps],
        sh_degree=0, width=RES_W, height=RES_H,
    )

    rendered = (target_rgb[0, 0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(rendered).save(out_path)

    del predictions, views
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Saved reconstructed image to {out_path}")


if __name__ == "__main__":
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    reconstruct_image(input_path, output_path)
