"""
Reconstruction-only Gradio app.
Loads only the reconstructor (~1-2 GB VRAM) — no diffusion model required.

Launch:
    python app_reconstruct.py
    python app_reconstruct.py --reconstructor_path models/da3_giant_1.1.safetensors
    python app_reconstruct.py --low_vram
    python app_reconstruct.py --low_vram --max_frames 49 --resolution 448x256
"""

import argparse
import gc
import os
import torch
import gradio as gr
from PIL import Image
from torchvision.transforms import functional as F

from diffsynth.models import ModelManager
from diffsynth.utils.auxiliary import load_video, homo_matrix_inverse
from diffsynth.utils.app import extract_point_cloud, build_scene_glb
from diffsynth import save_video, save_frames

parser = argparse.ArgumentParser()
parser.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt")
parser.add_argument("--low_vram", action="store_true",
                    help="Keep model on CPU; move to GPU only during inference")
parser.add_argument("--max_frames", type=int, default=81,
                    help="Maximum number of frames to load (default: 81)")
parser.add_argument("--resolution", type=str, default="560x336",
                    help="WxH resolution for input frames (default: 560x336)")
args, _ = parser.parse_known_args()

res_w, res_h = (int(x) for x in args.resolution.split("x"))

OUTPUT_DIR = "outputs/reconstruct"
GLB_PATH = os.path.join(OUTPUT_DIR, "scene.glb")
SPLAT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "splat_render.mp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
load_device = "cpu" if args.low_vram else device

print(f"Loading reconstructor from {args.reconstructor_path} (device: {load_device})...")
model_manager = ModelManager()
model_manager.load_model(args.reconstructor_path, device=load_device, torch_dtype=torch.bfloat16)
reconstructor = model_manager.fetch_model("reconstructor")
print("Reconstructor loaded.")


@torch.no_grad()
def reconstruct(video_path, scene_type):
    if video_path is None:
        raise gr.Error("Please upload a video first.")
    static = scene_type == "Static scene"
    pil_images = load_video(video_path, args.max_frames, resolution=(res_w, res_h),
                            resize_mode="center_crop", static_scene=static)
    pil_images = pil_images[:6]
    S = len(pil_images)

    views = {
        "img": torch.stack([F.to_tensor(img)[None] for img in pil_images], dim=1).to(device),
        "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
        "is_static": torch.ones((1, S), dtype=torch.bool, device=device) if static
                     else torch.zeros((1, S), dtype=torch.bool, device=device),
        "timestamp": torch.zeros((1, S), dtype=torch.int64, device=device) if static
                     else torch.arange(S, dtype=torch.int64, device=device).unsqueeze(0),
    }

    if args.low_vram:
        reconstructor.to(device)

    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = reconstructor(views, is_inference=True, use_motion=False)
    finally:
        if args.low_vram:
            reconstructor.to("cpu")
            torch.cuda.empty_cache()

    gaussians = predictions["splats"]
    input_c2w = predictions["rendered_extrinsics"][0]   # [S, 4, 4]
    input_intrs = predictions["rendered_intrinsics"][0] # [S, 3, 3]
    input_timestamps = predictions["rendered_timestamps"][0]  # [S]
    classifications = predictions["seg_labels"][0]
    print(f"classifications: {type(classifications)}, {classifications.shape}")
    # --- Point cloud GLB ---
    points, colors, frame_indices = extract_point_cloud(predictions)
    scene = build_scene_glb(points, colors, frame_indices, input_c2w.cpu().numpy())
    scene.export(file_obj=GLB_PATH)

    # --- Gaussian splat render (re-render from input viewpoints) ---
    input_w2c = homo_matrix_inverse(input_c2w)  # [S, 4, 4]

    target_rgb, _, _ = reconstructor.gs_renderer.rasterizer.forward(
        gaussians,
        render_viewmats=[input_w2c],
        render_Ks=[input_intrs],
        render_timestamps=[input_timestamps],
        sh_degree=0, width=res_w, height=res_h,
        render_classes = [0, 1, 0, 1]
    )
    # target_rgb: [1, S, H, W, 3] float in [0,1]
    frames = [
        Image.fromarray((target_rgb[0, i].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())
        for i in range(target_rgb.shape[1])
    ]
    save_video(frames, SPLAT_VIDEO_PATH, fps=16)

    n_gaussians = sum(gs.means.shape[0] for gs in predictions["splats"][0])
    info = (f"Frames: {S}  |  Gaussians: {n_gaussians:,}  |  "
            f"Points shown: {points.shape[0]:,}")

    del predictions, views
    gc.collect()
    torch.cuda.empty_cache()

    return GLB_PATH, SPLAT_VIDEO_PATH, info


with gr.Blocks(title="NeoVerse — Reconstruction Viewer") as demo:
    gr.Markdown("# NeoVerse — Reconstruction Viewer\n"
                "Upload a video to reconstruct the 4D Gaussian scene. "
                "No diffusion model needed — runs on ~2 GB VRAM.")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Input Video", sources=["upload"],
                                   value="examples/videos/driving.mp4")
            scene_type = gr.Radio(["General scene", "Static scene"],
                                  value="General scene", label="Scene Type")
            reconstruct_btn = gr.Button("Reconstruct", variant="primary")
            info_box = gr.Textbox(label="Scene Info", interactive=False)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Gaussian Splat Render"):
                    splat_video = gr.Video(label="Splat Re-render (input viewpoints)", height=400)
                with gr.Tab("3D Point Cloud"):
                    model3d = gr.Model3D(label="3D Scene", height=400,
                                        zoom_speed=0.5, pan_speed=0.5)

    reconstruct_btn.click(
        fn=reconstruct,
        inputs=[video_input, scene_type],
        outputs=[model3d, splat_video, info_box],
    )

if __name__ == "__main__":
    # demo.queue(max_size=3).launch(show_error=True, server_name="0.0.0.0", server_port=7860)
    glb_path, splat_video_path, info = reconstruct(
        "examples/videos/driving.mp4", "General scene"
    )
    print(info)
    print(f"GLB saved to: {glb_path}")
    print(f"Splat render saved to: {splat_video_path}")