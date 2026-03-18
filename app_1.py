import os
import numpy as np
import gradio as gr
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_ROOT = "outputs/gradio"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
GLB_PATH = os.path.join(OUTPUT_ROOT, "scene.glb")


# ---------------------------------------------------------------------------
# Gaussian splatting data loader (stub — replace with real loader)
# ---------------------------------------------------------------------------
def load_gaussian_splat(file_path):
    """Load a .ply/.splat file and return (points, colors) as numpy arrays."""
    # TODO: replace with real Gaussian splatting loader
    # e.g. parse PLY xyz + sh coefficients, or use gsplat / nerfstudio reader
    points = np.random.randn(1000, 3).astype(np.float32)
    colors = np.clip(np.random.rand(1000, 3).astype(np.float32), 0, 1)
    return points, colors


def points_to_glb(points, colors, out_path):
    """Export point cloud to GLB for the Model3D viewer."""
    import trimesh
    pc = trimesh.PointCloud(vertices=points, colors=(colors * 255).astype(np.uint8))
    scene = trimesh.Scene(pc)
    scene.export(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def handle_upload(file):
    if file is None:
        return gr.update(), gr.update()
    points, colors = load_gaussian_splat(file)
    glb = points_to_glb(points, colors, GLB_PATH)
    info = f"Loaded {len(points)} points from: {os.path.basename(file)}"
    return glb, info


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Gaussian Splatting Viewer") as demo:
    gr.Markdown("## Gaussian Splatting Viewer")
    gr.Markdown("Upload a `.ply` or `.splat` file to visualise the point cloud.")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload Gaussian Splat (.ply / .splat)",
                file_types=[".ply", ".splat"],
                file_count="single",
            )
            info_box = gr.Textbox(label="Info", interactive=False)

        with gr.Column(scale=2):
            model3d = gr.Model3D(label="Point Cloud", height=500,
                                 zoom_speed=0.5, pan_speed=0.5)

    file_upload.upload(
        fn=handle_upload,
        inputs=[file_upload],
        outputs=[model3d, info_box],
    )


if __name__ == "__main__":
    demo.queue(max_size=5).launch(show_error=True, share=True)
