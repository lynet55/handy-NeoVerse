import os
import imageio
import argparse
import numpy as np
import tarfile
import clip_util

print("fffff")

def extract_clip(
    clip_path: str,
    output_dir: str,
    streams_to_extract=("1201-1", "1201-2"),
) -> None:
    """
    Extract images from a clip tar, separated by stream.

    Args:
        clip_path: Path to a clip saved as a tar file.
        output_dir: Root output directory.
        streams_to_extract: Streams to extract (e.g. ("1201-1", "1201-2")).
    """

    # Open the tar file (same as vis_clip)
    tar = tarfile.open(clip_path, mode="r")

    clip_name = os.path.basename(clip_path).split(".tar")[0]
    print(f"Extracting clip {clip_name}")

    for frame_id in range(clip_util.get_number_of_frames(tar)):
        frame_key = f"{frame_id:06d}"

        # Load camera parameters to know available streams
        cameras, _ = clip_util.load_cameras(tar, frame_key)

        for stream_id in cameras.keys():
            stream_key = str(stream_id)

            if stream_key not in streams_to_extract:
                continue

            # Prepare output directory PER STREAM
            stream_output_dir = os.path.join(
                output_dir, f"stream{stream_key}", clip_name
            )
            os.makedirs(stream_output_dir, exist_ok=True)

            # Load image (same call as vis_clip)
            image = clip_util.load_image(tar, frame_key, stream_key)

            # Ensure 3 channels
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)

            # Orient upright (same as vis_clip)
            image = np.rot90(image, k=3)

            # Save image
            out_path = os.path.join(stream_output_dir, f"{frame_key}.png")
            imageio.imwrite(out_path, image)

    tar.close()


def main() -> None:
    print("extractor started")
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--clip_start", type=int, default=0)
    parser.add_argument("--clip_end", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    clips = sorted([p for p in os.listdir(args.clips_dir) if p.endswith(".tar")])

    for clip in clips:
        clip_id = int(clip.split(".tar")[0].split("clip-")[1])

        if clip_id < args.clip_start or (
            args.clip_end >= 0 and clip_id > args.clip_end
        ):
            continue

        extract_clip(
            clip_path=os.path.join(args.clips_dir, clip),
            output_dir=args.output_dir,
        )

if __name__ == "__main__":
    main()

