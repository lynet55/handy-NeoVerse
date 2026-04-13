import os
import requests


def tar_hf_import(num: int, dest_dir: str = "diffsynth/data/tar_recv") -> bool:
    """Download clip `num` into `dest_dir`.

    Returns True  – clip is available (already on disk OR just downloaded).
    Returns False – clip does not exist on HuggingFace (HTTP 404), i.e. end of dataset.
    Raises        – on any other network / HTTP error so the caller can decide.
    """
    num_str = str(num).rjust(6, "0")
    url = (
        f"https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/main/"
        f"train_quest3/clip-{num_str}.tar?download=true"
    )
    output_path = os.path.join(dest_dir, f"clip-{num_str}.tar")

    if os.path.exists(output_path):
        return True  # already have it; more clips may exist beyond this index

    tmp_path = output_path + ".tmp"
    try:
        with requests.get(url, stream=True) as r:
            if r.status_code == 404:
                return False  # no such clip → end of dataset
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 17):  # 128 KiB chunks
                    f.write(chunk)
        os.rename(tmp_path, output_path)  # atomic-ish: avoid leaving partial tars
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return True


if __name__ == "__main__":
    tar_hf_import(1)
