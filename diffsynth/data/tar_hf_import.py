import os
import requests

def tar_hf_import(num: int):
    num_str = str(num).rjust(6, "0")
    url = f"https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/main/train_quest3/clip-{num_str}.tar?download=true"
    output_path = f"diffsynth/data/tar_recv/clip-{num_str}.tar" 
    if os.path.exists(output_path):
        return False
    try: 
        with requests.get(url, stream=True) as r: 
            r.raise_for_status()
            with open(output_path, "wb") as f: 
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
    except Exception as e: 
        print(e)
        return False
    return True

if __name__ == "__main__": 
    tar_hf_import(1)
