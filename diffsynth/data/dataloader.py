#import torch
import webdataset as wds

urls = [
    f"https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/main/train_quest3/clip-{i:06d}.tar"
    for i in range(1)
]

dataset = wds.WebDataset(urls, shardshuffle=False).shuffle(10)

q = 1
for i, sample in enumerate(dataset):
    print(sample["hand.json"])
    print(sample["cameras.json"])
    if q == 1:
        break
"""
dataset = (
        wds.WebDataset(url)
    .shuffle(1000)
    .decode("rgb")
    .to_tuple("jpg","json")
    .batched(16)
)

loader = DataLoader(
    dataset,
    batch_size=None,
    num_workers=2,
)
"""

