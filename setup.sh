conda create -n neoverse python=3.10 -y
conda activate neoverse
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git
pip install git+https://github.com/facebookresearch/hand_tracking_toolkit
pip install git+https://github.com/mattloper/chumpy --no-build-isolation
pip install git+https://github.com/vchoutas/smplx.git
hf download Yuppie1204/NeoVerse --include reconstructor.ckpt --local-dir models/NeoVerse
