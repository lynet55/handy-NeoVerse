# ETH Student Cluster Reference

## VPN Connection

```bash
sudo openconnect -u 'bbalinov@student-net.ethz.ch' \
  --useragent=AnyConnect \
  -g student-net \
  sslvpn.ethz.ch \
  --no-external-auth
```

## Available GPUs

- `1080ti` — NVidia GeForce GTX 1080 Ti (11 GB VRAM)
- `2080ti` — NVidia GeForce GTX 2080 Ti (11 GB VRAM)
- `5060ti` — NVidia GeForce RTX 5060 Ti (16 GB VRAM)

## Submitting Jobs

```bash
sbatch jobscript.sh
```

## TensorBoard

Forward the remote TensorBoard port to your local machine:

```bash
ssh -N -L 6006:localhost:6006 bbalinov@student-cluster1.inf.ethz.ch
```

Then open <http://localhost:6006> in your local browser.

### What to expect in TensorBoard

From the current `train()` code, the following scalars are logged:

**Per step** (x-axis = `global_step`):
- `train/loss_step`

**Per epoch** (x-axis = `epoch`):
- `train/loss_epoch`
- `train/mIoU_epoch`
- `train/epoch_time_s`
- `train/lr`
- `train/IoU_right_hand`, `train/IoU_left_hand`, `train/IoU_object`, `train/IoU_background`

## Useful Cluster Commands

### Inspect partitions and GPUs

```bash
sinfo -o "%P %G %N"           # partitions and their GRES config
scontrol show partition jobs  # detailed partition info
```

### Manage your jobs

```bash
squeue -u $USER
scancel $SLURM_JOB_ID
```

### Inspect a running job's log

```bash
JOBID=<your job id>
LOG=/work/courses/3dv/team32/handy-NeoVerse/logs/neoverse_train/neoverse_train_${JOBID}.out

# 1. Dataset size
grep "Dataset built" $LOG

# 2. Current step and recent timings
grep "step " $LOG | tail -10

# 3. Any warnings or errors
grep -iE "warn|error|nan|inf" $LOG

# 4. First few steps vs recent — are steps slowing down?
grep "step " $LOG | head -5
grep "step " $LOG | tail -5
```

### Data + Training

---

## Optimized Training (`training_optimized.py`)

Drop-in alternative to `training_with_debug.py` for training the `hand_pred_head` segmentation head. Use this when you've confirmed the original pipeline works end-to-end and want faster iteration.

### What it changes

The original script calls `reconstructor(views)` every step, which:
1. Runs the frozen ViT backbone (~300M params, ~85% of step time)
2. Runs it a **second time** inside `prepare_contexts` (identical output, pure waste)
3. Runs every other frozen head (depth, pts, normals, GS, camera)
4. Runs the Gaussian-splat rasterizer (output never enters the loss)

`training_optimized.py` calls the backbone **once** under `torch.no_grad()` and feeds the result directly to `hand_pred_head`. Everything else is skipped.

Additional changes:
- **Frame subsampling** (`frame_stride=5`): consecutive video frames are near-identical; keeping every 5th cuts the dataset to ~20% with minimal diversity loss
- **Cosine LR schedule**: decays from `3e-4` to `3e-6`, letting the model refine after initial fast convergence
- Separate checkpoint namespace (`hand_seg_model_opt`) and TensorBoard log dir (`runs/neoverse_seg_opt`) so runs don't collide

### Usage

```bash
# Same as the original — run from repo root
python -m diffsynth.data.training.training_optimized
```

### Config knobs (`TrainConfig`)

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `frame_stride` | `5` | Keep every Nth frame (1 = all frames, matches original) |
| `lr_min_factor` | `0.01` | Cosine annealing decays LR to `learning_rate * lr_min_factor` |
| `batch_size` | `10` | Same as original |
| `epochs` | `3` | Same as original |

### Optional: pre-compute backbone features

For maximum speedup, extract and cache backbone outputs once, then train the head on cached tensors. This eliminates the backbone forward pass entirely (~5-8x faster per step).

```bash
python -m diffsynth.data.training.precompute_features \
    --data_root diffsynth/data/training_data \
    --output_dir diffsynth/data/training_features \
    --frame_stride 5
```

Each frame produces ~8 MB of cached tensors (bfloat16). Scale storage accordingly.

