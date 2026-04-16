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