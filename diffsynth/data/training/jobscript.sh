#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=neoverse_train
#SBATCH --partition=jobs
#SBATCH --time=7-00:00:00      # 7 days (max allowed)
#SBATCH --chdir=/work/courses/3dv/team32/handy-NeoVerse
#SBATCH --output=/work/courses/3dv/team32/handy-NeoVerse/diffsynth/data/training/logs/classificaitonDPT/%x_%j.out
#SBATCH --error=/work/courses/3dv/team32/handy-NeoVerse//diffsynth/data/training/logs/classificaitonDPT/%x_%j.err

# Make sure the log dir exists (sbatch won't create it for you)
mkdir -p /work/courses/3dv/team32/handy-NeoVerse/diffsynth/data/training/logs/classificaitonDPT


# ---------- Modules ----------
. /etc/profile.d/modules.sh
module add cuda/12.8

# ---------- Reporting ----------
echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Node:          $SLURMD_NODENAME"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "GPUs:          ${SLURM_GPUS_ON_NODE:-?} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?})"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-default}"
echo "Memory:        ${SLURM_MEM_PER_NODE:-default} MB"
echo "Working dir:   $(pwd)"
echo "Start time:    $(date)"
echo "------------------------------------------"
nvcc --version
nvidia-smi || true
echo "=========================================="

# Fail fast on errors; show piped command failures
set -eo pipefail

# ---------- Environment ----------
source ./neoverse/bin/activate
echo "Python:        $(which python) ($(python --version 2>&1))"
echo "------------------------------------------"

# ---------- Run ----------
START=$(date +%s)

python -u -m diffsynth.data.training.training_with_debug
EXIT_CODE=$?

END=$(date +%s)
echo "------------------------------------------"
echo "Exit code:     $EXIT_CODE"
echo "Duration:      $((END - START)) seconds"
echo "End time:      $(date)"
echo "=========================================="

exit $EXIT_CODE