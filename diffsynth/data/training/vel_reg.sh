#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=gs_mask_train
#SBATCH --partition=jobs
#SBATCH --time=2-06:00:00
#SBATCH --chdir=/work/courses/3dv/team32/handy-NeoVerse
#SBATCH --output=/work/courses/3dv/team32/handy-NeoVerse/logs/vel_reg_train_%j.out
#SBATCH --error=/work/courses/3dv/team32/handy-NeoVerse/logs/vel_reg_train_%j.err

set -eo pipefail

mkdir -p logs models/NeoVerse runs

. /etc/profile.d/modules.sh
module add cuda/12.8

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID:-?}"
echo "Job name:      ${SLURM_JOB_NAME:-?}"
echo "Node:          ${SLURMD_NODENAME:-?}"
echo "Partition:     ${SLURM_JOB_PARTITION:-?}"
echo "GPUs:          ${SLURM_GPUS_ON_NODE:-?} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?})"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-default}"
echo "Memory:        ${SLURM_MEM_PER_NODE:-default} MB"
echo "Working dir:   $(pwd)"
echo "Start time:    $(date)"
echo "------------------------------------------"
nvidia-smi || true
echo "=========================================="

source ./neoverse/bin/activate
echo "Python:        $(which python) ($(python --version 2>&1))"
echo "------------------------------------------"

START=$(date +%s)
python -u -m diffsynth.data.training.training_gs_mask
EXIT_CODE=$?
END=$(date +%s)

echo "------------------------------------------"
echo "Exit code:     $EXIT_CODE"
echo "Duration:      $((END - START)) seconds"
echo "End time:      $(date)"
echo "=========================================="

exit $EXIT_CODE
