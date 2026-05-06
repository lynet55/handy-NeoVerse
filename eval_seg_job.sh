#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=neoverse_eval_seg
#SBATCH --partition=jobs
#SBATCH --time=0-01:00:00
#SBATCH --chdir=/work/courses/3dv/team32/handy-NeoVerse
#SBATCH --output=/work/courses/3dv/team32/handy-NeoVerse/logs/eval_seg_%j.out
#SBATCH --error=/work/courses/3dv/team32/handy-NeoVerse/logs/eval_seg_%j.err

mkdir -p logs outputs

. /etc/profile.d/modules.sh
module add cuda/12.8

source ./neoverse/bin/activate

python -u eval_segmentation.py \
    --npz \
        diffsynth/data/training_data/clip-001053.npz \
        diffsynth/data/training_data/clip-001068.npz \
        diffsynth/data/training_data/clip-001083.npz \
        diffsynth/data/training_data/clip-001100.npz \
        diffsynth/data/training_data/clip-001120.npz \
    --hand_head_path models/NeoVerse/hand_seg_model_opt_run20260426-130617_epoch004.ckpt \
    --output outputs/eval_seg.mp4
