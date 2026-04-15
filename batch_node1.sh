#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --account=3dv
#SBATCH --partition=interactive-cpu
#SBATCH --output=logs/node1_%j.out

source /work/courses/3dv/team32/handy-NeoVerse/neoverse/bin/activate
cd /work/courses/3dv/team32/handy-NeoVerse
python diffsynth/data/optimized_download.py --clip-start 801 \
  --tar-dir diffsynth/data/tar_recv_1 --num-workers 2 --max-stored-clips 4
