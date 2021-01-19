#!/bin/bash
#SBATCH -N 8
#SBATCH -C P100
#SBATCH --time=06:00:00


source ~/.bashrc

conda activate ncar_torch

# LR=0.01
export LR=0.001 # CNN

srun --cpus-per-task 18 --ntasks-per-node 1 -N 8 -u python pytorch_eke.py --lr $LR --log-interval 1000 --batch-size 512 --epochs 100  --weighted-sampling
