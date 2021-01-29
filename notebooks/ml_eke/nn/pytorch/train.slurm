#!/bin/bash
#SBATCH -N 1
#SBATCH -p stormP100
#SBATCH -C P100
#SBATCH --time=06:00:00


source ~/.bashrc

conda activate old_torch
module load gcc openmpi/gcc cudatoolkit

#LR=0.01
export LR=0.0005 # CNN

srun --cpus-per-task 6 --ntasks-per-node 8 -N 1 -u python pytorch_eke.py --lr $LR \
     --log-interval 1000 --model 'resnet_small' --batch-size 512 --epochs 100 \
     --weighted-sampling
