#!/bin/bash
#SBATCH -N 1
#SBATCH -p spider
#SBATCH -C V100
#SBATCH --time=06:00:00


source ~/.bashrc

conda activate ncar
module load gcc openmpi/gcc cudatoolkit

#LR=0.01
export LR=0.0005 # CNN

srun --cpus-per-task 6 --ntasks-per-node 8 -N 1 -u python pytorch_eke.py --lr $LR \
     --log-interval 1000 --model 'resnet' --batch-size 512 --epochs 100 \
     --weighted-sampling
