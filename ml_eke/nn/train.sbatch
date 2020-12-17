#!/bin/bash
#SBATCH -p stormP100
#SBATCH --ntasks-per-node 8
#SVATCH -u
#SBATCH --time=12:00:00
#SBATCH -J HVD_TRAIN
#SBATCH -o full_train.%A

source ~/.bashrc
export TF_CPP_MIN_LOG_LEVEL=3

module load gcc/9.1.0
module load openmpi/gcc/4.0.1
conda activate ncar
cd /lus/scratch/arigazzi/ncar/machine_learning_eke
export PYTHONPATH=$PYTHONPATH:$PWD
cd examples

srun python train_hvd.py
