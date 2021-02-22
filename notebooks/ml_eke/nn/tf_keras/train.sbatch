#!/bin/bash
#SBATCH -p stormP100
#SBATCH --ntasks-per-node 8
#SBATCH --time=12:00:00
#SBATCH -J HVD_TRAIN
#SBATCH -o full_train.%A

source ~/.bashrc
export TF_CPP_MIN_LOG_LEVEL=3

module load gcc/9.1.0
module load openmpi/gcc/4.0.1
conda activate ncar
cd /lus/snx11049/arigazzi/ncar/NCAR_ML_EKE/notebooks/ml_eke/nn
export PYTHONPATH=$PYTHONPATH:$PWD

srun -u python train_hvd.py
