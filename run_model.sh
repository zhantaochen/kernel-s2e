#!/bin/bash
#SBATCH --account lcls
#SBATCH --constraint gpu
##SBATCH --qos regular
##SBATCH --time 12:00:00
#SBATCH --qos debug
#SBATCH --time 00:05:00
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 1
#SBATCH --output=/pscratch/sd/e/edmundxu/kernel-s2e/slurm_logs/%x.%j.out

export SLURM_CPU_BIND="cores"

module load python
conda activate backsub

srun python train_model.py

# perform any cleanup or short post-processing here