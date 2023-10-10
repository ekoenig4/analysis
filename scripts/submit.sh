#!/bin/bash

#SBATCH --job-name=studies
#SBATCH --qos=avery
#SBATCH --account=avery
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB


source /home/$USER/.bashrc
ml conda parallel
mamba activate studies

export CUDA_LAUNCH_BLOCKING=1

echo $@ 
$@ 