#!/bin/bash

#SBATCH --job-name=studies
#SBATCH --qos=avery
#SBATCH --account=avery
#SBATCH --time=4:00:00
#SBATCH --partition=hpg-dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB


source /home/$USER/.bashrc
ml conda parallel
mamba activate studies

export CUDA_LAUNCH_BLOCKING=1

echo $@ 
$@ 