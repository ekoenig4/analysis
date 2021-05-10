#!/bin/sh
#SBATCH --job-name=sixb_features    # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ekoenig4@wisc.edu  # Where to send mail	
#SBATCH --nodes=1                     # Use one node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=1             # Use 1 core
#SBATCH --mem=2000mb                   # Memory limit
#SBATCH --time=02:00:00               # Time limit hrs:min:sec
#SBATCH --output=slurm/logs/sixb_features_%j.out   # Standard output and error log

pwd; hostname; date

ml conda
conda activate /home/ekoenig/anaconda3/envs/sixb

python scripts/2_generate_features.py

date
