#!/bin/sh
#SBATCH --job-name=sixb_training   # Job name
#SBATCH --mail-type=ALL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ekoenig4@wisc.edu # Where to send mail	
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --mem-per-cpu=1gb           # Memory per processor
#SBATCH --time=00:05:00             # Time limit hrs:min:sec
#SBATCH --output=slurm/logs/sixb_training_%A-%a.out    # Standard output and error log
#SBATCH --array=1-5                 # Array range

pwd; hostname; date

ml conda
conda activate /home/ekoenig/anaconda3/envs/sixb

python scripts/3_train_neural_network.py --run ${SLURM_ARRAY_TASK_ID} $@

date
