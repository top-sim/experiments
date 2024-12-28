#!/bin/bash
#SBATCH --job-name=multi_experiments         # Job name
#SBATCH --array=0-17                          # Total number of tasks (experiments)
#SBATCH --nodes=3                            # Number of nodes to use
#SBATCH --cpus-per-task=1                    # 1 core per task (since the app is single-threaded)
#SBATCH --time=02:00:00                      # Maximum runtime
#SBATCH --partition=main                     # Partition/queue to use

source /home/rbunney/github/topsim/venv/bin/activate

input_dir="/scratch/rbunney/week_test/low"

input_files=($(ls $input_dir/*.json))

input_file="${input_files[$SLURM_ARRAY_TASK_ID]}"

srun -n 1 -N 1 python slurm_tester.py "$input_dir/$input_file" "$SLURM_ARRAY_TASK_ID"