#!/bin/bash

#SBATCH --job-name=maximal_generation        # Job name
#SBATCH --output=output_%A_%a.txt	         # Output file per task
#SBATCH --ntasks=1 			                 # 
#SBATCH --cpus-per-task=8                    # 
#SBATCH --time=48:00:00                      # Maximum runtime
#SBATCH --partition=main                     # Partition/queue to use
#SBATCH --mem-per-cpu=16G

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

source /home/rbunney/github/topsim/venv/bin/activate

srun python ../github/experiments/workflow_scheduling_experiments/basic_experiment/create_maximal_workflows.py /scratch/rbunney/maximal_contimg/ all cont_img_mvp maximal
