#!/bin/bash

#SBATCH --job-name=workflow_scheduling       # Job name
#SBATCH --output=output_%A_%a.txt	     # Output file per task
#SBATCH --nodes=1                            #
#SBATCH --ntasks=1 			     # 
#SBATCH --cpus-per-task=24                   # 
#SBATCH --time=72:00:00                      # Maximum runtime
#SBATCH --partition=main                     # Partition/queue to use
#SBATCH --mem-per-cpu=4GB

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

source /home/rbunney/github/topsim/venv/bin/activate

srun python ../github/experiments/workflow_scheduling_experiments/basic_experiment/run_comparisons_metadata.py /scratch/rbunney/maximal_contimg/all/
