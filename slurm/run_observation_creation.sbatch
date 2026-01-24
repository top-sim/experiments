#!/bin/bash

#SBATCH --job-name=obs_generation        # Job name
#SBATCH --output=output_%A_%a.txt	         # Output file per task
#SBATCH --ntasks=1 			                 # 
#SBATCH --cpus-per-task=8                    # 
#SBATCH --time=48:00:00                      # Maximum runtime
#SBATCH --partition=main                     # Partition/queue to use
#SBATCH --mem-per-cpu=16G

source /home/rbunney/github/venv/bin/activate

srun python ../github/experiments/simulation_scheduling_experiments/create_observation_plans.py /scratch/rbunney/baseline_variation_prototype/ low prototype 
