#!/bin/bash
#SBATCH --job-name=multi_experiments         # Job name
#SBATCH --output=output_%A_%a.txt	     # Output file per task
#SBATCH --ntasks=1 			     # 
#SBATCH --cpus-per-task=2                    # 1 core per task (since the app is single-threaded)
#SBATCH --array=0-72	                     # 21 files 
#SBATCH --time=48:00:00                      # Maximum runtime
#SBATCH --partition=main                     # Partition/queue to use
#SBATCH --mem-per-cpu=8G


echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

source /home/rbunney/github/topsim/venv/bin/activate

input_dir="/scratch/rbunney/pulsarPlans/low"

input_files=($(ls $input_dir/*.json))

input_file=${input_files[$SLURM_ARRAY_TASK_ID]}

#srun python ../github/experiments/slurm/slurm_tester.py "$input_file" "$SLURM_ARRAY_TASK_ID"
srun python ../github/experiments/simulator_demonstration_experiments/topsim_test_simulations/playground_scalability_new.py -f "$input_file" static
