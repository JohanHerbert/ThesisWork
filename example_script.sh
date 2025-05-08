#!/bin/bash
# Here you should put your own project id
#SBATCH -A hpc2n2024-146
# Allocate GPU and 14 CPU cores
#SBATCH --gres=gpu:v100:1
#SBATCH -c 14
# Ask for a suitable amount of time. Remember, this is the time the Jupyter notebook will be available!
#SBATCH --time=00:15:00
# Name of job
#SBATCH -J ml_pytorch
# Send mail when job begins
#SBATCH --mail-type=BEGIN
 
# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
 
# Load the module environment suitable for the job
ml restore Johans_collection



python BigTestKebnekaise.py 