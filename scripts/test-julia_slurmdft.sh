#!/bin/bash
#SBATCH --job-name=CNFDT_test
#SBATCH --output=outputdt.txt
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

module load Julia/1.8/5

srun julia DTwithCNF.jl