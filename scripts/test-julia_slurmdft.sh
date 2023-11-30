#!/bin/bash
#SBATCH --job-name=CNF_test
#SBATCH --output=output.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

module load Julia/1.8/5

srun julia DTwithCNF.jl