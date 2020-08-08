#!/bin/bash -l

#SBATCH
#SBATCH -A lgraham1
#SBATCH --job-name=Generate_Data
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=shared
#SBATCH --mail-type=end
#SBATCH --mail-user=audrey.olivier@jhu.edu

module load python/3.6-anaconda

python CreateInputData_v3.py
