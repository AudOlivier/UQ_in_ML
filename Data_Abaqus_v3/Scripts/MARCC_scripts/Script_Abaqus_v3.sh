#!/bin/bash -l

#SBATCH
#SBATCH -A mshiel10
#SBATCH --job-name=AbaqusModel
#SBATCH --time=6:00:00
#SBATCH --ntasks=5
####SBATCH --ntasks-per-node=2
####SBATCH --partition=debug
#SBATCH --partition=shared
#SBATCH --mail-type=end
#SBATCH --mail-user=audrey.olivier@jhu.edu
####SBATCH --array=0-10%5

cd /home-1/aolivie1@jhu.edu/scratch/AbaqusModel_v3/

module load abaqus
module load python/3.6-anaconda

abaqus cae noGUI=RunModels_v3_forUQ.py -- $SLURM_ARRAY_TASK_ID _v3_inputs[4]_ndata[20]_nsame[20]_lhs
#### abaqus cae noGUI=RunModels_v3_forUQ.py -- 0 _v3_inputs[4]_ndata[10]_nsame[10]_beta
