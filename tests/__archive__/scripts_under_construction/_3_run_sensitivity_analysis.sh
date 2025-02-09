#!/bin/bash
#SBATCH --job-name=sensitivity_analysis
#SBATCH -A m2043
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --image=docker:adammwea/netsims_docker:v1

module load conda
conda activate netsims_env
#source activate netsims_env
#verify conda environment
conda env list
#shifter --image adammwea/netsims_docker:v1 \
python /pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/_3_run_sensitivity_analysis_on_sim_v2.py