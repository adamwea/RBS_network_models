#!/bin/bash
#SBATCH --job-name=plots
#SBATCH -A m2043
#SBATCH -t 05:00:00
#SBATCH -N 1
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --output=./NERSC/output/latest_job_init_error.txt
#SBATCH --error=./NERSC/output/latest_job_init_output.txt

#bash NERSC/neuron_run.sh
module load conda
conda activate preshifter
python3 NERSC/plot_simData_debug.py