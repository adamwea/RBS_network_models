#!/bin/bash
#SBATCH --job-name=test_sensitivity_analysis
#SBATCH --output=/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs
#SBATCH --error=/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/errors
#SBATCH --time=00:30:00
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --constraint=cpu
#SBATCH --mail-type=ALL

# Load the necessary modules
#module load conda
#conta activate netsims_env
module load cudatoolkit
module load python
source activate netsims_env

#TODO: This doesnt work in sbatch job for some reason.
# works in login node.

# Run sensitivity analysis for the CDKL5 DIV21 model
# python /pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/_4_test_sensitivity_analysis.py \
#     > /pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/_4_test_sensitivity_analysis.out

python /pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/tests/_4_test_sensitivity_analysis.py \
    > /pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/tests/outputs/_4_test_sensitivity_analysis.out
