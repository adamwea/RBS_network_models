#'''
#run with the following command in perlmutter bash terminal:
#bash NERSC/run_USER_INPUTS.sh
#'''

#module load python/3.9
module load conda
conda activate 2DSims
export $(python NERSC/generate_and_run_sbatch.py)