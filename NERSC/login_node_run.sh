module load python/3.9
module load conda
conda activate 2DSims

# Execute the application and capture all output
python3 NERSC/batchRun.py hpcslurm_NetPyNE_test 30