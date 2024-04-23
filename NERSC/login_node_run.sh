#module load python/3.9
export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose
export FI_LOG_LEVEL=debug

module load python
module load conda
module unload cray-mpich
module unload cray-libsci
module use /global/common/software/m3169/perlmutter/modulefiles
module load openmpi

conda activate 2DSims

mkdir -p NERSC/output/job_outputs

# Execute the application and capture all output
python3 NERSC/batchRun.py test_uneven 30