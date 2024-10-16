module load python
module load conda

#
# cray-mpich and cray-libsci conflict with openmpi so unload them
#
module unload cray-mpich
module unload cray-libsci
module use /global/common/software/m3169/perlmutter/modulefiles
module load openmpi
#module load libfabric

# Set environment variables
export OMP_NUM_THREADS=256
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose
export FI_LOG_LEVEL=debug

# # Network diagnostics
# echo "Checking network interface details..."
# ifconfig

# echo "Available fabric interfaces:"
# fi_info -l

# # Ensure the provider is available
# if fi_info -p "verbs"; then
#     echo "Required provider is available, proceeding..."
# else
#     echo "Required provider is not available, aborting..."
#     exit 1
# fi

conda activate 2DSims

# Check if output directory exists, create if not
mkdir -p NERSC/output/job_outputs

python3 NERSC/batchRun.py mpi_interactive_test30 30 > NERSC/output/job_outputs/job_output_mpi_interactive_test30.txt