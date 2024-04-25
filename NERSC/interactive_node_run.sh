#module load python/3.9
# export OMP_PROC_BIND=spread
# export KMP_AFFINITY=verbose
# export FI_LOG_LEVEL=debug

module load conda
conda activate 2DSims
module load openmpi

mkdir -p NERSC/output/job_outputs

# Execute the application and capture all output
#    mpiexec_command = f"mpiexec {mpiexec_flags} --use-hwthread-cpus -np {USER_total_cores} nrniv -mpi NERSC/batchRun.py {USER_JobName} {USER_seconds}"
cd NERSC
mpiexec --use-hwthread-cpus -np 128 nrniv -mpi batchRun.py scaling_up 5