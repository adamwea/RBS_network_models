#To run this, run these lines:
#salloc --nodes=2 -C cpu --ntasks-per-node=256 -q interactive -t 04:00:00 --exclusive
#bash NERSC/interactive_node_run.sh

#specify the duration of the simulation and the label of the batch run
Duration_Seconds=10
Batch_Run_Label=$SLURM_JOB_NAME

#initialize
module load conda
conda activate 2DSims_nersc
#module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
#MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
#module load openmpi
# mkdir -p NERSC/output/job_outputs
# nrniv --version #check the versions of the installed software
# mpiexec --version
# mpicc --version

# prepare batch run
export PYTHONWARNINGS="ignore:DeprecationWarning" 
run_path=$(python3 NERSC/USER_init_new_batch.py ${Batch_Run_Label}) # Initialize the batch file and store the return value in run_path
echo "Run path: ${run_path}"
JOB_ID=$SLURM_JOB_ID
# export OMP_NUM_THREADS=1
# NODES=${SLURM_NNODES} # Number of nodes (automatically retrieved from the SLURM environment variable)
# PPN=${SLURM_NTASKS_PER_NODE} # Processes per node (set by SLURM ntasks-per-node)
# PPN=128
# NP=$((NODES * PPN)) # Total number of processes

#save copy of this batch_script in the run_path
full_path=$(realpath $0)
cp ${full_path} ${run_path}/run_in_interactive_node.sh

# Run the MPI command
cd NERSC # Change directory
echo "Running the simulation"
mpiexec --display-map --map-by ppr:1:core \
nrniv -mpi batchRun.py ${run_path} ${Duration_Seconds} \
> ${run_path}/job_${JOB_ID}_mpi_output.txt \
2> ${run_path}/job_${JOB_ID}_mpi_error.txt