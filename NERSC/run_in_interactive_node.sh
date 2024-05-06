#To run this, run these lines:
#salloc --nodes=2 -C cpu --ntasks-per-node=128 -q interactive -t 04:00:00 --exclusive
#bash NERSC/interactive_node_run.sh

#Get the path of this script
#full_path=$(realpath $0)

#initialize
module load conda
conda activate 2DSims_nersc
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
module load openmpi
mkdir -p NERSC/output/job_outputs
export OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore:DeprecationWarning" 
NODES=${SLURM_NNODES} # Number of nodes (automatically retrieved from the SLURM environment variable)
PPN=${SLURM_NTASKS_PER_NODE} # Processes per node (set by SLURM ntasks-per-node)
NP=$((NODES * PPN)) # Total number of processes
python3 NERSC/USER_init_new_batch.py # Initialize the batch file

cd NERSC # Change directory
nrniv --version #check the versions of the installed software
mpiexec --version
mpicc --version
echo "Running the simulation"

# Run the MPI command
echo "mpiexec --display-map --map-by ppr:${PPN}:node -np ${NP} nrniv -mpi batchRun.py int_node_run 10"
#mpiexec --display-map --map-by ppr:${PPN}:node -np ${NP} nrniv -mpi batchRun.py int_node_run 10