#run in terminal before running this
#salloc --nodes=2 -C cpu --ntasks-per-node=128 -q interactive -t 04:00:00 --job-name=interactive_node_run --exclusive
#bash NERSC/interactive_node_run.sh
export PYTHONWARNINGS="ignore:DeprecationWarning"

module load conda
conda activate 2DSims_nersc
module load openmpi
export OMP_NUM_THREADS=1
mkdir -p NERSC/output/job_outputs

# # Set the library path for NEURON and Python libraries
# export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$HOME/neuron/lib:$LD_LIBRARY_PATH

# # Set the path to include the NEURON binaries
# export PATH=$HOME/neuron/bin:$PATH

# Number of nodes (automatically retrieved from the SLURM environment variable)
NODES=${SLURM_NNODES}
# Processes per node (set by SLURM ntasks-per-node)
PPN=${SLURM_NTASKS_PER_NODE}
# Total number of processes
NP=$((NODES * PPN))

# Change directory
cd NERSC
#check the versions of the installed software
nrniv --version
mpiexec --version
mpicc --version
echo "Running the simulation"
# Run the MPI command
#mpiexec --display-map --map-by ppr:1:core -np ${NP} nrniv -mpi batchRun.py interactive_node_run 10
echo "mpiexec --display-map --map-by ppr:${PPN}:node -np ${NP} nrniv -mpi batchRun.py interactive_node_run 10"
mpiexec --display-map --map-by ppr:${PPN}:node -np ${NP} nrniv -mpi batchRun.py interactive_node_run 10