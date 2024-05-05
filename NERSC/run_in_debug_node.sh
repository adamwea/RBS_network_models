#!/bin/bash
#SBATCH --job-name=debug_node_run
#SBATCH -A m2043
#SBATCH -t 00:05:00
#SBATCH --nodes=3
#SBATCH --output=NERSC/output/job_outputs/job_output_%j_debug_node_run.txt
#SBATCH --error=NERSC/output/job_outputs/job_error_%j_debug_node_run.txt
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH --exclusive
    
#bash NERSC/neuron_check.sh

#bash NERSC/neuron_run.sh
module load conda
conda activate 2DSims_nersc
module load openmpi

mkdir -p NERSC/output/job_outputs

# # Set the library path for NEURON and Python libraries
# export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$HOME/neuron/lib:$LD_LIBRARY_PATH

# # Set the path to include the NEURON binaries
# export PATH=$HOME/neuron/bin:$PATH

# change directory
cd NERSC

#check the versions of the installed software
nrniv --version
mpiexec --version
mpicc --version
echo "Running the simulation"

#run the simulation
# Number of nodes
NODES=2
# Processes per node
PPN=128
# Total number of processes
NP=$((NODES * PPN))
# Run the MPI command
echo mpiexec --display-map --map-by ppr:1:core -np ${NP} nrniv -mpi batchRun.py debug_node_run 10
#mpiexec --display-map --map-by ppr:1:core -np ${NP} nrniv -mpi batchRun.py debug_node_run 10



#graveyard:
#mpiexec --map-by ppr:128:node -np --display-map 128 nrniv -mpi batchRun.py debug_node_run_reg 10
#mpiexec --map-by ppr:128:node -np 128 nrniv -mpi batchRun.py debug_node_run 10
#mpiexec --map-by ppr:128:node -np 128 --report-bindings nrniv -mpi batchRun.py debug_node_run 10
#test other display options
#mpiexec --map-by ppr:128:node -np 128 --display-allocation nrniv -mpi batchRun.py debug_node_run 10
# mpiexec --map-by ppr:128:node -np 128 --display-map nrniv -mpi batchRun.py debug_node_run 10
#mpiexec --map-by ppr:128:node -np 128 --display-devel-map nrniv -mpi batchRun.py debug_node_run 10
#mpiexec --map-by ppr:128:node -np 128 --display-topo nrniv -mpi batchRun.py debug_node_run 10
# mpiexec --map-by ppr:128:node -np 128 --display-diff nrniv -mpi batchRun.py debug_node_run 10
# mpiexec --map-by ppr:128:node -np 128 --display-diffable nrniv -mpi batchRun.py debug_node_run 10
#mpiexec --map-by ppr:128:node -np 128 --display-topo-verbose nrniv -mpi batchRun.py debug_node_run 10

#mpiexec --display-map --map-by ppr:${PPN}:node -np ${NP} nrniv -mpi batchRun.py debug_node_run 10

#mpiexec --display-map --map-by ppr:128:node -np 128 nrniv -mpi batchRun.py debug_node_run 10
#mpiexec --use-hwthread-cpus --display-map --map-by ppr:256:node -np 256 nrniv -mpi batchRun.py debug_node_run 10
#mpiexec --use-hwthread-cpus --display-map --map-by ppr:1:hwthread -np 256 nrniv -mpi batchRun.py debug_node_run 10
#mpiexec --use-hwthread-cpus --display-map --map-by ppr:1:core -np 256 nrniv -mpi batchRun.py debug_node_run 10
#mpiexec --display-map --report-bindings --map-by ppr:128:node -np 128 nrniv -mpi $((python3 -c "print("hello world")"))



    