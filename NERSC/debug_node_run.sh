#!/bin/bash
#SBATCH --job-name=debug_node_run
#SBATCH -A m2043
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --output=NERSC/output/job_outputs/job_output_%j_debug_node_run.txt
#SBATCH --error=NERSC/output/job_outputs/job_error_%j_debug_node_run.txt
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH --exclusive
    
bash NERSC/neuron_check.sh

#bash NERSC/neuron_run.sh
module load conda
conda activate neuron_env
#conda activate 2DSims
module load openmpi
#module load openmpi/5.0.0rc12

mkdir -p NERSC/output/job_outputs

# Set the library path for NEURON and Python libraries
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/neuron/lib:$LD_LIBRARY_PATH

# Set the path to include the NEURON binaries
export PATH=$HOME/neuron/bin:$PATH

cd NERSC
nrniv --version
mpiexec --version
mpicc --version
echo "Running the simulation"
mpiexec --map-by ppr:128:node -np --display-map 128 nrniv -mpi batchRun.py debug_node_run 10
#mpiexec --map-by ppr:128:node -np 128 nrniv -mpi batchRun.py debug_node_run 10
    