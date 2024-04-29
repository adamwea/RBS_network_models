#!/bin/bash
#SBATCH --job-name=4Nodes_4x128
#SBATCH -A m2043
#SBATCH -t 00:30:00
#SBATCH --nodes=4
#SBATCH --output=NERSC/output/job_outputs/job_output_%j_4Nodes_4x128.txt
#SBATCH --error=NERSC/output/job_outputs/job_error_%j_4Nodes_4x128.txt
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH --exclusive
    
module load conda
conda activate neuron_env
module load openmpi
mkdir -p NERSC/output/job_outputs

# Set the library path for NEURON and Python libraries
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set the path to include the NEURON binaries
export PATH=$HOME/neuron/bin:$PATH

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
cd NERSC
mpiexec --map-by ppr:128:node -np --display-map 512 nrniv -mpi batchRun.py 4Nodes_4x128 5
    