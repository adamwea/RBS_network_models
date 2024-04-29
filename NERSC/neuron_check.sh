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

#check the neuron version
nrniv --version
mpiexec --version
mpicc --version