#'''
#run with the following command in perlmutter bash terminal:
#bash NERSC/run_USER_INPUTS.sh [option]
#'''

#module load python/3.9
module load conda
conda activate neuron_env
module load openmpi
module load cmake
#mkdir -p NERSC/output/job_outputs

# Set the library path for NEURON and Python libraries
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

#Set the path to include the NEURON binaries
export PATH=$HOME/neuron/bin:$PATH

nrniv --version
export $(python3 NERSC/generate_and_run_sbatch.py)