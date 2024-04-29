#!/bin/bash
# Load the necessary modules
module load conda
module load openmpi
module load cmake

# Create a new Conda environment
conda remove --name neuron_env_1 --all -y
conda create -n neuron_env_1 python=3.9 -y
source activate neuron_env_1

# Install necessary Python packages
conda install -c conda-forge numpy cython matplotlib -y

# Clone the NEURON repository
cd $HOME
git clone https://github.com/neuronsimulator/nrn.git
cd nrn


# Create a build directory
mkdir build
cd build
make clean

# Configure NEURON with CMake
cmake .. -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_MPI=ON -DNRN_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python -DCMAKE_INSTALL_PREFIX=$HOME/neuron

# Build and install NEURON
make -j128
make install

# Set the library path for NEURON and Python libraries
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/neuron/lib:$LD_LIBRARY_PATH

# Set the path to include the NEURON binaries
export PATH=$HOME/neuron/bin:$PATH

# Test the installation
nrniv --version
mpiexec --version
mpicc --version