#!/bin/bash
# Load/install the necessary modules
conda activate 2DSims
pip uninstall neuron
conda install -c conda-forge openmpi -y
conda install -c conda-forge cmake
conda install -c conda-forge numpy cython matplotlib -y

# Clone the NEURON repository
#record cwd 
cwd=$(pwd)
cd ${cwd}
git clone https://github.com/neuronsimulator/nrn.git
cd nrn

# Create a build directory
mkdir build
cd build
make clean

# Configure NEURON with CMake
cmake .. -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_MPI=ON -DNRN_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python -DCMAKE_INSTALL_PREFIX=$HOME/neuron

# Build and install NEURON
make -j10
make install

# Set the library path for NEURON and Python libraries
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$HOME/neuron/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${pwd}/neuron/lib:$LD_LIBRARY_PATH
export PATH=/home/adam/miniconda3/envs/2DSims/bin:$PATH
export OMPI_CC=/home/adam/miniconda3/envs/2DSims/bin/x86_64-conda_cos6-linux-gnu-gcc

# Set the path to include the NEURON binaries
export PATH=$HOME/neuron/bin:$PATH

# Test the installation
nrniv --version
mpiexec --version
mpicc --version