#!/bin/bash

# Preliminary error handling
#set -e

# Load necessary modules. Ensure these modules are available on all node types.
module load conda
conda activate neuron_env
module load openmpi
module load cmake

# Activate the existing Conda environment
#conda activate neuron_env

# Install necessary Python packages
conda install -c conda-forge numpy cython matplotlib -y

# Navigate to the home directory and clone/update NEURON repository
cd $HOME
if [ -d "nrn" ]; then
    echo "Updating existing NEURON repository..."
    cd nrn
    git pull
else
    echo "Cloning NEURON repository..."
    git clone https://github.com/neuronsimulator/nrn.git
    cd nrn
fi

# Create or clear the build directory
mkdir -p build
cd build
rm -rf *  # Clean the build directory

# Configure NEURON using CMake
cmake .. -DNRN_ENABLE_INTERVIEWS=OFF \
         -DNRN_ENABLE_MPI=ON \
         -DNRN_ENABLE_PYTHON=ON \
         -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python \
         -DCMAKE_INSTALL_PREFIX=$HOME/neuron

# Build and install NEURON
make -j$(nproc)
make install

# Setting the environment variables to be used by NEURON
echo "Exporting necessary PATH and LD_LIBRARY_PATH variables..."
export PATH=$HOME/neuron/bin:$PATH
export LD_LIBRARY_PATH=$HOME/neuron/lib:$LD_LIBRARY_PATH

# Output the versions to check successful installation
echo "Checking installed versions of NEURON, MPI, and C Compiler..."
nrniv --version
mpiexec --version
mpicc --version

echo "NEURON installation completed successfully."
