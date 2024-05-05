source ~/miniconda3/etc/profile.d/conda.sh
conda activate 2DSims
conda install -c conda-forge cmake -y
pip install "Cython<3.0"
sudo ln -s /usr/lib/x86_64-linux-gnu/libpthread.so.0 /lib/libpthread.so.0
sudo ln -s /home/adam/miniconda3/envs/2DSims/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib/libpthread_nonshared.a /usr/lib/libpthread_nonshared.a

# Clone NEURON repository
rm -rf nrn
git clone https://github.com/neuronsimulator/nrn.git
cd nrn

# Create a build directory
mkdir build
cd build

# Configure NEURON with CMake
cmake .. -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_MPI=ON -DNRN_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python -DCMAKE_INSTALL_PREFIX=$HOME/neuron

# Build and install NEURON
make -j10
make install

# Set environment variables
export PATH=`pwd`/x86_64/bin:$PATH
export PYTHONPATH=`pwd`/lib/python:$PYTHONPATH

# Install netpyne
pip install netpyne

# Test installations
pip show neuron | grep Version
python -c "import netpyne; print(netpyne.__version__)"