#for persistent environment, run source build_2DSims_nersc.sh
module load python
module load conda
conda env remove -n 2DSims_nersc #delete 2DSims_nersc environment if it exists
#conda create --name 2DSims_nersc --clone nersc-mpi4py
conda create -n 2DSims_nersc python=3.9
conda activate 2DSims_nersc
#install python=3.9
#conda install python=3.9 -y
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
module load openmpi
conda install pip
#conda install -y bison cmake flex git ncurses openmpi x11 xcomposite python-dev
pip install -r nrn_requirements.txt
pip install neuron
pip install netpyne
pip install inspyred

nrniv --version
mpiexec --version
mpicc --version