# bin/bash

module load conda
#conda create -n my_mpi4py_env python=3.8
conda activate my_mpi4py_env
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
#MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
# python -m pip install --upgrade pip

# # requirements
# pip install numpy
pip install /pscratch/sd/a/adammwea/workspace/RBS_network_models
pip install /pscratch/sd/a/adammwea/workspace/MEA_Analysis
# pip install scipy
# python -m pip install matplotlib
# pip install tdqm
# pip install pandas
# pip install h5py
# pip install spikeinterface==0.100.4
# pip install neuron
pip install netpyne
# pip install /pscratch/sd/a/adammwea/workspace/netpyne
# pip install dill
# pip install inspyred
# pip install psutil

# print slurm allocation info
# srun -n 1 -c 1 --cpu_bind=cores --mem=0 --time=0-00:10:00


datetime=$(date '+%Y_%m_%d_%H_%M_%S')

# echo file link so its easier to find
echo "printing outputs to workspace/RBS_network_models/data/CDKL5/DIV21/batch_runs/logs/${datetime}_test_batch_opt_DIV21_WT_interact_node.log" # send outputs and errors to file
mkdir -p workspace/RBS_network_models/data/CDKL5/DIV21/batch_runs/logs/

python -u workspace/RBS_network_models/scripts/CDKL5/DIV21_WT/test_batch_opt_DIV21_WT_interact.py > workspace/RBS_network_models/data/CDKL5/DIV21/batch_runs/logs/${datetime}_test_batch_opt_DIV21_WT_interact_node.log 2>&1 # send outputs and errors to file