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
# pip install netpyne
# pip install dill
# pip install inspyred
# pip install psutil

# print slurm allocation info
# srun -n 1 -c 1 --cpu_bind=cores --mem=0 --time=0-00:10:00


#specify datetime to include in output file name
datetime=$(date '+%Y_%m_%d_%H_%M_%S')
mkdir -p workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/batch_runs/logs/

#python workspace/RBS_network_models/scripts/CDKL5/DIV21_WT/test_batch_opt_DIV21_WT_interact.py > workspace/RBS_network_models/data/CDKL5/DIV21/($datetime)_test_batch_opt_DIV21_WT_interact_node.log 2>&1 # send outputs and errors to file
python -u workspace/RBS_network_models/scripts/Organoid_RTT_R270X/DIV112_WT/test_batch_optimize_in_interact_node.py > workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/batch_runs/logs/${datetime}_test_batch_optimize_in_interact_node.log 2>&1 # send outputs and errors to file
#_batch_opt_DIV21_WT_interact_node.log 2>&1 # send outputs and errors to file