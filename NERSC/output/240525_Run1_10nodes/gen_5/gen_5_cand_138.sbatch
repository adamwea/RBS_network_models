#!/bin/bash

cd .
        
sleep 4        
echo $(pwd)        
python srun_extractor.py srun -N 1 -n 16 --sockets-per-node 1 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240525_Run1_10nodes/gen_5/gen_5_cand_138_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240525_Run1_10nodes/240525_Run1_10nodes_netParams.py 
    