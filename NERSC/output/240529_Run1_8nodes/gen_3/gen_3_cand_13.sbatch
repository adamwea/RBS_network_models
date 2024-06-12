#!/bin/bash

cd .
        
sleep 10        
echo $(pwd)        
python srun_extractor.py srun -N 1 -n 32 --sockets-per-node 1 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240529_Run1_8nodes/gen_3/gen_3_cand_13_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240529_Run1_8nodes/240529_Run1_8nodes_netParams.py 
    