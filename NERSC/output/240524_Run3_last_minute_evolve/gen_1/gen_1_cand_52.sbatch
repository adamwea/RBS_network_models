#!/bin/bash

cd .
        
sleep 4        
echo $(pwd)        
python srun_extractor.py srun -N 1 -n 16 --sockets-per-node 1 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240524_Run3_last_minute_evolve/gen_1/gen_1_cand_52_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240524_Run3_last_minute_evolve/240524_Run3_last_minute_evolve_netParams.py 
    