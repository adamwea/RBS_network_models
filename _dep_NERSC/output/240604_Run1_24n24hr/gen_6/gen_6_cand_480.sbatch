#!/bin/bash

cd .
        
sleep 10        
echo $(pwd)        
python srun_extractor.py srun -N 1 -n 48 --sockets-per-node 1 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240604_Run1_24n24hr/gen_6/gen_6_cand_480_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240604_Run1_24n24hr/240604_Run1_24n24hr_netParams.py 
    