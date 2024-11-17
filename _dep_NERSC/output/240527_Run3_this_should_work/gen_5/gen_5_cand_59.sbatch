#!/bin/bash

cd .
        
sleep 10        
echo $(pwd)        
python srun_extractor.py srun -N 1 -n 32 --sockets-per-node 1 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240527_Run3_this_should_work/gen_5/gen_5_cand_59_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240527_Run3_this_should_work/240527_Run3_this_should_work_netParams.py 
    