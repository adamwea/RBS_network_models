#!/bin/bash

cd .
        
sleep 4        
echo $(pwd)        
python srun_extractor.py srun -N 1 -n 16 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240524_Run1_big_run/gen_0/gen_0_cand_63_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240524_Run1_big_run/240524_Run1_big_run_netParams.py 
    