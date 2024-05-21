#!/bin/bash

cd .
srun -n 16 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240521_Run1_overnight_debug/gen_0/gen_0_cand_28_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240521_Run1_overnight_debug/240521_Run1_overnight_debug_netParams.py 
    