#!/bin/bash

cd .
srun -n 128 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run22_2proc/gen_0/gen_0_cand_124_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run22_2proc/240511_Run22_2proc_netParams.py 
    