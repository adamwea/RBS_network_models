#!/bin/bash

cd .
srun -n 128 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run28_32proc/gen_0/gen_0_cand_16_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run28_32proc/240511_Run28_32proc_netParams.py 
    