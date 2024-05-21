#!/bin/bash

cd .
srun -n 16 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240521_Run2_overnight_reg/gen_3/gen_3_cand_21_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240521_Run2_overnight_reg/240521_Run2_overnight_reg_netParams.py 
    