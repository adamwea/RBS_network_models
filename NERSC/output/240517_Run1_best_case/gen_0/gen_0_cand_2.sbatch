#!/bin/bash

cd .
srun -n 16 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240517_Run1_best_case/gen_0/gen_0_cand_2_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240517_Run1_best_case/240517_Run1_best_case_netParams.py 
    