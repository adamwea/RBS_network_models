#!/bin/bash

cd .
srun -n 32 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240523_Run9_it_srun_sims_8nodes/gen_1/gen_1_cand_10_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240523_Run9_it_srun_sims_8nodes/240523_Run9_it_srun_sims_8nodes_netParams.py 
    