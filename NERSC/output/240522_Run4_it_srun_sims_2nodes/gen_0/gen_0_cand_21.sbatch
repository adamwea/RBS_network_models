#!/bin/bash

cd .
srun -n 32 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240522_Run4_it_srun_sims_2nodes/gen_0/gen_0_cand_21_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240522_Run4_it_srun_sims_2nodes/240522_Run4_it_srun_sims_2nodes_netParams.py 
    