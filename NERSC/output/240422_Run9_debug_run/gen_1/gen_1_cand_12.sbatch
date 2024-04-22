#!/bin/bash

export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose
export FI_LOG_LEVEL=debug

cd .
mpirun --mca mtl_base_verbose 100 --map-by ppr:4:node -n 4 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240422_Run9_debug_run/gen_1/gen_1_cand_12_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240422_Run9_debug_run/240422_Run9_debug_run_netParams.py 
    