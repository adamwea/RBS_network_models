#!/bin/bash

export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose
export FI_LOG_LEVEL=debug

cd .
mpirun --mca mtl_base_verbose 100 --map-by ppr:8:node -n 32 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240422_Run2_MPIsxOMPs_1x16x16/gen_4/gen_4_cand_15_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240422_Run2_MPIsxOMPs_1x16x16/240422_Run2_MPIsxOMPs_1x16x16_netParams.py 
    