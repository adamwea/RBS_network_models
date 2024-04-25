#!/bin/bash

export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose
export FI_LOG_LEVEL=debug

cd .
mpirun --mca mtl_base_verbose 100 --map-by ppr:8:node -n 16 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240422_Run6_MPIsxOMPs_1x16x8/gen_1/gen_1_cand_13_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240422_Run6_MPIsxOMPs_1x16x8/240422_Run6_MPIsxOMPs_1x16x8_netParams.py 
    