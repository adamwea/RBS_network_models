#!/bin/bash

export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose
export FI_LOG_LEVEL=debug

cd .
mpirun --mca mtl_base_verbose 100 --use-hwthread-cpus --nooversubscribe --map-by ppr:1:core --bind-to core --report-bindings --display-map --display-topo --display-devel-map -n 128 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240422_Run31_MPIsxOMPs_1x128x16/gen_0/gen_0_cand_0_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240422_Run31_MPIsxOMPs_1x128x16/240422_Run31_MPIsxOMPs_1x128x16_netParams.py 
    