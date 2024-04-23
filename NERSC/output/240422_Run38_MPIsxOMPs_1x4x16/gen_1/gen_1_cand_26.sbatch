#!/bin/bash

export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose
export FI_LOG_LEVEL=debug
export OMPI_MCA_pml=ob1

cd .
mpirun --mca mtl_base_verbose 100 --use-hwthread-cpus --nooversubscribe --map-by ppr:1:core --bind-to core --report-bindings --display-map --display-topo --display-devel-map -n 4 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240422_Run38_MPIsxOMPs_1x4x16/gen_1/gen_1_cand_26_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240422_Run38_MPIsxOMPs_1x4x16/240422_Run38_MPIsxOMPs_1x4x16_netParams.py 
    