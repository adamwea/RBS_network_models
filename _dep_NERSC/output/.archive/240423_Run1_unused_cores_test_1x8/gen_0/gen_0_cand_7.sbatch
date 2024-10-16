#!/bin/bash

export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose
export FI_LOG_LEVEL=debug
export OMPI_MCA_pml=ob1

# Get the list of used cores
used_cores=$(ps -e -o psr | sort -u | tr '
' ',' | sed 's/,$//')

# Get the total number of cores
total_cores=$(nproc)

# Generate the list of unused cores
unused_cores=""
for ((i=0; i<$total_cores; i++)); do
    if [[ $used_cores != *"$i"* ]]; then
        unused_cores="$unused_cores,$i"
    fi
done
unused_cores=${unused_cores#,}


cd .
taskset -c $unused_cores mpirun --mca mtl_base_verbose 100 --use-hwthread-cpus --nooversubscribe --map-by ppr:1:core --bind-to core --report-bindings --display-map --display-topo --display-devel-map -n 8 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240423_Run1_unused_cores_test_1x8/gen_0/gen_0_cand_7_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240423_Run1_unused_cores_test_1x8/240423_Run1_unused_cores_test_1x8_netParams.py 
    