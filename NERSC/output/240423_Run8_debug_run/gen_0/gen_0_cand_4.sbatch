#!/bin/bash

export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose
export FI_LOG_LEVEL=debug
export OMPI_MCA_pml=ob1

# Get the list of used cores
export $(python NERSC/get_used_cores.py)

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
echo "Total cores: $total_cores"
echo "Used cores: $used_cores"
echo "Unused cores: $unused_cores"


cd .
mpirun --mca mtl_base_verbose 100 --use-hwthread-cpus --nooversubscribe --map-by ppr:1:core --bind-to core --report-bindings --display-map --display-topo --display-devel-map -n 8 nrniv -python -mpi /mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/init.py simConfig=/mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240423_Run8_debug_run/gen_0/gen_0_cand_4_cfg.json netParams=/mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240423_Run8_debug_run/240423_Run8_debug_run_netParams.py 
    