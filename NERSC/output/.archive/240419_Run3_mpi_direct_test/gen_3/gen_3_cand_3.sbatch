#!/bin/bash

cd .
mpirun --mca mtl_base_verbose 100 -n 16 nrniv -python -mpi /global/u2/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240419_Run3_mpi_direct_test/gen_3/gen_3_cand_3_cfg.json netParams=/global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240419_Run3_mpi_direct_test/240419_Run3_mpi_direct_test_netParams.py 
    