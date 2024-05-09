#!/bin/bash

cd .
mpirun --mca mtl_base_verbose 100 -n 32 nrniv -python -mpi /global/u2/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240419_Run13_mpi_interactive_test30/gen_4/gen_4_cand_3_cfg.json netParams=/global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240419_Run13_mpi_interactive_test30/240419_Run13_mpi_interactive_test30_netParams.py 
    