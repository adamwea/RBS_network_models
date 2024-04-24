#!/bin/bash
srun -n 32 check-hybrid.gnu.pm | sort -k4,6 #> output.log 2>&1
cd .
mpirun --mca mtl_base_verbose 100 -n 32 nrniv -python -mpi /global/u2/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240419_Run14_mpi_interactive_test30/gen_3/gen_3_cand_1_cfg.json netParams=/global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240419_Run14_mpi_interactive_test30/240419_Run14_mpi_interactive_test30_netParams.py 
    