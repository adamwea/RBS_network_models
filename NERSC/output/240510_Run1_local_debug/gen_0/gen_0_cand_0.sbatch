#!/bin/bash
None
cd .
mpirun -bootstrap fork -n 4 nrniv -python -mpi /app/NERSC/init.py simConfig=/app/NERSC/output/240510_Run1_local_debug/gen_0/gen_0_cand_0_cfg.json netParams=/app/NERSC/output/240510_Run1_local_debug/240510_Run1_local_debug_netParams.py 
    