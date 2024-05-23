#!/bin/bash

cd .
echo $(pwd)        
cd NERSC        
python srun_extractor.py srun -n 32 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /workspaces/2DNetworkSimulations/NERSC/init.py simConfig=/workspaces/2DNetworkSimulations/NERSC/output/240523_Run1_vscode_debug/gen_0/gen_0_cand_2_cfg.json netParams=/workspaces/2DNetworkSimulations/NERSC/output/240523_Run1_vscode_debug/240523_Run1_vscode_debug_netParams.py 
    