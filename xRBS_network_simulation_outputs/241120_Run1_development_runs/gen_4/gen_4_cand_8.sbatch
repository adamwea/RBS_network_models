#!/bin/bash

cd .
sleep 10
python ./sbatch_scripts/srun_extractor.py srun -N 1 --sockets-per-node 1 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -n 1  -python -mpi simulate/_config_files/init.py simConfig=/pscratch/sd/a/adammwea/RBS_network_simulations/xRBS_network_simulation_outputs/241120_Run1_development_runs/gen_4/gen_4_cand_8_cfg.json netParams=/pscratch/sd/a/adammwea/RBS_network_simulations/xRBS_network_simulation_outputs/241120_Run1_development_runs/241120_Run1_development_runs_netParams.py 
    