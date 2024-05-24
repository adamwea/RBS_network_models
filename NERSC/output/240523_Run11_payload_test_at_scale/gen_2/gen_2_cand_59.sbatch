#!/bin/bash

cd .
        
sleep 2        
echo $(pwd)        
python srun_extractor.py srun -N 1 -n 32 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240523_Run11_payload_test_at_scale/gen_2/gen_2_cand_59_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240523_Run11_payload_test_at_scale/240523_Run11_payload_test_at_scale_netParams.py 
    