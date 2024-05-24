#!/bin/bash

cd .
        
sleep 2        
echo $(pwd)        
python srun_extractor.py srun -N 1 -n 32 --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240523_Run10_payload_test/gen_0/gen_0_cand_5_cfg.json netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240523_Run10_payload_test/240523_Run10_payload_test_netParams.py 
    