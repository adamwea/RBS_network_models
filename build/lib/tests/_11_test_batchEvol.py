from RBS_network_models.developing.CDKL5.DIV21.src.batch import batchEvol
# =============================================================================
seed_dir = "/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/CDKL5/DIV21/seeds"
output_dir = "/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/tests/outputs/test_batchRun"
# ================================================================================
'''
salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=2 --tasks-per-node=32 --cpus-per-task=4 --image=adammwea/netsims_docker:v1
module load conda
conda activate netsims_env
python /pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/tests/_9_test_batchEvol.py

'''
# shifter --image=adammwea/netsims_docker:v1 /bin/bash
# conda create -n netsims_env38 -f /pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/conda/netsims_env38.yml
kwargs = {
    'seed_dir': seed_dir,
    'output_dir': output_dir
    }
batchEvol(**kwargs)