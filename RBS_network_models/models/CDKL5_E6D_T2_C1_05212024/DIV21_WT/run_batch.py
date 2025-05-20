from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.batch import batchEvol_v2 as batchEvol
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.evol_params import params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.conv_params import conv_params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.conv_params import mega_params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.seeds import seeds
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.fitness_schema.schema_1 import fit_schema
import netpyne

try:
    from mpi4py import MPI
    print("MPI4PY is installed, running in parallel mode")
except ImportError:
    print("WARNING: mpi4py not installed, running in single process mode")
    print("this is fine if debugging in login node, but not for batch jobs")
    pass

# main ========================================================================================
kwargs = {
    'parameter_space': params,
    'batchFolder': (
        #'/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/batch_runs'
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs'
        ),
    'reference_data_paths': { # for fitting against
        #'/global/homes/a/adammwea/pscratch/zoutputs/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well005/metrics.npy'
        '/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well005/metrics.npy'
        },
    'runCfg_script_path': (
        #'/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/init.py'
        '/global/homes/a/adammwea/workspace/repos/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_WT/src/init.py'
        ),
    "conv_params": conv_params,
    "mega_params": mega_params,
    "seeds": seeds,
    "fit_schema": fit_schema,
    }

batchEvol(**kwargs)

# run options =======================================================================
'''
shifter --image=adammwea/axonkilo_docker:v7 /bin/bash
'''


''' **************************************************************************
# run in login node for testing/debugging

# then run with python debugger or python as needed
python -m pdb /global/homes/a/adammwea/workspace/aw_scripts/Organoid_RTT_R270X_models/DIV112_WT/run_batch_login.py
python /global/homes/a/adammwea/workspace/aw_scripts/Organoid_RTT_R270X_models/DIV112_WT/run_batch_login.py

'''

''' **************************************************************************
# run everything in interactive node - run each script, one at a time

# step 1:
bash ~/workspace/aw_scripts/network_model_development/Organoid_RTT_R270X/DIV112_WT/test_batch_config_interact_allocate.sh

# step 2:
bash ~/workspace/aw_scripts/network_model_development/Organoid_RTT_R270X/DIV112_WT/test_batch_config_interact_run.sh

'''