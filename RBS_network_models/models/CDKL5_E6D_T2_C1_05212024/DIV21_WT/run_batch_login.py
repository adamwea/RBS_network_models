from RBS_network_models.Organoid_RTT_R270X.DIV112_WT.src.batch import batchEvol_v2 as batchEvol
from RBS_network_models.Organoid_RTT_R270X.DIV112_WT.src.evol_params import params
from RBS_network_models.Organoid_RTT_R270X.DIV112_WT.src.conv_params import conv_params
from RBS_network_models.Organoid_RTT_R270X.DIV112_WT.src.conv_params import mega_params
try:
    from mpi4py import MPI
except ImportError:
    print("WARNING: mpi4py not installed, running in single process mode")
    print("this is fine if debugging in login node, but not for batch jobs")
    pass

# main ========================================================================================
kwargs = {
    'parameter_space': params,
    'batchFolder': (
        #'/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/batch_runs'
        ),
    'reference_data_paths': { # for fitting against
        '/global/homes/a/adammwea/pscratch/zoutputs/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well005/metrics.npy'
        },
    'runCfg_script_path': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/init.py',
    #'seed_dir': "/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/seeds"
    "conv_params": conv_params,
    "mega_params": mega_params,
    }

batchEvol(**kwargs)

# run options =======================================================================
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