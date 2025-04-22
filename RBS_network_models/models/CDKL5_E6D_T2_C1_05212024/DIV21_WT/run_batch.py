from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.batch import batchEvol_v2 as batchEvol
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.evol_params import params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.conv_params import conv_params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.conv_params import mega_params
import netpyne

try:
    from mpi4py import MPI
    print("MPI4PY is installed, running in parallel mode")
except ImportError:
    print("WARNING: mpi4py not installed, running in single process mode")
    print("this is fine if debugging in login node, but not for batch jobs")
    pass

# import sys
# sys.exit()

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
    #'seed_dir': "/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/seeds"
    'seeds': [
        # initial seed prior to # aw 2025-04-21 14:43:45
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/_propVelocity_3/_propVelocity_3_data.pkl',
        
        # # aw 2025-04-21 14:43:49 - adding stuff from last batch and from sensitivity analysis from a while ago prior to big batch
        # # I added like, more than half of the good ones from sensitivity analysis. They're all permutations of the same simualtion...so its probably not worth adding more.
        # # I'm also going to tweak fitness func to weigh burst rate more heavily than other things for this batch
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-21/gen_5/gen_5_cand_0_data.pkl',
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-21/gen_10/gen_10_cand_3_data.pkl',
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/tau2_exc_2/tau2_exc_2_data.pkl',
        #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/I_L_mean_4/I_L_mean_4_data.pkl',
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probII_1/probII_1_data.pkl',
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probEE_4/probEE_4_data.pkl',
        #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/I_L_mean_2/I_L_mean_2_data.pkl',
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/gkbar_I_4/gkbar_I_4_data.pkl',
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightII_0/weightII_0_data.pkl',
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/tau1_inh_1/tau1_inh_1_data.pkl',
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/gnabar_I_2/gnabar_I_2_data.pkl',
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/propVelocity_1/propVelocity_1_data.pkl',
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/gnabar_I_4/gnabar_I_4_data.pkl',
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probEE_1/probEE_1_data.pkl',
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/gnabar_E_3/gnabar_E_3_data.pkl',
        #weightEI_3
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightEI_3/weightEI_3_data.pkl',
        #probEI_1
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probEI_1/probEI_1_data.pkl',
        #weightEE_3
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightEE_3/weightEE_3_data.pkl',
        #weightEI_2
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightEI_2/weightEI_2_data.pkl',
        #gnabar_I_3
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/gnabar_I_3/gnabar_I_3_data.pkl',
        #weightEI_5
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightEI_5/weightEI_5_data.pkl',
        #probEE_2
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probEE_2/probEE_2_data.pkl',
        #weightII_3
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightII_3/weightII_3_data.pkl',
        #weightII_5
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightII_5/weightII_5_data.pkl',
        #weightEI_1
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightEI_1/weightEI_1_data.pkl',
        #weightII_1
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightII_1/weightII_1_data.pkl',
        #tau2_inh_2
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/tau2_inh_2/tau2_inh_2_data.pkl',
        #gnabar_E_2
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/gnabar_E_2/gnabar_E_2_data.pkl',
        #tau1_exc_4
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/tau1_exc_4/tau1_exc_4_data.pkl',
        #tau1_inh_5
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/tau1_inh_5/tau1_inh_5_data.pkl',
        #gnabar_E_5
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/gnabar_E_5/gnabar_E_5_data.pkl',
        #probII_5
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probII_5/probII_5_data.pkl',
        #tau1_inh_0
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/tau1_inh_0/tau1_inh_0_data.pkl',
        #tau1_exc_1
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/tau1_exc_1/tau1_exc_1_data.pkl',
        #probIE_5
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probIE_5/probIE_5_data.pkl',
        #gnabar_I_1
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/gnabar_I_1/gnabar_I_1_data.pkl',
        #tau1_exc_3
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/tau1_exc_3/tau1_exc_3_data.pkl',
        #weightEE_4
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightEE_4/weightEE_4_data.pkl',
        #tau1_inh_2
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/tau1_inh_2/tau1_inh_2_data.pkl',
        #probLengthConst_0
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probLengthConst_0/probLengthConst_0_data.pkl',
        #probLengthConst_1
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probLengthConst_1/probLengthConst_1_data.pkl',
        #probLengthConst_2
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probLengthConst_2/probLengthConst_2_data.pkl',
        #probLengthConst_3
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probLengthConst_3/probLengthConst_3_data.pkl',
        #probLengthConst_4
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probLengthConst_4/probLengthConst_4_data.pkl',
        #probLengthConst_5
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probLengthConst_5/probLengthConst_5_data.pkl',
        #weightEI_4
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightEI_4/weightEI_4_data.pkl',
        #weightEE_5
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightEE_5/weightEE_5_data.pkl',
        #probII_2
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probII_2/probII_2_data.pkl',
        #weightII_2
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightII_2/weightII_2_data.pkl',
        #probEI_2
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probEI_2/probEI_2_data.pkl',
        #gkbar_I_2
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/gkbar_I_2/gkbar_I_2_data.pkl',
        #gkbar_E_0
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/gkbar_E_0/gkbar_E_0_data.pkl',
        #gnabar_E_4
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/gnabar_E_4/gnabar_E_4_data.pkl',
        #tau2_inh_4
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/tau2_inh_4/tau2_inh_4_data.pkl',
        #gkbar_E_5
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/gkbar_E_5/gkbar_E_5_data.pkl',
        #weightEI_4
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightEI_4/weightEI_4_data.pkl',
        #probIE_4
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probIE_4/probIE_4_data.pkl',
        #tau1_exc_5
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/tau1_exc_5/tau1_exc_5_data.pkl',
        #tau2_inh_5
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/tau2_inh_5/tau2_inh_5_data.pkl',
        #probEI_4
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probEI_4/probEI_4_data.pkl',
        #weightEI_5
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightEI_5/weightEI_5_data.pkl',
        #probII_4
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/probII_4/probII_4_data.pkl',
        #weightII_4
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/weightII_4/weightII_4_data.pkl',
    ],
    "conv_params": conv_params,
    "mega_params": mega_params,
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