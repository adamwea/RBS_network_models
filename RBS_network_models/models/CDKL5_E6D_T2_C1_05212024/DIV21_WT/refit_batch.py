'''
This script aims to simulate the optimization case of runing batch simulations to help debug/develop the fitness function.
'''
# imports ===============================================================================
from netpyne import sim
from RBS_network_models.fitnessFunc import fitnessFunc_v3
import numpy as np
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.conv_params import conv_params, mega_params
import glob

# main logic =================================================================
# define reference data path
reference_data_path = '/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well005/metrics.npy'

# define batch path
#batch_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-22/gen_1'

# aw 2025-04-30 13:39:56
batch_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26'


# get all sim data paths, look for files ending in _data.pkl
#sim_data_paths = glob.glob(batch_path + '/*_data.pkl')

# get all sim data paths, look for files ending in _data.pkl, recursively
sim_data_paths = glob.glob(batch_path + '/**/*_data.pkl', recursive=True)

for sim_data_path in sim_data_paths:
    # load sim data
    sim.load(sim_data_path)
    simData = sim.allSimData.todict().copy()

    # define fitness function args
    fitnessFuncArgs = {
        #'reference_data': reference_data,
        'conv_params': conv_params,
        'mega_params': mega_params,
        'plot_sim': True,
        'reference_data_path': reference_data_path,
        'batching': False, #NOTE: this is the only difference between this and the fitnessFunc_v3 during batch processing - if true fitnessFunc will try to load data from call stack, which wont work here.
        'sim_data_path': sim_data_path, # needed if batching is False
        
        # compute_network_metrics args
        #'try_load': True,
        'try_load': False,
        'run_parallel': True,
        'max_workers': 16,
        #'max_workers': 256,
        'burst_sequencing': True,       
    } 

    #run fitness function
    avg_fitness = fitnessFunc_v3(simData, **fitnessFuncArgs)
    print(f'Avg fitness for {sim_data_path}: {avg_fitness}')
    
    # clear sim object
    sim.clearAll()