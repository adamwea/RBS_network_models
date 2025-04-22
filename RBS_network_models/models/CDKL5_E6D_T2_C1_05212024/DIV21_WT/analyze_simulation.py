from netpyne import sim
import os
from datetime import datetime as dt
import json
from RBS_network_models.sim_analysis import process_simulation_v3
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.conv_params import conv_params, mega_params
#from RBS_network_models.sensitivity_analysis import prepare_permuted_sim_v2
# Notes ===================================================================================================
reference_data_path = (
    '/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well005/metrics.npy'
    )
sim_data_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/_propVelocity_3/_propVelocity_3_data.pkl'
sim.load(sim_data_path) # load simulation data
saveFolder = sim.cfg.saveFolder # get save folder from sim.cfg

kwargs = {
    'sim_data_path': sim_data_path,
    'reference_data_path': reference_data_path,
    'conv_params': conv_params,
    'mega_params': mega_params,
    'simData': sim.allSimData,
    'popData': sim.net.allPops,
    'cellData': sim.net.allCells,
    'output_dir': saveFolder,
    #'fitnessFuncArgs': fitnessFuncArgs,
    'DEBUG_MODE': False,
}

process_simulation_v3(kwargs)
#print('Parameters are selected randomly here, if neither E nor I cells are firing, try running again.') #TODO: I need a config that I reliably know will work for testing purposes.