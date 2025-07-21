from netpyne import sim
import os
from RBS_network_models.sim_analysis import process_simulation_v3
#from RBS_network_models.models.Organoid_RTT_R270X.DIV112_WT.src.conv_params import conv_params, mega_params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.conv_params import conv_params, mega_params
from datetime import datetime

# paths ===================================================================================================
file_name = 'test_run' # define the filename for the simulation
date = datetime.now().strftime("%y%m%d")
npy_path = ('/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well005/metrics.npy') # define the path to the metrics.npy file
saveFolder = (f'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/single_runs/{date}_{file_name}') # define the path to the save folder for the simulation
#os.makedirs(saveFolder, exist_ok=True) # create the save folder if it does not exist

# run simulation ===================================================================================================
#from RBS_network_models.models.Organoid_RTT_R270X.DIV112_WT.src import init # NOTE this will create, run and analyze a simulation
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src import init # NOTE this will create, run and analyze a simulation
print('Simulation successfully ran!')
sim.cfg.filename = file_name
sim.cfg.saveFolder = os.path.abspath(saveFolder)

#save cfg and netParams to file ============================================================================================
netParamsPath = os.path.join(saveFolder, sim.cfg.filename+'_netParams.json')
netParamsPath = os.path.abspath(netParamsPath)
sim.net.params.save(netParamsPath)
sim.saveData()
print('Data saved successfully!')

# test typical simulation analysis - including fitness function ===========================================================
sim_data_path = os.path.join(sim.cfg.saveFolder, sim.cfg.filename + '_data.pkl')
reference_data_path = npy_path
kwargs = {
    'simData': sim.allSimData,
    'popData': sim.net.allPops,
    'cellData': sim.net.allCells,
    'sim_data_path': sim_data_path,
    'reference_data_path': reference_data_path,
    'conv_params': conv_params,
    'mega_params': mega_params,
    #'fitnessFuncArgs': fitnessFuncArgs,
    'DEBUG_MODE': False,
}


# TODO: broken , fix later # aw 2025-03-20 21:49:56
# process_simulation_v3(kwargs)
# print('Parameters are selected randomly here, if neither E nor I cells are firing, try running again.') #TODO: I need a config that I reliably know will work for testing purposes.