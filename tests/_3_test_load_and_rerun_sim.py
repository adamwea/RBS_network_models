global PROGRESS_SLIDES_PATH, SIMULATION_RUN_PATH, REFERENCE_DATA_NPY, CONVOLUTION_PARAMS, DEBUG_MODE
#from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import *
#from DIV21.utils.sim_helper import *
from RBS_network_models.developing.utils.sim_helper import *
from netpyne import sim
import os
# ===================================================================================================

# load simulation data
# sim_data_path = ('/pscratch/sd/a/adammwea/workspace/'
#                 'RBS_neuronal_network_models/optimization_projects/'
#                 'CDKL5_DIV21/scripts/_1_sims/testing/test_run_2_data.pkl')
# #sim_data_path = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/_1_sims/testing/test_data.json'
# sim_net_path = (
#     '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing'
#     '/CDKL5/CDKL5_DIV21/tests/outputs/test_run_a_simulation/test_run_5_net.json'
# )

# sim_cfg_path = (
#     '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing'
#     '/CDKL5/CDKL5_DIV21/tests/outputs/test_run_a_simulation/test_run_5_cfg.json'
# )

sim_data_path = (
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5'
    '/CDKL5_DIV21/tests/outputs/test_run_a_simulation/test_run_5_data.pkl'
)

#apparently need to modify simcfg before loading
simConfig = sim.loadSimCfg(sim_data_path, setLoaded=False)
#simConfig = sim.loadSimCfg(sim_cfg_path, setLoaded=False)

# modify simulation parameters as needed
simConfig.simLabel = 'test_rerun'# I think, to apply changes, it's necessary to modfity simConfg, not cfg
saveFolder = '..test/outputs/test_load_and_rerun_sim'
simConfig.saveFolder = os.path.abspath(saveFolder)

#modify runtime options
duration_seconds = 1
simConfig.duration = 1e3 * duration_seconds  # likewise, I think it's necessary to modify netParams, not net.params or net
simConfig.verbose = True # NOTE: during connection formation, this will be VERY verbose
simConfig.validateNetParams = True
#simConfig.coreneuron = True

# load and run simulation
sim.load(sim_data_path, simConfig=simConfig)
#sim.load(sim_net_path, simConfig=simConfig)

# run simulation
sim.simulate()
sim.analyze()
sim.saveData()
print('Simulation successfully re-ran!')