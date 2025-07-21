global PROGRESS_SLIDES_PATH, SIMULATION_RUN_PATH, REFERENCE_DATA_NPY, CONVOLUTION_PARAMS, DEBUG_MODE
#from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import *
#from DIV21.utils.sim_helper import *
from RBS_network_models.developing.utils.sim_helper import *
from netpyne import sim
# ===================================================================================================

# get configured simConfig and netParams
#from DIV21.src import init
from RBS_network_models.developing.CDKL5.DIV21.src import init
simConfig = init.simConfig
netParams = init.netParams

# set simulation configuration parameters
duration_seconds = 1
simConfig.duration = 1e3 *duration_seconds
simConfig.verbose = True # NOTE: during connection formation, this will be VERY verbose
simConfig.validateNetParams = True
simConfig.progressBar = True

#set paths
simConfig.simLabel = 'test_run_5' #change simLabel to avoid overwriting previous data
#simConfig.saveFolder = 'outputs/test_run_a_simulation'
saveFolder = '../tests/outputs/test_run_a_simulation'
simConfig.saveFolder = os.path.abspath(saveFolder)

#create simulation object
output = sim.create(netParams=netParams, simConfig=simConfig, output=True)
filename = sim.cfg.filename

# save cfg, netParams, and net
# NOTE: The following data should be all you need to re-run the simulation,
#       you shouldn't need the full data pickle file
#       ideally, deleting the data pickle file should not affect the ability to re-run the simulation
#      (it may be enough to just have the cfg and netParams files)

#save cfg and netParams to file
sim.cfg.save(filename+'_cfg.json')
sim.net.params.save(filename+'_netParams.json')

#save net as json
# sim.gatherData()
# sim.saveJSON(filename+'_net.json', {
#     'netCells': sim.net.allCells,
#     'netPops': sim.net.allPops,
#     })

# run simulation
#output_2 = sim.simulate()
#output_3 = sim.analyze(output=True)
sim.simulate()
sim.analyze()
sim.saveData()
print('Simulation successfully ran!')
print('Parameters are selected randomly here, if neither E nor I cells are firing, try running again.') #TODO: I need a config that I reliably know will work for testing purposes.