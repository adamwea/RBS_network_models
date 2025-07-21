#from CDKL5.DIV21.utils.sim_helper import *
#from utils.sim_helper import *
from RBS_network_models.developing.utils.sim_helper import *
from netpyne import sim
from netpyne.sim import validator
# ===================================================================================================

# validate netParams and simConfig  #NOTE:  duration is automatically set to 1 second in the simConfig object, 
                                    #       it needs to be changed if a different duration is desired
#from DIV21.src import init
from RBS_network_models.developing.CDKL5.DIV21.src import init
simConfig = init.simConfig  
netParams = init.netParams
validator.validateNetParams(netParams)

# set simulation configuration parameters
duration_seconds = 1
simConfig.duration = 1e3 *duration_seconds
simConfig.verbose = True
simConfig.validateNetParams = False #no need to validate twice here

#set paths
saveFolder = '../tests/outputs/validate_netparams_and_cfg'
simConfig.saveFolder = os.path.abspath(saveFolder)
simConfig.simLabel = 'validation_test'

# create simulation object
output = sim.create(netParams=netParams, simConfig=simConfig, output=True)
filename = sim.cfg.filename
sim.cfg.save(filename+'_cfg.json')
sim.net.params.save(filename+'_netParams.json')

print('Simulation Object Created!')
print('If all validation tests passed, the simulation object has been created successfully.')