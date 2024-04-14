## Imports
import os
from netpyne import specs
from netpyne import sim
import sys

'''
Define the parameter space for the evolutionary search
'''
params = specs.ODict()

'''
Modify the following parameters as needed
'''
## Constant Params
#Prepare constant netparams parameterrs
try: 
    sys.argv[2]
    seconds = int(sys.argv[2])
except: 
    #sys.argv.append('100')
    seconds = 1
print(f"Duration: {seconds} seconds")
params['duration_seconds'] = seconds
params['probLengthConst'] = [50, 2000] # length constant for conn probability (um)    

## General Params

#Propagation velocity
propVelocity = 1
params['propVelocity'] = propVelocity #, propVelocity/100, propVelocity/10, propVelocity*10, propVelocity*100]

#Cell Params
params['probIE'] = [0, 1] # min: 0.2, max: 0.6
params['probEE'] = [0, 1] # min: 0.1, max: 0.3
params['probII'] = [0, 1] # min: 0.2, max: 0.6
params['probEI'] = [0, 1] # min: 0.1, max: 0.3

params['weightEI'] = [0.0001, 0.1] # min: 0.001, max: 0.01
#params['weightIE'] = [0.001, 0.1] # min: 0.01, max: 0.1
params['weightIE'] = [0.0001, 0.1]
params['weightEE'] = [0.0001, 0.1] # min: 0.001, max: 0.01
#params['weightII'] = [0.001, 0.1] # min: 0.01, max: 0.1
params['weightII'] = [0.0001, 0.1]

params['gnabar_E'] = [0, 5] 
params['gkbar_E'] = [0, 0.5] 
params['gnabar_I'] = [0, 1.5] 
params['gkbar_I'] = [0.005, 2] 

#Hold these constant for now
# params['tau1_exc'] = [0.5, 1.5] 
# params['tau2_exc'] = [3.0, 10.0] 
# params['tau1_inh'] = [0.05, 0.2] 
# params['tau2_inh'] = [5.0, 15.0]
#default values for now
params['tau1_exc'] = 0.8
params['tau2_exc'] = 6.0
params['tau1_inh'] = 0.8
params['tau2_inh'] = 9.0

#Stimulation Params
params['stimWeight'] = [0, 0.002] 
params['stim_rate'] = [0, 0.15] 
params['stim_noise'] = [0.2, 0.6]

# Prepare params for evol batching.
# If any param is a single value, convert to list with that value twice
for key, value in params.items():
    if isinstance(value, (int, float)):
        params[key] = [value, value]
    elif isinstance(value, list) and len(value) == 1:
        params[key] = [value[0], value[0]]

evol_param_space = params

def define_population_params(batch_run_path = None):
    '''
    These parameters are run seperately so that neuron locs are consistent across simulations.
    '''
    assert batch_run_path is not None, 'batch_run_path must be specified'

    ## define pathing
    filename = 'netParams_popParams'
    full_path = os.path.join(batch_run_path, filename+'.json')

    ## Check if file exists, if so, return
    if os.path.exists(full_path):
        try: 
            netParams = sim.loadNetParams(filename+'.json', setLoaded=False)
            return netParams
        except: pass
        

    ## Hold Neuron Locations Constant Across Simulations
    netParams = specs.NetParams()   # object of class NetParams to store the network parameters

    ## Population parameters
    netParams.sizeX = 4000 # x-dimension (horizontal length) size in um
    netParams.sizeY = 2000 # y-dimension (vertical height or cortical depth) size in um
    netParams.sizeZ = 0 # z-dimension (horizontal length) size in um
    #netParams.probLengthConst = 500 # length constant for conn probability (um)    
    netParams.popParams['E'] = {
        'cellType': 'E', 
        'numCells': 300, 
        'yRange': [100,1900], 
        'xRange': [100,3900]}
    netParams.popParams['I'] = {
        'cellType': 'I', 
        'numCells': 100, 
        'yRange': [100,1900], 
        'xRange': [100,3900]}
        
    ##Save network params to file
    netParams.save(full_path)

    return netParams