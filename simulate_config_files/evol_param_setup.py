## Imports
import os
from netpyne import specs
from netpyne import sim
import sys

'''
Define the parameter space for the evolutionary search
'''
from simulate_config_files.evol_param_space import params
#assert USER_seconds, 'USER_seconds must be specified in USER_INPUTS.py'
#seconds = USER_seconds
#print(f"Duration: {seconds} seconds")
#params['duration_seconds'] = seconds

# Prepare params for evol batching.
# If any param is a single value, convert to list with that value twice
for key, value in params.items():
    if isinstance(value, (int, float)):
        params[key] = [value, value]
    elif isinstance(value, list) and len(value) == 1:
        params[key] = [value[0], value[0]]

evol_param_space = params
#print(evol_param_space)

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
            #sys.exit()
 
            netParams = sim.loadNetParams(filename+'.json', setLoaded=False)
            return netParams
        except: pass

    from netParams_constant import netParams
        
    ##Save network params to file
    netParams.save(full_path)

    return netParams