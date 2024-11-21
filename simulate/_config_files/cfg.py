'''Helper functions'''
def handle_params_import():
	'''Handle the import of the evolutionary parameter space
	
	Note: This function is necessary because the evolutionary parameter space 
	is handled differently when running simulations directly or from a batchRun script.
	'''
	try:
		from __main__ import params # Import evolutionary parameters from main i.e. batchRun script
	except ImportError:
		from simulate._config_files.evolutionary_parameter_space import params
		from RBS_network_simulations.simulate.batchRun_evol import rangify_params # use rangify_params from batchRun script to treat params as if it were imported from batchRun script
		params = rangify_params(params)
	return params

def get_cell_numbers_from_fitness_target_script():
    from RBS_network_simulations.simulate.batchRun_evol import import_module_from_path
    from temp_user_args import USER_fitness_target_script
    import_module_from_path(USER_fitness_target_script, 'fitnessFuncArgs') #dynamically import fitnessFuncArgs from USER_fitness_target_script defined as python scripts so that we can optimize different data
    from fitnessFuncArgs import fitnessFuncArgs
    
    num_excite = fitnessFuncArgs['features']['num_excite']
    num_inhib = fitnessFuncArgs['features']['num_inhib']
    
    return num_excite, num_inhib

'''Main cfg.py'''
# Import necessary packages
from netpyne import specs
import random

# Initialize simulation configuration
cfg = specs.SimConfig()

# --------------------------------------------------------
# Network configuration selection
# --------------------------------------------------------
cfg.networkType = '07Oct24'  # Updating everything while working odxckl4

# --------------------------------------------------------
# Network configuration iterations
# --------------------------------------------------------
if cfg.networkType == '07Oct24':
    # Import evolutionary parameters
    params = handle_params_import()
    print('Evolutionary parameters imported successfully.')

    # Import user arguments
    #from temp_user_args import USER_seconds, USER_num_excite, USER_num_inhib
    
    # Constants
    cfg.duration_seconds = params['duration_seconds'][0]
    cfg.duration = cfg.duration_seconds * 1e3  # Duration of the simulation, in ms
    cfg.cache_efficient = True  # Use CVode cache_efficient option to optimize load on many cores
    cfg.dt = 0.025  # Internal integration timestep to use
    cfg.verbose = False  # Show detailed messages
    cfg.recordStep = 0.1  # Step size in ms to save data (e.g., V traces, LFP, etc)
    cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'netCells', 'netPops']  # Data to save
    cfg.saveJson = True  # Save data in JSON format
    cfg.printPopAvgRates = [100, cfg.duration]  # Print population average rates
    cfg.savePickle = True  # Save params, network and sim output to pickle file

    # Record traces
    cfg.recordTraces['soma_voltage'] = {"sec": "soma", "loc": 0.5, "var": "v"}
    print('Recording soma voltage traces.')

    # Select cells for recording
    num_excite, num_inhib = get_cell_numbers_from_fitness_target_script()
    #numExcitatory, numInhibitory = USER_num_excite, USER_num_inhib
    E_cells = random.sample(range(num_excite), min(2, num_excite))
    I_cells = random.sample(range(num_inhib), min(2, num_inhib))
    cfg.num_excite = num_excite
    cfg.num_inhib = num_inhib
    cfg.recordCells = [('E', E_cells), ('I', I_cells)]
    print(f'Cells selected for recording: {cfg.recordCells}')