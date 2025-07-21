from batching.cfg_helper import *

# Path to runtime kwargs
cfg_runtime_kwargs_path = './workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/batching/cfg_kwargs.json'

# Initialize simulation configuration
cfg = specs.SimConfig()

# Import runtime kwargs
assert os.path.exists(cfg_runtime_kwargs_path), f'{cfg_runtime_kwargs_path} not found'
with open(cfg_runtime_kwargs_path, 'r') as f:
    cfg_kwargs = json.load(f)
cfg.duration_seconds = cfg_kwargs['duration_seconds']
cfg.target_script_path = cfg_kwargs['target_script_path']

# Import evolutionary parameters
cfg.params_script_path = cfg_kwargs['param_script_path']
params = handle_params_import(cfg.params_script_path)
print('Evolutionary parameters imported successfully.')

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
num_excite, num_inhib = get_cell_numbers_from_fitness_target_script(target_script_path=cfg.target_script_path)
#numExcitatory, numInhibitory = USER_num_excite, USER_num_inhib
E_cells = random.sample(range(num_excite), min(2, num_excite))
I_cells = random.sample(range(num_inhib), min(2, num_inhib))
cfg.num_excite = num_excite
cfg.num_inhib = num_inhib
cfg.recordCells = [('E', E_cells), ('I', I_cells)]
print(f'Cells selected for recording: {cfg.recordCells}')

#testing new params
#cfg.coreneuron = True
#cfg.dump_coreneuron_model = True
# cfg.cache_efficient = True
# #cfg.cvode_active = True
# cfg.use_fast_imem = True
# cfg.allowSelfConns = True
# cfg.oneSynPerNetcon = False

# success message
print('cfg.py script completed successfully.')    