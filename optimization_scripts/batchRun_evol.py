''' 
Evolutionary algorithm optimization of a network using NetPyNE
'''
'''Setup Python environment for running the script'''
from pprint import pprint
import setup_environment as setup_environment
setup_environment.set_pythonpath()
#print sys path
import sys
pprint(sys.path)

'''
Import Local Modules

Note: Specify settings in kwargs dictionary, they are handled by parse_user_args.main(**kwargs)
'''
from modules.simulation_config import parse_kwargs


'''Import External Modules'''
from netpyne.batch import Batch

'''
Initialize
'''
##General Imports
import json
import os
import pandas as pd
from datetime import datetime
import logging
import importlib.util
import sys

'''Helper Functions'''
## Function to serialize the batch_config dictionary
class batchcfgEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj):
            return str(obj)
        return super().default(obj)

## Function to get HOF seeds
def get_HOF_seeds():

    print(f'Loading Hall of Fame from {USER_HOF}')
    assert os.path.exists(USER_HOF), f'USER_HOF file not found: {USER_HOF}'
    seeded_HOF_cands = pd.read_csv(USER_HOF).values.flatten()
    seeded_HOF_cands = [cfg.replace('_data', '_cfg') for cfg in seeded_HOF_cands]
    seeded_HOF_cands = [os.path.abspath(f'./{cfg}') for cfg in seeded_HOF_cands]
    for cfg in seeded_HOF_cands:
        if 'NERSC/NERSC' in cfg: 
            seeded_HOF_cands[seeded_HOF_cands.index(cfg)] = cfg.replace('NERSC/NERSC', 'NERSC')
        else: continue

    seeds = []
    for cfg in seeded_HOF_cands:
        if not os.path.exists(cfg): cfg = None #check if file exists, else set to None       
        if cfg is not None:    
            
            # open cfg file and extract simConfig
            with open(cfg, 'r') as f:
                seed = json.load(f)
            seed = seed['simConfig']
            
            #only keep overlap with USER_evelo_param_space.py
            seed = {
                key: evolutionary_parameter_space[key][0] if evolutionary_parameter_space[key][0] == evolutionary_parameter_space[key][1] 
                else seed[key] for key in evolutionary_parameter_space if key in seed
            }

            seed = list(seed.values()) #get rid of keys, just make list of values
            seed = [float(val) for val in seed] #make sure all values are floats
            seeds.append(seed)
            print(f'Successfully loaded seed from {cfg}')
            if len(seeds) >= USER_pop_size: break
        else: continue

    print(f'Loaded {len(seeds)} seeds from Hall of Fame')
    assert len(seeds) > 0, 'No seeds loaded from Hall of Fame'
    return seeds

# Define the function to import the module from a file path
def import_module_from_path(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

## Function to get batch config
def get_batch_config(batch_config_options=None):
    from modules.analysis.calculate_fitness import fitnessFunc
    import_module_from_path(USER_fitness_target_script, 'fitnessFuncArgs') #dynamically import fitnessFuncArgs from USER_fitness_target_script defined as python scripts so that we can optimize different data
    from fitnessFuncArgs import fitnessFuncArgs
    
    # Load HOF seeds if USER_seed_evol is True
    HOF_seeds = get_HOF_seeds() if USER_seed_evol else None

    # Extract the parameters
    run_path = batch_config_options['run_path']
    batch_label = batch_config_options['batchLabel']

    batch_config = {
        'batchLabel': batch_label,
        'saveFolder': run_path,
        'method': USER_method,
        'cfgFile': USER_cfg_file,
        'netParamsFile': USER_netParamsFile,
        'runCfg': {
            'type': USER_runCfg_type,
            'script': USER_init_script,
            'mpiCommand': USER_mpiCommand,
            'nrnCommand': USER_nrnCommand,
            'nodes': USER_nodes,
            'coresPerNode': USER_cores_per_node,
            'reservation': None,
            'skip': USER_skip,
        },
        'evolCfg': {
            'evolAlgorithm': 'custom',
            'fitnessFunc': fitnessFunc,
            'fitnessFuncArgs': {**fitnessFuncArgs, 'pop_size': USER_pop_size},
            'pop_size': USER_pop_size,
            'num_elites': USER_num_elites,
            'mutation_rate': USER_mutation_rate,
            'crossover': USER_crossover,
            'maximize': False,
            'max_generations': USER_max_generations,
            'time_sleep': USER_time_sleep,
            'maxiter_wait': USER_maxiter_wait,
            'defaultFitness': 1000,
            'seeds': HOF_seeds,
        }
    }

    return batch_config

## Function to initialize the batch config
def init_batch_cfg():
    '''
    Generate Config
    '''
    batch_config_options = {
        "run_path": USER_run_path,
        'batchLabel': os.path.basename(USER_run_path),
        #'batchLabel': USER_run_label,
    }

    #Generate batch config from batch_config_options
    batch_config = get_batch_config(batch_config_options = batch_config_options)
    batch_run_path =  batch_config['saveFolder']

    # # Save batch_config in run_path as JSON    
    with open(f'{batch_run_path}/batch_config.json', 'w') as f:
        json.dump(batch_config, f, cls = batchcfgEncoder, indent =4)

    #Validate batch_config before running
    assert 'method' in batch_config, 'method must be specified in batch_config'
    assert 'skip' in batch_config['runCfg'], 'skip must be specified in batch_config'
    assert 'batchLabel' in batch_config, 'batchLabel must be specified in batch_config'

    return batch_config

## Function to convert parameter space to ranges
def rangify_params(params):
    for key, value in params.items():
        if isinstance(value, list):
            if len(value) == 2:
                params[key] = [min(value), max(value)]
            # else:
            #     params[key] = [min(value), max(value), value[2]]
        elif isinstance(value, int) or isinstance(value, float):
            params[key] = [value, value]
            
    assert all(isinstance(value, list) for value in params.values()), 'All values in params must be lists'
    return params

def build_run_paths(output_folder_name, fitness_target_script, outside_of_repo = False):
    workspace_path = setup_environment.get_git_root()
    fitness_target_script = os.path.abspath(fitness_target_script)
    #step out of workspace_path to avoid writing to the repository
    if outside_of_repo:
        workspace_path = os.path.dirname(workspace_path)
    output_folder_path = os.path.join(workspace_path, output_folder_name) # Output folder path for all runs
    output_folder_path = os.path.abspath(output_folder_path)
    
    return output_folder_path, fitness_target_script

'''Run Batch'''
def batchRun(batch_config=None):
    global params #make it global so it can be accessed by the cfg.py file easily
    
    '''Main function to run the batch'''
    from modules.simulation_config import evolutionary_parameter_space
    assert batch_config is not None, 'batch_config must be specified'  # Ensure batch_config is provided    
    params = evolutionary_parameter_space.params # Get parameter space from user-defined file
    params = rangify_params(params) # Convert parameter space to ranges
    batch = Batch(params=params) # Create Batch object with parameters to modify

    # Set attributes from batch_config to batch object
    for key, value in batch_config.items():
        if hasattr(batch, key):
            setattr(batch, key, value)

    # Prepare method-specific parameters
    batch.runCfg = batch_config['runCfg']
    if 'evolCfg' in batch_config:
        batch.evolCfg = batch_config['evolCfg']

    # Run the batch
    batch.run()

'''Main Code'''
if __name__ == '__main__':    
    
    '''Initialize'''
    batch_type = 'evol'
    output_folder_name = 'zRBS_network_simulation_outputs'
    run_label = f'development_runs' # subfolder for this run will be created in output_folder_path
    fitness_target_script = 'tunning_scripts/CDKL5-E6D_T2_C1_DIV21/derived_fitness_args/fitness_args_20241123-000932.py'
    
    # Build output folder and fitness target script paths
    output_folder_path, fitness_target_script = build_run_paths(
        output_folder_name, 
        fitness_target_script, 
        outside_of_repo = True
        ) # Build output folder path

    kwargs = {
        'duration': 0.5,
        'pop_size': 4,
        'num_elites': 1,
        'max_generations': 10,
        'continue_run': False,
        'overwrite': True,
        'maxiter_wait': 100,
        'time_sleep': 10,
        # 'num_excite': 100,
        # 'num_inhib': 46,
        'batch_type': batch_type,
        'label': run_label,
        'output_path': output_folder_path,
        'fitness_target_script': fitness_target_script,
        'mpi_type': 'mpi_bulletin', # local
        #'mpi_type': 'mpi_direct', # HPC (perlmutter)
        'debugging_in_login_node': True,
    }
    parse_kwargs.main(**kwargs) # Parse user arguments
    from temp_user_args import * # Import user arguments from temp file created by parse_kwargs.main(**kwargs)

    # Run batch
    print('Initializing batch config...')    
    batch_config = init_batch_cfg()
    
    '''Run batch'''
    print(f'Running batch: {batch_config["batchLabel"]}')
    batchRun(batch_config = batch_config)    
    print(f'Batch run completed')