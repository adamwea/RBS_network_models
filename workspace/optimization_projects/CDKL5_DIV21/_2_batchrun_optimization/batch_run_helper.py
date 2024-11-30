'''batchRun helper functions'''

'''reimplemented'''

def setup_environment_wrapper(verbose = False):
    from pprint import pprint
    import workspace.RBS_network_simulations.workspace.optimization_projects.CDKL5_DIV21.setup_environment as setup_environment
    setup_environment.set_pythonpath()
    import sys
    
    if verbose:
        #print sys path
        pprint(sys.path)
    #sys.exit(0)
    return sys.path

def add_output_path_to_kwargs(output_folder_name, fitness_target_script, outside_of_repo = False, **kwargs):
    import workspace.RBS_network_simulations.workspace.optimization_projects.CDKL5_DIV21.setup_environment as setup_environment
    
    workspace_path = setup_environment.get_git_root()
    fitness_target_script = os.path.abspath(fitness_target_script)
    #step out of workspace_path to avoid writing to the repository
    if outside_of_repo:
        workspace_path = os.path.dirname(workspace_path)
    output_folder_path = os.path.join(workspace_path, output_folder_name) # Output folder path for all runs
    output_folder_path = os.path.abspath(output_folder_path)
    
    kwargs['output_path'] = output_folder_path
    kwargs['fitness_target_script'] = fitness_target_script
    #kwargs['fitness_target_script'] = fitness_target_script
    
    return kwargs

def check_for_existing_runs(output_path, current_date):
    '''Check for existing runs in the output path'''
    try:
        existing_runs = [run for run in os.listdir(output_path) if run.startswith(current_date)]
    except:
        existing_runs = []
        
    # Find the highest run number for the day
    if existing_runs:
        highest_run_number = max(int(run.split('_Run')[1].split('_')[0]) for run in existing_runs)
    else:
        highest_run_number = 0
    return existing_runs, highest_run_number

def add_run_path_to_kwargs(**kwargs):
    '''Build the output run path for the current run'''
    import datetime
    
    '''init'''
    rmtree = False
    mkrundir = False
    
    #extract current kwargs
    #output_path = kwargs.get('output_path', None)
    overwrite_run = kwargs.get('overwrite_run', False)
    continue_run = kwargs.get('continue_run', False)
    label = kwargs.get('label', 'default')
    output_path = kwargs.get('output_path', None)
    assert output_path is not None, 'output_path must be specified'
    
    #initialize current_date
    current_date = datetime.datetime.now().strftime('%y%m%d')     # Get current date in YYMMDD format
    existing_runs, highest_run_number = check_for_existing_runs(output_path, current_date)     # Get list of existing runs for the day

    # Increment the run number for the new run
    new_run_number = highest_run_number + 1
    prev_run_number = new_run_number - 1

    # Update run_name with the new format
    run_name = f'{current_date}_Run{new_run_number}_{label}'
    prev_run_name = f'{current_date}_Run{prev_run_number}_{label}'

    # Get unique run path
    run_path = os.path.join(output_path, run_name)
    prev_run_path = os.path.join(output_path, prev_run_name)

    # Check if run exists and if it should be overwritten or continued
    if overwrite_run or continue_run:
        if prev_run_name in existing_runs:
            if overwrite_run and os.path.exists(prev_run_path):
                #print(f'Overwriting previous run at: {prev_run_path}')
                # print(f'Deleting previous run at: {prev_run_path}')
                #shutil.rmtree(prev_run_path)
                rmtree = True
                #assert that prev_run_path has been deleted
                #assert not os.path.exists(prev_run_path), f'Previous run at {prev_run_path} was not deleted'
                run_path = prev_run_path
            elif continue_run and os.path.exists(prev_run_path):
                #print(f'Continuing previous run at: {prev_run_path}')
                run_path = prev_run_path
    elif os.path.exists(prev_run_path) and os.path.isdir(prev_run_path) and not os.listdir(prev_run_path):
        rmtree = True #delete empty directory even if not overwriting, avoid creating new directory unnecessarily
        run_path = prev_run_path

    # Create the new run directory if it does not exist
    if not os.path.exists(run_path):
        #print(f'Creating new run directory at: {run_path}')
        #os.makedirs(run_path)
        mkrundir = True
        
    #update kwargs
    kwargs['run_path'] = run_path
    kwargs['rmtree'] = rmtree
    kwargs['mkrundir'] = mkrundir
    kwargs['prev_run_path'] = prev_run_path

    return kwargs

def init_output_paths_as_needed(rmtree, mkrundir, **kwargs):
    '''Initialize output paths as needed'''
    import shutil
    import time
    
    #extract current kwargs
    output_path = kwargs.get('output_path', None)
    prev_run_path = kwargs.get('prev_run_path', None)
    run_path = kwargs.get('run_path', None)
    
    #remove previous run if rmtree is True
    if rmtree:
        print(f'Overwriting previous run at: {prev_run_path}')
        print(f'Deleting previous run at: {prev_run_path}')
        shutil.rmtree(prev_run_path)
        #delay for a few seconds to allow the system to catch up
        time.sleep(2)
        assert not os.path.exists(prev_run_path), f'Previous run at {prev_run_path} was not deleted'
        
    if mkrundir:
        print(f'Creating new run directory at: {run_path}')
        os.makedirs(run_path, exist_ok=True)
        time.sleep(2)
        assert os.path.exists(run_path), f'New run directory at {output_path} was not created'
    
    return

## Function to serialize the batch_config dictionary
import json
class batchcfgEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj):
            return str(obj)
        return super().default(obj)

# Define the function to import the module from a file path
def import_module_from_path(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

## Function to initialize batch config
def init_batch_cfg(USER_vars, **kwargs):
    '''
    Generate Config
    '''
    
    USER_run_path = USER_vars.get('USER_run_path', None)
    USER_fitness_target_script = USER_vars.get('USER_fitness_target_script', None)
    USER_seed_evol = USER_vars.get('USER_seed_evol', False)
    USER_method = USER_vars.get('USER_method', None)
    USER_cfg_file = USER_vars.get('USER_cfg_file', None)
    USER_netParamsFile = USER_vars.get('USER_netParamsFile', None)
    USER_runCfg_type = USER_vars.get('USER_runCfg_type', None)
    USER_init_script = USER_vars.get('USER_init_script', None)
    USER_mpiCommand = USER_vars.get('USER_mpiCommand', None)
    USER_nrnCommand = USER_vars.get('USER_nrnCommand', None)
    USER_nodes = USER_vars.get('USER_nodes', None)
    USER_cores_per_node = USER_vars.get('USER_coresPerNode', None)
    USER_skip = USER_vars.get('USER_skip', None)
    USER_pop_size = USER_vars.get('USER_pop_size', None)
    USER_num_elites = USER_vars.get('USER_num_elites', None)
    USER_mutation_rate = USER_vars.get('USER_mutation_rate', None)
    USER_crossover = USER_vars.get('USER_crossover', None)
    USER_max_generations = USER_vars.get('USER_max_generations', None)
    USER_time_sleep = USER_vars.get('USER_time_sleep', None)
    USER_maxiter_wait = USER_vars.get('USER_maxiter_wait', None)

    import_module_from_path(USER_fitness_target_script, 'fitnessFuncArgs') # dynamically import fitnessFuncArgs from USER_fitness_target_script defined as python scripts so that we can optimize different data
    from fitnessFuncArgs import fitnessFuncArgs
    from workspace.RBS_network_simulations.workspace.optimization_projects.CDKL5_DIV21.calculate_fitness import fitnessFunc
    
    # Load HOF seeds if USER_seed_evol is True
    HOF_seeds = get_HOF_seeds() if USER_seed_evol else None

    batch_config_options = {
        "run_path": USER_run_path,
        'batchLabel': os.path.basename(USER_run_path),
    }

    batch_config = {
        'batchLabel': batch_config_options['batchLabel'],
        'saveFolder': batch_config_options['run_path'],
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

    batch_run_path = batch_config['saveFolder']

    # Save batch_config in run_path as JSON    
    with open(f'{batch_run_path}/batch_config.json', 'w') as f:
        json.dump(batch_config, f, cls=batchcfgEncoder, indent=4)

    # Validate batch_config before running
    assert 'method' in batch_config, 'method must be specified in batch_config'
    assert 'skip' in batch_config['runCfg'], 'skip must be specified in batch_config'
    assert 'batchLabel' in batch_config, 'batchLabel must be specified in batch_config'

    return batch_config

## Function to run the batch
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

def pre_run_checks(USER_vars, **kwargs):
    from pprint import pprint
    
    print("\n" + "="*50)
    print("PRE-RUN CHECKS")
    print("="*50 + "\n")
    
    def alert_user(*args, color='yellow'):
        colors = {
            'yellow': '\033[93m',
            'red': '\033[91m',
            'white': '\033[97m',
            'reset': '\033[0m'
        }
        for arg in args:
            print(f"{colors[color]}{arg}{colors['reset']}")
    
    def print_dict(d, indent=0, color='yellow'):
        for key, value in d.items():
            if isinstance(value, dict):
                alert_user(' ' * indent + f'{key}:', color=color)
                print_dict(value, indent + 4, color)
            else:
                alert_user(' ' * indent + f'{key}: {value}', color=color)
    
    #print all USER_vars in yellow, as a list
    for key, value in USER_vars.items():
        alert_user(f'{key}: {value}', color='yellow')
    print('')  # newline
    
    # Print batch config in yellow
    batch_cfg = kwargs.get('batch_cfg', None)
    if batch_cfg:
        alert_user('Batch config:', color='yellow')
        print_dict(batch_cfg)
        print('')  # newline
    #
    import os   
    output_path = kwargs.get('output_path', None)
    assert output_path is not None, 'output_folder_path must be specified'
    alert_user(f'Output path: {output_path}', color='white')
    print('') #newline    
    
    run_path = kwargs.get('run_path', None)
    assert run_path is not None, 'run_path must be specified'
    alert_user(f'Run path: {run_path}', color='white')
    print('') #newline
    
    example_command = kwargs.get('example_command', None)
    alert_user(f'Example command:\n{example_command}', color='white')
    print('') #newline
    
    run_path = kwargs.get('run_path')
    rmtree = kwargs.get('rmtree')
    mkrundir = kwargs.get('mkrundir')
    overwrite = kwargs.get('overwrite')
    continue_run = kwargs.get('continue_run')
    run_path_exists = os.path.exists(run_path)   
    alert_user(f"Flags:\n- rmtree={rmtree}\n- mkrundir={mkrundir}\n- overwrite={overwrite}\n- continue_run={continue_run}", color='white')
    print('') #newline
    
    if rmtree and overwrite:
        assert run_path_exists, 'run_path must exist to use rmtree - this is flagged incorrectly by the program. please check the code.'
        alert_user('Overwrite is set to True, and rmtree has been flagged by the program to do this properly.',
                   'This will delete the entire run_path and all of its contents.', color='red')
    if rmtree and not overwrite:
        assert run_path_exists, 'run_path must exist to use rmtree - this is flagged incorrectly by the program. please check the code.'
        alert_user('Overwrite is set to False, and rmtree has been flagged.',
                   'This is probably because the current run_path is empty and removing it is an easy way to deal with it before initiating the run.',
                   'If the run_path is not empty, this will delete the entire run_path and all of its contents.', color='yellow')
    if mkrundir and run_path_exists:
        alert_user('The run_path already exists.',
                   'This is probably because the run_path is being reused from a previous run.',
                   'If this is not the case, the run_path will be deleted and recreated.', color='yellow')
    if mkrundir and not run_path_exists:
        alert_user('The run_path does not exist.',
                   'This is probably because the run_path is being created for the first time.',
                   'If this is not the case, the run_path will be created.', color='yellow')
    if continue_run and run_path_exists:
        alert_user('Continue run is set to True and the run_path already exists.',
                   'Make sure that you mean to select these options.',
                   'Running the command will continue the run from the latest run in the output folder.', color='yellow')
    
    print("\n" + "="*50)
    print("END OF PRE-RUN CHECKS")
    print("="*50 + "\n")
# ______________________________________________________________________________________________________________________

'''not re-implemented'''
'''Setup Python environment for running the script'''
from pprint import pprint
import workspace.RBS_network_simulations.workspace.optimization_projects.CDKL5_DIV21.setup_environment as setup_environment
setup_environment.set_pythonpath()
#print sys path
import sys
#pprint(sys.path)

'''
Import Local Modules

Note: Specify settings in kwargs dictionary, they are handled by parse_user_args.main(**kwargs)
'''
from RBS_network_simulations.optimization_scripts import parse_kwargs


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









'''Run Batch'''

