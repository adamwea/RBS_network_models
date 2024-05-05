''' 
This file is a *highly* modified version of the file batchRun.py 
from the NetPyNE tutorial 9. 

Evolutionary algorithm optimization of a network using NetPyNE

To run use: mpiexec -np [num_cores] nrniv -mpi batchRun.py
'''

'''
Initialize
'''
##General Imports
import os
import shutil
import json
import pickle
import glob
import sys
import json
import os
import pickle
import time
import datetime
import pandas as pd
import sys

## NetPyne Imports
#sys.path.insert(0, 'netpyne')
from netpyne import specs
from netpyne.batch import Batch
from fitness_functions import *
from fitness_config import *
from USER_INPUTS import *
from evol_param_setup import evol_param_space
#from batch_config_setup import *

## Logger
import logging
#set up logging
script_dir = os.path.dirname(os.path.realpath(__file__))
log_file = f'{script_dir}/batchRun.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
# Define a logger
logger = logging.getLogger(__name__)

'''functions'''
## Function to serialize the batch_config dictionary
class batchcfgEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj):
            return str(obj)
        return super().default(obj)
def get_HOF_seeds():
    #seeded_HOF_cands = ['.'+'/NERSC/output/240430_Run2_debug_node_run/gen_5/gen_5_cand_39_cfg.json']
    #import list of paths from .csv USER_HOF
    assert os.path.exists(USER_HOF), f'USER_HOF file not found: {USER_HOF}'
    #print(f'Loading Hall of Fame from {USER_HOF}')

    seeded_HOF_cands = pd.read_csv(USER_HOF).values.flatten()
    seeded_HOF_cands = [cfg.replace('_data', '_cfg') for cfg in seeded_HOF_cands]
    seeded_HOF_cands = [os.path.abspath(f'./{cfg}') for cfg in seeded_HOF_cands]
    #compensate for running in NERSC dir...this is bad code...fix later
    #seeded_HOF_cands = [cfg.replace('NERSC/NERSC', 'NERSC') for cfg in seeded_HOF_cands if 'NERSC/NERSC' in cfg]
    for cfg in seeded_HOF_cands:
        if 'NERSC/NERSC' in cfg: 
            seeded_HOF_cands[seeded_HOF_cands.index(cfg)] = cfg.replace('NERSC/NERSC', 'NERSC')
        else: continue

    #print(f'Loaded {len(seeded_HOF_cands)} Hall of Fame candidates')

    #get abs path
    seeds = []
    #i=0
    for cfg in seeded_HOF_cands:
        #cfg = os.path.abspath(cfg)
        #print(f'Loading Hall of Fame candidate: {cfg}')
        if not os.path.exists(cfg): cfg = None #check if file exists, else set to None
        #print(f'Loading Hall of Fame candidate: {cfg}')
        #selected_cand_cfg = None        
        if cfg is not None:    
            with open(cfg, 'r') as f:
                seed = json.load(f)
            seed = seed['simConfig']
            #only keep overlap with USER_evelo_param_space.py
            seed = {
                key: evol_param_space[key][0] if evol_param_space[key][0] == evol_param_space[key][1] 
                else seed[key] for key in evol_param_space if key in seed
            }
            #get rid of keys, just make list of values
            seed = list(seed.values())
            #make sure all values are floats
            seed = [float(val) for val in seed]
            seeds.append(seed)
        else: continue
        #seed = [] #set to empty dict for compatibility with NetPyNE Batch
        #sys.exit()
    print(f'Loaded {len(seeds)} seeds from Hall of Fame')
    #sys.exit()
    assert len(seeds) > 0, 'No seeds loaded from Hall of Fame'
    return seeds
def get_batch_config(batch_config_options = None):
    
    USER_seed_evol = True
    if USER_seed_evol == True:
        HOF_seeds = get_HOF_seeds()
        #load HOF of previous runs
    else: HOF_seeds = {}
    
    # Extract the parameters
    run_path =  batch_config_options['run_path']
    batch_label = batch_config_options['batchLabel']
    # if 'initCfg' in batch_config_options:
    #     initCfg = batch_config_options['initCfg']
    # else: initCfg = {}

    batch_config = {
        'batchLabel': batch_label,
        'saveFolder': run_path,
        'method': USER_method,
        'cfgFile': USER_cfgFile,
        'netParamsFile': USER_netParamsFile,
        #'initCfg': initCfg,
        'runCfg': {
            'type': USER_runCfg_type,
            'script': USER_init_script,
            'mpiCommand': USER_mpiCommand,
            'nodes': USER_nodes,
            'coresPerNode': USER_cores_per_node,
            'allocation': USER_allocation,
            'reservation': None,
            'skip': USER_skip,
            'email': USER_email,
            'walltime': USER_walltime,
            'custom': USER_custom_slurm,
            #'initCfg': initCfg,
        },
        'evolCfg': {
            'evolAlgorithm': 'custom',
            'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
            'fitnessFuncArgs': {**fitnessFuncArgs, 'pop_size': USER_pop_size},
            'pop_size': USER_pop_size,
            'num_elites': USER_num_elites,
            'mutation_rate': USER_mutation_rate, # using high mutation rate to explore more of the parameter space
            'crossover': USER_crossover,
            'maximize': False,
            'max_generations': USER_max_generations,
            'time_sleep': USER_time_sleep, # wait this time before checking again if sim is completed (for each generation)
            'maxiter_wait': USER_maxiter_wait, # max number of times to check if sim is completed (for each generation)
            'defaultFitness': 1000,
            'seeds': HOF_seeds,
            #'initCfg': initCfg,
        }
    }

    return batch_config
def batchRun(batch_config = None):
    
    #Get batch_config
    assert batch_config is not None, 'batch_config must be specified'

    #load param_space from batch_run_path json
    #batch_run_path = batch_config['saveFolder']
    batch_method = batch_config['method']
    if batch_method == 'evol':
        #define population params held constant across batch of simulations
        params = evol_param_space
        # define_population_params(batch_run_path = batch_run_path)
    elif batch_method == 'grid':
        logger.error('Grid method not yet implemented')
    else:
        logger.error('Invalid method specified in batch_config')

	# create Batch object with paramaters to modify, and specifying files to use
    #batch = Batch(initCfg=batch_config['initCfg'], params=params)
    bool = os.path.exists(batch_config['cfgFile'])
    bool2 = os.path.exists(batch_config['netParamsFile'])
    batch = Batch(
        #cfgFile=batch_config['cfgFile'],
        #netParamsFile=batch_config['netParamsFile'],
        #cfg=None,
        #netParams=None,
        params=params,
        #groupedParams=None,
        #initCfg=batch_config['initCfg'],
        #seed=None,
        )
    
    # Iterate over items in batch_config
    for key, value in batch_config.items():
        # If batch has an attribute with the same name as the key
        if hasattr(batch, key):
            # Set the attribute to the value from batch_config
            setattr(batch, key, value)

    ## Prepare method specific parameters
    batch.runCfg = batch_config['runCfg']
    if 'evolCfg' in batch_config: batch.evolCfg = batch_config['evolCfg']    

    #run batch
    batch.run()
def init_batch_cfg():
    '''
    Initialize
    '''
    # Get the rank of the current process
    rank = os.environ.get('OMPI_COMM_WORLD_RANK')
    if rank is None: rank = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK')
    if rank is None: rank = os.environ.get('OMPI_COMM_WORLD_NODE_RANK')
    # Check if the environment variable is set
    if rank is not None:
        rank = int(rank)  # Convert the rank to an integer
        logger.info(f"I am process: {rank}")
    else:
        logger.info("OMPI_COMM_WORLD_RANK is not set")
        rank = 0

    # Get current date in YYMMDD format
    current_date = datetime.datetime.now().strftime('%y%m%d')
    # Prepare Batch_Run_Folder and Initial Files
    script_path = os.path.dirname(os.path.realpath(__file__))
    output_path = script_path+'/output'

    # Get list of existing runs for the day
    try: existing_runs = [run for run in os.listdir(output_path) if run.startswith(current_date)]
    except: existing_runs = []

    # Find the highest run number for the day
    if existing_runs: highest_run_number = max(int(run.split('_Run')[1].split('_')[0]) for run in existing_runs)
    else: highest_run_number = 0

    # Increment the run number for the new run
    new_run_number = highest_run_number + 1
    prev_run_number = new_run_number - 1

    # Update run_name with new format
    run_name = f'{current_date}_Run{new_run_number}_{USER_run_label}'
    prev_run_name = f'{current_date}_Run{prev_run_number}_{USER_run_label}'
    #print(f'Run Name: {run_name}')
    #sys.exit()   

    # Get unique run path
    run_path = f'{output_path}/{run_name}'
    prev_run_path = f'{output_path}/{prev_run_name}'

    if USER_overwrite or USER_continue:
        if prev_run_name in existing_runs:
            assert not (USER_overwrite and USER_continue), 'overwrite_run and continue_run cannot both be True'
            if USER_overwrite and os.path.exists(prev_run_path):
                shutil.rmtree(prev_run_path)
                run_path = prev_run_path   
                logger.info(f'Overwriting existing batch_run: {os.path.basename(run_path)}')
            elif USER_continue and os.path.exists(prev_run_path):
                run_path = prev_run_path
                logger.info(f'Continuing existing batch_run: {os.path.basename(run_path)}')

    if not os.path.exists(run_path) and rank == 0:
        logger.info(f'Creating new batch_run: {os.path.basename(run_path)}')
        os.makedirs(run_path)
    else:
        # Wait for the directory to be created by rank 0
        if not os.path.exists(run_path):
            logger.info(f'Rank {rank} waiting for run_path to be created: {run_path}')
            while not os.path.exists(run_path):
                time.sleep(1)
    #sys.exit()

    '''
    Generate Config
    '''

    ##Create a dictionary with the given variables and their values
    batch_config_options = {
        "run_path": run_path,
        'batchLabel': os.path.basename(run_path),
        #"initCfg": initCfg,
    }
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    logging.info(f'Batch script path: {script_dir}')

    ## Output path
    output_path = f'{script_dir}/output/' 
    print(f'Output path: {output_path}')
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.info(f'Output path: {output_path}')

    ## Load batch_config_options.json    
    #logging.info(f'Batch config options: {batch_config_options}')
    batch_run_path = run_path
    # Save batch_config in run_path as JSON    
    # with open(f'{batch_run_path}/batch_config_options.json', 'w') as f:
    #     json.dump(batch_config_options, f, cls = batchcfgEncoder, indent = 4)
    # logging.info(f'Batch config options saved in {batch_run_path}/batch_config_options.json')

    #Generate batch config from batch_config_options
    batch_config = get_batch_config(batch_config_options = batch_config_options)
    logging.info(f'Batch config: {batch_config}')
    #extract_batch params
    batch_run_path =  batch_config['saveFolder']
    # Save batch_config in run_path as JSON    
    with open(f'{batch_run_path}/batch_config.json', 'w') as f:
        json.dump(batch_config, f, cls = batchcfgEncoder, indent =4)
    logging.info(f'Batch config saved in {batch_run_path}/batch_config.json')

    #Validate batch_config before running
    assert 'method' in batch_config, 'method must be specified in batch_config'
    assert 'skip' in batch_config['runCfg'], 'skip must be specified in batch_config'
    assert 'batchLabel' in batch_config, 'batchLabel must be specified in batch_config'

    return batch_config
def main():
    logging.info(f'Batch run script started')
    # logging.info(f'Timer Started: {datetime.datetime.now()}')

    # # Start timers
    # start_time_wall = datetime.datetime.now()
    # start_time_cpu = time.process_time()

    # Run batch
    run_batch = True
    if run_batch:    
        logging.info(f'Initializing batch config')        
        batch_config = init_batch_cfg()
        logging.info(f'Running batch: {batch_config["batchLabel"]}')
        batchRun(batch_config = batch_config)
        logging.info(f'Batch run completed')

    # End timers
    # end_time_wall = datetime.datetime.now()
    # end_time_cpu = time.process_time()

    # logging.info(f'Timer Ended: {end_time_wall}')
    # logging.info(f'Total CPU Time: {end_time_cpu - start_time_cpu}')
    # logging.info(f'Total Time Elapsed: {end_time_wall - start_time_wall}')
    # logging.info(f'Estimated Wait Time: {end_time_wall - start_time_wall - datetime.timedelta(seconds=end_time_cpu - start_time_cpu)}')

'''Main code'''
if __name__ == '__main__':
    main()