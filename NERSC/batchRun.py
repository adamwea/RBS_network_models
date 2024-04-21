''' 
This file is a *highly* modified version of the file batchRun.py 
from the NetPyNE tutorial 9. 

Evolutionary algorithm optimization of a network using NetPyNE
To run use: mpiexec -np [num_cores] nrniv -mpi batchRun.py
Note: May or may not include capability for other options in here later.
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

## NetPyne Imports
#sys.path.insert(0, 'netpyne')
from netpyne import specs
from netpyne.batch import Batch

## Fitness Function Imports
from fitness_functions import *
from fitness_config import *

## Logger
import logging
#set up logging
script_dir = os.path.dirname(os.path.realpath(__file__))
log_file = f'{script_dir}/batchRun.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
# Define a logger
logger = logging.getLogger(__name__)

## Get the batch configuration
def get_batch_config(batch_config_options = None):

    ## Validate batch_config_options    
    assert not batch_config_options is None, "batch_config_options must be defined"
    assert 'method' in batch_config_options, 'method must be specified in batch_config_options'
    # assert 'core_num' in batch_config_options, 'core_num must be specified in batch_config_options'
    # assert 'nodes' in batch_config_options, 'nodes must be specified in batch_config_options'
    # assert 'pop_per_core' in batch_config_options, 'pop_per_core must be specified in batch_config_options'
    # assert 'duration_seconds' in batch_config_options, 'duration_seconds must be specified in batch_config_options'
    assert 'pop_size' in batch_config_options, 'pop_size must be specified in batch_config_options'
    assert 'max_generations' in batch_config_options, 'max_generations must be specified in batch_config_options'
    assert 'num_elites' in batch_config_options, 'num_elites must be specified in batch_config_options'
    # assert 'time_sleep' in batch_config_options, 'time_sleep must be specified in batch_config_options'
    # assert 'maxiter_wait' in batch_config_options, 'maxiter_wait must be specified in batch_config_options'
    # assert 'skip' in batch_config_options, 'skip must be specified in batch_config_options'
    assert 'cfgFile' in batch_config_options, 'cfgFile must be specified in batch_config_options'
    assert 'netParamsFile' in batch_config_options, 'netParamsFile must be specified in batch_config_options'
    
    # Extract the parameters
    pop_size = batch_config_options['pop_size']
    max_generations = batch_config_options['max_generations']
    method = batch_config_options['method']
    run_path =  batch_config_options['run_path']
    cores_per_node = batch_config_options['cores_per_node']
    nodes = batch_config_options['nodes']
    batch_label = batch_config_options['batchLabel']
    time_sleep = batch_config_options['time_sleep']
    maxiter_wait = batch_config_options['maxiter_wait']
    skip = batch_config_options['skip']
    num_elites = batch_config_options['num_elites']
    script = batch_config_options['script']
    cfgFile = batch_config_options['cfgFile']
    netParamsFile = batch_config_options['netParamsFile']
    initCfg = None
    if 'initCfg' in batch_config_options:
        initCfg = batch_config_options['initCfg']

    assert isinstance(pop_size, int), 'pop_size must be an integer'
    assert isinstance(max_generations, int), 'max_generations must be an integer'
    assert method in ['evol', 'grid'], 'method must be either "evol" or "grid"'

    # Save the dictionary as a JSON file
    #method = 'evol'

    from USER_INPUTS import USER_runCfg_type
    from USER_INPUTS import USER_walltime
    from USER_INPUTS import USER_email
    from USER_INPUTS import USER_custom_slurm
    from USER_INPUTS import USER_allocation
    from USER_INPUTS import USER_MPI_processes_per_node
    from USER_INPUTS import USER_nodes

    batch_config = {
        'batchLabel': batch_label,
        'saveFolder': run_path,
        'method': method,
        'cfgFile': cfgFile,
        'netParamsFile': netParamsFile,
        'runCfg': {
            'type': USER_runCfg_type,
            'script': script,
            'mpiCommand': USER_mpiCommand,
            'nodes': USER_nodes,
            'coresPerNode': USER_MPI_processes_per_node,
            'allocation': USER_allocation,
            'reservation': None,
            'skip': skip,
            'email': USER_email,
            'walltime': USER_walltime,
            'custom': USER_custom_slurm,
        },
        'evolCfg': {
            'evolAlgorithm': 'custom',
            'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
            'fitnessFuncArgs': {**fitnessFuncArgs, 'pop_size': pop_size},
            'pop_size': pop_size,
            'num_elites': num_elites,
            #'mutation_rate': 0.4,
            'mutation_rate': 0.7, # using high mutation rate to explore more of the parameter space
            'crossover': 0.5,
            'maximize': False,
            'max_generations': max_generations,
            'time_sleep': time_sleep, # wait this time before checking again if sim is completed (for each generation)
            'maxiter_wait': maxiter_wait, # max number of times to check if sim is completed (for each generation)
            #effectively 1.5 hours per gen, max
            'defaultFitness': 1000,
            'initCfg': initCfg,
        }
    }  
    
    # # Save the dictionary as a pickle file
    # with open('batch_config.pickle', 'wb') as f:
    #     pickle.dump(batch_config, f)

    return batch_config


## Function to serialize the batch_config dictionary
class batchcfgEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj):
            return str(obj)
        return super().default(obj)

def batchRun(batch_config = None):
    
    #Get batch_config
    assert batch_config is not None, 'batch_config must be specified'

    #load param_space from batch_run_path json
    batch_run_path = batch_config['saveFolder']
    batch_method = batch_config['method']
    if batch_method == 'evol':
        #define population params held constant across batch of simulations
        from evol_param_setup import evol_param_space
        params = evol_param_space
        # define_population_params(batch_run_path = batch_run_path)
    elif batch_method == 'grid':
        logger.error('Grid method not yet implemented')
    else:
        logger.error('Invalid method specified in batch_config')

	# create Batch object with paramaters to modify, and specifying files to use
    batch = Batch(params=params)
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
    batch_run_path = batch_config_options['run_path']
    # Save batch_config in run_path as JSON    
    with open(f'{batch_run_path}/batch_config_options.json', 'w') as f:
        json.dump(batch_config_options, f, cls = batchcfgEncoder, indent = 4)
    logging.info(f'Batch config options saved in {batch_run_path}/batch_config_options.json')

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

# Main code
import time
import datetime
if __name__ == '__main__':
    
    logging.info(f'Batch run script started')
    logging.info(f'Timer Started: {datetime.datetime.now()}')

    # Start timers
    start_time_wall = datetime.datetime.now()
    start_time_cpu = time.process_time()

    # Run batch
    run_batch = True
    if run_batch:    
        logging.info(f'Initializing batch config')
        from batch_config_setup import *
        batch_config = init_batch_cfg()

        logging.info(f'Running batch: {batch_config["batchLabel"]}')
        batchRun(batch_config = batch_config)
        logging.info(f'Batch run completed')

    # End timers
    end_time_wall = datetime.datetime.now()
    end_time_cpu = time.process_time()

    logging.info(f'Timer Ended: {end_time_wall}')
    logging.info(f'Total CPU Time: {end_time_cpu - start_time_cpu}')
    logging.info(f'Total Time Elapsed: {end_time_wall - start_time_wall}')
    logging.info(f'Estimated Wait Time: {end_time_wall - start_time_wall - datetime.timedelta(seconds=end_time_cpu - start_time_cpu)}')
