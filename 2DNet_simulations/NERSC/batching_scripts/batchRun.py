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
#sys.path.insert(0, '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/netpyne')
from netpyne import specs
from netpyne.batch import Batch

## Fitness Function Imports
from fitness_functions import *

## Logger
import logging
#set up logging
logging.basicConfig(filename='logfile.log', level=logging.INFO)
# Define a logger
logger = logging.getLogger(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
logging.info(f'Batch script path: {script_dir}')

## Output path
output_path = f'{script_dir}/output/' 
# Check if the output directory exists, if not, create it
if not os.path.exists(output_path):
    os.makedirs(output_path)
logging.info(f'Output path: {output_path}')

## Get the batch configuration
def get_batch_config(batch_config_options = None):
    #from fitnessFunc_config import fitnessFunc, fitnessFuncArgs
    
    if batch_config_options is None:
        batch_config_options = {
            "run_path": os.getcwd(),
            'batchLabel': "unnamed_run",       
            "method": "evol",
            "core_num": 1,
            "nodes": 1,
           # "pop_per_core": 1,
           # "duration_seconds": 5,
            "pop_size": 4,
            "max_generations": 4,
            "time_sleep": 5,
            "maxiter_wait": 40,
            "skip": True,
            "num_elites": 1, 
        }
    else:
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
    
    # Extract the parameters
    pop_size = batch_config_options['pop_size']
    max_generations = batch_config_options['max_generations']
    method = batch_config_options['method']
    run_path =  batch_config_options['run_path']
    core_num = batch_config_options['core_num']
    nodes = batch_config_options['nodes']
    batch_label = batch_config_options['batchLabel']
    time_sleep = batch_config_options['time_sleep']
    maxiter_wait = batch_config_options['maxiter_wait']
    skip = batch_config_options['skip']
    num_elites = batch_config_options['num_elites']
    initCfg = None
    if 'initCfg' in batch_config_options:
        initCfg = batch_config_options['initCfg']

    assert isinstance(pop_size, int), 'pop_size must be an integer'
    assert isinstance(max_generations, int), 'max_generations must be an integer'
    assert method in ['evol', 'grid'], 'method must be either "evol" or "grid"'

    # Save the dictionary as a JSON file
    #method = 'evol'

    batch_config = {
        'batchLabel': batch_label,
        'saveFolder': run_path,
        'method': method,
        'runCfg': {
            'type': 'mpi_bulletin',
            'script': 'init.py',
            'mpiCommand': 'mpirun',
            'nodes': nodes,
            'coresPerNode': core_num,
            'allocation': 'default',
            'reservation': None,
            'skip': skip,
        },
        'evolCfg': {
            'evolAlgorithm': 'custom',
            'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
            'fitnessFuncArgs': {**fitnessFuncArgs, 'pop_size': pop_size},
            'pop_size': pop_size,
            'num_elites': num_elites,
            'mutation_rate': 0.4,
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


def batchRun(batchLabel = 'batchRun', method = 'evol', skip = True, batch_config = None):
    
    #Get batch_config as needed         
    if batch_config is None:
        with open('batch_config_options.json') as f:
            batch_config_options = json.load(f)
        batch_config = get_batch_config(batch_config_options = batch_config_options)
        assert 'method' in batch_config, 'method must be specified in batch_config'
        assert 'skip' in batch_config['runCfg'], 'skip must be specified in batch_config'
        assert 'batchLabel' in batch_config, 'batchLabel must be specified in batch_config'
            
    #extract_batch params
    batch_run_path =  batch_config['saveFolder']
    # Save the dictionary as a pickle file
    with open(f'{batch_run_path}/batch_config.pickle', 'wb') as f:
        pickle.dump(batch_config, f)

    #load param_space from batch_run_path json
    assert os.path.exists(f'{batch_run_path}/param_space.pickle'), 'params.json does not exist in run_path'
    with open(f'{batch_run_path}/param_space.pickle', 'rb') as f:
        params = pickle.load(f)   

	# create Batch object with paramaters to modify, and specifying files to use
    batch = Batch(params=params)
    batch.method = batch_config['method']
    batch.batchLabel = batch_config['batchLabel']
    batch.saveFolder = batch_config['saveFolder'] # = batch_run_path

    #prepare run and evol configuration
    batch.runCfg = batch_config['runCfg']
    batch.evolCfg = batch_config['evolCfg']    

    #run batch
    batch.run()

# Main code
if __name__ == '__main__':
    
    ## load batch_config_options.json
    # with open('batch_config_options.json') as f:
    #     batch_config_options = json.load(f)
    from batch_run_config import *
    batch_config = get_batch_config(batch_config_options = batch_config_options)
    assert 'method' in batch_config, 'method must be specified in batch_config'
    assert 'skip' in batch_config['runCfg'], 'skip must be specified in batch_config'
    assert 'batchLabel' in batch_config, 'batchLabel must be specified in batch_config'
    
    #run batch
    batchRun(
        batchLabel = batch_config['batchLabel'], 
        method = batch_config['method'],
        skip = batch_config['runCfg']['skip'],
        ) 
