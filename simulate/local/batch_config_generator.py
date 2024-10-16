import os
import json
from hof_seed_loader import get_HOF_seeds
from simulate._temp_files.temp_user_args import *
from simulation_config_files import fitnessFuncArgs


class BatchCfgEncoder(json.JSONEncoder):
    '''Custom encoder for the batch configuration.'''
    def default(self, obj):
        if callable(obj):
            return str(obj)
        return super().default(obj)

def get_batch_config(batch_config_options=None):
    '''Generates the batch configuration.'''
    if USER_seed_evol:
        HOF_seeds = get_HOF_seeds()
    else:
        HOF_seeds = None

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

def init_batch_cfg():
    '''Initializes and generates the batch configuration.'''
    run_path = USER_run_path
    batch_config_options = {
        "run_path": run_path,
        'batchLabel': os.path.basename(run_path),
    }
    
    batch_config = get_batch_config(batch_config_options=batch_config_options)
    batch_run_path = batch_config['saveFolder']

    with open(f'{batch_run_path}/batch_config.json', 'w') as f:
        json.dump(batch_config, f, cls=BatchCfgEncoder, indent=4)

    return batch_config
