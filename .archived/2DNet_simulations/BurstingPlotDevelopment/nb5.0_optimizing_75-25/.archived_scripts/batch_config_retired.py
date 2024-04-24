import json
from fitnessFunc_config import fitnessFunc, fitnessFuncArgs
import os
import pickle

def get_batch_config(method, nodes, pop_size, max_generations, run_path, core_num, time_sleep, maxiter_wait, skip = True):
    assert isinstance(pop_size, int), 'pop_size must be an integer'
    assert isinstance(max_generations, int), 'max_generations must be an integer'
    assert method in ['evol', 'grid'], 'method must be either "evol" or "grid"'

    # Save the dictionary as a JSON file
    #method = 'evol'

    batch_config = {
        'batchLabel': os.path.basename(run_path),
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
            'fitnessFuncArgs': {**fitnessFuncArgs, 'simLabel': 'batch.cfg.simLabel'},
            'pop_size': pop_size,
            'num_elites': 1,
            'mutation_rate': 0.4,
            'crossover': 0.5,
            'maximize': False,
            'max_generations': max_generations,
            'time_sleep': time_sleep, # wait this time before checking again if sim is completed (for each generation)
            'maxiter_wait': maxiter_wait, # max number of times to check if sim is completed (for each generation)
            #effectively 1.5 hours per gen, max
            'defaultFitness': 1000
        }
    }

    # Save the dictionary as a pickle file
    with open('batch_config.pickle', 'wb') as f:
        pickle.dump(batch_config, f)

    return batch_config

## Notes for potential future issues
# OG batch_config that was working:

    # batch.runCfg = {
    #     'type': 'mpi_bulletin',#'hpc_slurm',
    #     'script': 'init.py',
    #     # options required only for hpc
    #     'mpiCommand': 'mpirun',
    #     'nodes': 4,
    #     'coresPerNode': 2,
    #     'allocation': 'default',
    #     #'email': 'salvadordura@gmail.com',
    #     'reservation': None,
    #     'skip': skip,
    #     #'folder': '/home/salvadord/evol'
    #     #'custom': 'export LD_LIBRARY_PATH="$HOME/.openmpi/lib"' # only for conda users
    # }
    # batch.evolCfg = {
    # 	'evolAlgorithm': 'custom',
    # 	'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
    # 	#'fitnessFuncArgs': fitnessFuncArgs,
    #     'fitnessFuncArgs': {**fitnessFuncArgs, 'simLabel': 'batch.cfg.simLabel'},
    # 	'pop_size': batch_config['evolCfg']['pop_size'], # population size
    # 	'num_elites': 1, # keep this number of parents for next generation if they are fitter than children
    # 	'mutation_rate': 0.4,
    # 	'crossover': 0.5,
    # 	'maximize': False, # maximize fitness function?
    # 	'max_generations': batch_config['evolCfg']['max_generations'], # how many generations to run
    # 	'time_sleep': 30, # wait this time before checking again if sim is completed (for each generation)
    # 	#'maxiter_wait': 4000000000, # max number of times to check if sim is completed (for each generation)
    #     'maxiter_wait': 300000, # max number of times to check if sim is completed (for each generation)
    # 	'defaultFitness': 1000, # set fitness value in case simulation time is over
    #     #'gen': 'gen',
    #     #'cand': 'cand',
    # }