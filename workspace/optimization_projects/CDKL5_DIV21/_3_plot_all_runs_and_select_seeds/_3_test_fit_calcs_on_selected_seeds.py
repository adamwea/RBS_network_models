from setup_environment import set_pythonpath
set_pythonpath()

from _2_seed_review_241126_Run2_improved_netparams import seed_paths as current_seeds
#from workspace.optimization_projects.CDKL5_DIV21._2_batchrun_optimization.calculate_fitness_v2 import fitnessFunc
from workspace.optimization_projects.CDKL5_DIV21._2_batchrun_optimization.calculate_fitness_v3 import fitnessFunc
from workspace.optimization_projects.CDKL5_DIV21._1_derive_features_from_experimental_data.derived_fitness_args.fitness_args_20241123_155335 import fitnessFuncArgs


import json
for seed in current_seeds:
    
    #get seed path
    seed_path = seed['path']
    
    #load seed data
    with open(seed_path, 'r') as seed_file:
        seed_data = json.load(seed_file)
        
    #init target data
    targets = fitnessFuncArgs['targets']
    max_fitness = fitnessFuncArgs['maxFitness']
    
    #prep fitness save path
    filename = seed_path.replace('_data.json', '')
    fitness_save_path = filename + '_fitness.json'
    
    simData = seed_data['simData']
    kwargs = {
        'simConfig': seed_data['simConfig'],
        'popData': seed_data['net']['pops'],
        'targets': targets,
        'maxFitness': max_fitness,
        'fitness_save_path': fitness_save_path,
        
        #for dealbreaker analysis
        #'net': seed_data['net'],
    }   
    fitnessFunc(simData=simData, mode="simulated data", **kwargs)


