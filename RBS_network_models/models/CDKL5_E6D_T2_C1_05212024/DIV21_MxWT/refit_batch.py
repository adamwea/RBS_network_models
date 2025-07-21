'''
This script aims to simulate the optimization case of runing batch simulations to help debug/develop the fitness function.
'''
# imports ===============================================================================
from netpyne import sim
from RBS_network_models.fitnessFunc import fitnessFunc_v3
import numpy as np
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.conv_params import conv_params, mega_params
import glob
import json
#from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.fitness_schema.schema_1 import fit_schema
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.fitness_schema.schema_2 import fit_schema
#from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.seeds import seeds
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.seeds_3 import seeds
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.evol_params import params
from copy import deepcopy
import os

# helper functions ==========================================================
def get_seed_cfgs(params, **kwargs):
    seeds_iterable = []
    fitness_data = []
    seeds = kwargs.get('seeds', None)

    # reverse the order of seeds, if limited, use the newest seeds before the oldest
    if seeds is None: raise ValueError("seeds must be provided in kwargs")
    #seeds = seeds[::-1]

    # get number of elites, only use this many seeds if less than the number of seeds
    # num_elites = kwargs.get('num_elites', None)
    # if num_elites is None: raise ValueError("num_elites must be provided in kwargs")
    # if len(seeds) > num_elites:
    #     seeds = seeds[:num_elites]
    #     print(f'Using {num_elites} seeds: {seeds}')

    # limit to pop size instead #aw 2025-05-19 17:49:46
    pop_size = kwargs.get('pop_size', None)
    if pop_size is None: raise ValueError("pop_size must be provided in kwargs")
    if len(seeds) > pop_size:
        seeds = seeds[:pop_size]
        print(f'Using {pop_size} seeds: {seeds}')

    # format the seeds into a list of lists, where each list is a candidate - as netpyne expects  
    evol_params = params.copy()
    for i, seed in enumerate(seeds):
        #load simcfg from seed
        seed_iterable = []
        sim.loadSimCfg(seed, setLoaded=True)
        simcfg = deepcopy(sim.cfg.todict())
        fitpath = seed.replace('cfg', 'fitness')

        # load fitness data
        try:
            if os.path.exists(fitpath):
                with open(fitpath, 'r') as f:
                    fit = json.load(f)
                #simcfg['fitness'] = fitness_data
            else:
                fit = {}
        except Exception as e:
            print(f'Error loading fitness data from {fitpath}: {e}')
            fit = {}


        for key in evol_params:
            if key in simcfg:
                seed_iterable.append(simcfg[key])
            else:
                upper_bound = evol_params[key]['values'][1]
                lower_bound = evol_params[key]['values'][0]
                random_value = np.random.uniform(lower_bound, upper_bound)
                seed_iterable.append(random_value)
        seeds_iterable.append(seed_iterable)
        fitness_data.append(fit)
    if len(seeds_iterable) == 0: seeds_iterable = None
    return seeds_iterable, fitness_data

# main logic =================================================================
# define reference data path
reference_data_path = '/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well005/metrics.npy'

# define batch path
#batch_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-22/gen_1'

# aw 2025-04-30 13:39:56
#batch_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26'

# 2025-05-15 12:42:28
#batch_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-15'

# aw 2025-05-19 22:57:13
#batch_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-19_BRandFRratios'

# 2025-05-25 22:24:25
#batch_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-23_normBRandFRR_optBLandAmps_2'

# aw 2025-05-27 13:41:32
batch_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-27_reduced_dealbreakers'

# get all sim data paths, look for files ending in _data.pkl
#sim_data_paths = glob.glob(batch_path + '/*_data.pkl')

# load all _fitness.json files, sort them lowest to highest fit value
fit_json_paths = glob.glob(batch_path + '/**/*_fitness.json')
fit_scores = []
for fit_json_path in fit_json_paths:
    try:
        with open(fit_json_path, 'r') as f:
            fit_score = json.load(f)['fit']
            fit_scores.append((fit_json_path, fit_score))
    except:
        print(f'Error loading {fit_json_path}')
        continue

# sort fit scores by low to high, get the sort order, apply to fit_json_paths
sort_idx = np.argsort([fit_score[1] for fit_score in fit_scores])
fit_scores = [fit_scores[i] for i in sort_idx]

# derive sorted sime_data_paths from sorted fit_json_paths
sim_data_paths = []
for fit_json_path, fit_score in fit_scores:
    # get the sim_data_path from the fit_json_path
    sim_data_path = fit_json_path.replace('_fitness.json', '_data.pkl')
    sim_data_paths.append(sim_data_path)


# get all sim data paths, look for files ending in _data.pkl, recursively
#sim_data_paths = glob.glob(batch_path + '/**/*_data.pkl', recursive=True)

#convert seeds into interable list of params
kwargs = {
    'seeds': seeds,
    'pop_size': 64,
}
print('setting seeds configs...')
seeds_iterable, seed_fitness = get_seed_cfgs(params, **kwargs)
#kwargs['seed_fitness'] = seed_fitness

for sim_data_path in sim_data_paths:
    # load sim data
    sim.load(sim_data_path)
    simData = sim.allSimData.todict().copy()

    # define fitness function args
    fitnessFuncArgs = {
        #'reference_data': reference_data,
        'conv_params': conv_params,
        'mega_params': mega_params,
        'plot_sim': False,
        'reference_data_path': reference_data_path,
        'batching': False, #NOTE: this is the only difference between this and the fitnessFunc_v3 during batch processing - if true fitnessFunc will try to load data from call stack, which wont work here.
        'sim_data_path': sim_data_path, # needed if batching is False
        
        # compute_network_metrics args
        #'try_load': True,
        'try_load': False,
        'run_parallel': True,
        'max_workers': 32,
        #'max_workers': 256,
        'burst_sequencing': True,
        #'plot_fit_curve': True, #default is False
        'plot_fit_curve': False, #default is False

        #fitness schema
        'fit_schema': fit_schema,
        'seed_fitness': seed_fitness,     
    } 

    #run fitness function
    avg_fitness = fitnessFunc_v3(simData, **fitnessFuncArgs)
    print(f'Avg fitness for {sim_data_path}: {avg_fitness}')
    
    # clear sim object
    sim.clearAll()