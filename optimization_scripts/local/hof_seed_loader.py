import os
import json
import pandas as pd
from simulate._temp_files.temp_user_args import *
from simulation_config_files import evolutionary_parameter_space

def get_HOF_seeds():
    '''Loads seeds from the Hall of Fame (HOF).'''
    print(f'Loading Hall of Fame from {USER_HOF}')
    assert os.path.exists(USER_HOF), f'USER_HOF file not found: {USER_HOF}'
    seeded_HOF_cands = pd.read_csv(USER_HOF).values.flatten()
    seeded_HOF_cands = [cfg.replace('_data', '_cfg') for cfg in seeded_HOF_cands]
    seeded_HOF_cands = [os.path.abspath(f'./{cfg}') for cfg in seeded_HOF_cands]

    for cfg in seeded_HOF_cands:
        if 'NERSC/NERSC' in cfg: 
            seeded_HOF_cands[seeded_HOF_cands.index(cfg)] = cfg.replace('NERSC/NERSC', 'NERSC')

    seeds = []
    for cfg in seeded_HOF_cands:
        if not os.path.exists(cfg): continue
        with open(cfg, 'r') as f:
            seed = json.load(f)['simConfig']

        seed = {
            key: evolutionary_parameter_space[key][0] if evolutionary_parameter_space[key][0] == evolutionary_parameter_space[key][1] 
            else seed[key] for key in evolutionary_parameter_space if key in seed
        }
        seed = [float(val) for val in seed.values()]
        seeds.append(seed)
        if len(seeds) >= USER_pop_size: break

    print(f'Loaded {len(seeds)} seeds from Hall of Fame')
    assert len(seeds) > 0, 'No seeds loaded from Hall of Fame'
    return seeds