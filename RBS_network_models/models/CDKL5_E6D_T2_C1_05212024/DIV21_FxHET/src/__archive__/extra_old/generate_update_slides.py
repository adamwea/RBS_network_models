## Step 1: get data from simulation output

simulation_run_paths = [
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams',
]


import os
stats_paths = []
stats_indv_paths = []
for path in simulation_run_paths:
    run_basename = os.path.basename(path)
    stats_path = os.path.join(path, f'{run_basename}_stats.csv')
    stats_indv_path = os.path.join(path, f'{run_basename}_stats_indiv.csv')
    stats_paths.append(stats_path)
    stats_indv_paths.append(stats_indv_path)
    assert os.path.exists(stats_path), f'{stats_path} does not exist'
    assert os.path.exists(stats_indv_path), f'{stats_indv_path} does not exist'

print('') # for legibility
print(stats_paths)
print(stats_indv_paths)

import os
import json
import pandas as pd

import os
import json
import pandas as pd
import re

def clean_json(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Remove trailing brackets or other extraneous characters
    cleaned_lines = []
    open_brackets = 0
    for line in lines:
        open_brackets += line.count('{') - line.count('}')
        cleaned_lines.append(line)
        if open_brackets == 0:
            break
    
    # Remove any remaining lines that might contain extra brackets
    cleaned_content = ''.join(cleaned_lines)
    
    # Remove trailing characters after the most un-indented closing bracket
    last_closing_bracket_index = cleaned_content.rfind('}')
    cleaned_content = cleaned_content[:last_closing_bracket_index + 1]
    cleaned_content = cleaned_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
    # Replace NaN values with null
    cleaned_content = re.sub(r'\bNaN\b', 'null', cleaned_content)

    
    return cleaned_content
simulation_run_paths = [
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams',
]

optimization_runs = []

for path in simulation_run_paths:
    run_data = {
        'generations': {},
        'batch_cfg': None,
        'batch': None,
        'user_args': None,
        'stats': None,
        'stats_indv': None,
    }
    
    basename = os.path.basename(path)
    batch_cfg_path = os.path.join(path, f'batch_config.json')
    batch_path = os.path.join(path, f'{basename}_batch.json')
    user_args_path = os.path.join(path, 'temp_user_args.py')
    stats_path = os.path.join(path, f'{basename}_stats.csv')
    stats_indv_path = os.path.join(path, f'{basename}_stats_indiv.csv')
    
    assert os.path.exists(batch_cfg_path), f'{batch_cfg_path} does not exist'
    assert os.path.exists(batch_path), f'{batch_path} does not exist'
    assert os.path.exists(user_args_path), f'{user_args_path} does not exist'
    assert os.path.exists(stats_path), f'{stats_path} does not exist'
    assert os.path.exists(stats_indv_path), f'{stats_indv_path} does not exist'
    
    with open(batch_cfg_path, 'r') as f:
        batch_cfg = json.load(f)
        run_data['batch_cfg'] = batch_cfg
    with open(batch_path, 'r') as f:
        netparams_batch = json.load(f)
        run_data['netparams_batch'] = netparams_batch
    with open(user_args_path, 'r') as f:
        user_args = f.read()
        run_data['user_args'] = user_args
    stats = pd.read_csv(stats_path)
    run_data['stats'] = stats
    stats_indv = pd.read_csv(stats_indv_path)
    run_data['stats_indv'] = stats_indv
        
    for root, dirs, files in os.walk(path):
        dirs = sorted(dirs)
        for name in dirs:
            if name.startswith('gen'):
                print('getting data from', name)
                generation_data = {
                    'cfgs': {},
                    'data': {},
                    'fitness': {},
                    'netparams': {},
                }
                for _, _, files in os.walk(os.path.join(root, name)):
                    for file in files:
                        if file.endswith('_cfg.json'):
                            print('getting data from', file)
                            with open(os.path.join(root, name, file), 'r') as f:
                                cfg = json.load(f)
                                simLabel = cfg['simConfig']['simLabel']
                                generation_data['cfgs'][simLabel] = cfg
                            data_path = os.path.join(root, name, f'{simLabel}_data.json')
                            fitness_path = os.path.join(root, name, f'{simLabel}_fitness.json')
                            netparams_path = os.path.join(root, name, f'{simLabel}_netParams.json')
                            assert os.path.exists(data_path), f'{data_path} does not exist'
                            assert os.path.exists(netparams_path), f'{netparams_path} does not exist'
                            try: assert os.path.exists(fitness_path), f'{fitness_path} does not exist'
                            except: continue
                            with open(data_path, 'r') as f:
                                data = json.load(f)
                                generation_data['data'][simLabel] = data
                            with open(fitness_path, 'r') as f:
                                fitness = json.load(f)
                                generation_data['fitness'][simLabel] = fitness
                            print('getting data from', netparams_path)
                            try:
                                with open(netparams_path, 'r') as f:
                                    netparams = json.load(f)
                                    generation_data['netparams'][simLabel] = netparams
                            except json.JSONDecodeError:
                                print(f"Cleaning up JSON file: {netparams_path}")
                                cleaned_content = clean_json(netparams_path)
                                try:
                                    netparams = json.loads(cleaned_content)
                                    generation_data['netparams'][simLabel] = netparams
                                except json.JSONDecodeError as e:
                                    print(f"Failed to clean JSON file: {netparams_path}, error: {e}")
                            #break
                run_data['generations'][name] = generation_data
    optimization_runs.append(run_data)

print(optimization_runs)