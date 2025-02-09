import os

# ============================================================================================
batch_dir = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/batch_runs/batch_2025-02-07'
feature_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/features/20250204_features.py'
reference_data_path = (
    # "/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/"
    # "CDKL5_DIV21/_config/experimental_data_features/network_metrics/"
    # "CDKL5-E6D_T2_C1_05212024_240611_M06844_Network_000076_network_metrics_well000.npy"
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/'
    'network_metrics/CDKL5-E6D_T2_C1_05212024_240611_M08029_Network_000091_network_metrics_well001.npy'
)


# get penultimate gen dir

for root, dirs, files in os.walk(batch_dir):
        #dirs.sort(reverse=True)
        # sort by number in dir name preceeded by 'gen_'
        # skip pycache
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        print(dirs)
        dirs.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
                
        
        ultimate_dir = os.path.join(root, dirs[0])
        penultimate_dir = os.path.join(root, dirs[1])
        break
print(ultimate_dir)
print(penultimate_dir)

# load csv, get lowest and highest fitness from 3rd and 4th columns respectively from last row
import pandas as pd
for root, dirs, files in os.walk(batch_dir):
    for file in files:
        if file.endswith('_stats.csv'):
            csv_path = os.path.join(root, file)
            df = pd.read_csv(csv_path, delimiter=' ', skipinitialspace=True)
            print(df.head())
            print("DataFrame Shape:", df.shape)

            
            # print(df.iloc[:,:])
            # print(df.iloc[0,0])
            
            min_fitness = df.iloc[-1, 2]
            max_fitness = df.iloc[-1, 3]
            print('min fitness: ', min_fitness)
            print('max fitness: ', max_fitness)
            break
        
# collect a list of all sim data paths, fitness data paths. Sort by fitness.
data_paths = {}
for root, dirs, files in os.walk(penultimate_dir):
        # skip pycache
        #print(files)
        
        for file in files:
            if file.endswith('_data.pkl'):
                try:                    
                    data_path = os.path.join(root, file)
                    fitness_path = os.path.join(root, file.replace('_data.pkl', '_fitness.json'))
                    import json
                    with open(fitness_path, 'r') as f:
                        fitness = json.load(f)
                    fit = fitness['average_fitness']
                    #print(fit)
                    data_paths[fit] = {
                        'data_path': data_path,
                        'fitness_path': fitness_path,
                    }
                    #print(data_paths[fit])
                except Exception as e:                    
                    #print(e)
                    continue
                
        # sort by numeric values of keys
        data_paths = dict(sorted(data_paths.items()))
        # print(data_paths)
        # import sys
        # sys.exit()
        
# collect the top 10 data_paths
top_10_data_paths = {}
for i, (k, v) in enumerate(data_paths.items()):
    if i < 10:
        top_10_data_paths[k] = v
    else:
        break
    
print(top_10_data_paths)
# import sys
# sys.exit()

# walk through penultimate dir and plot each sim
#for root, dirs, files in os.walk(penultimate_dir):
        # skip pycache
        #print(files)
from netpyne import sim
for path in top_10_data_paths.values():
    # data_path = path['data_path']
    # file = data_path
    file = path['data_path']
    
    # if needed clear sim before runnning process_simulation
    try:
        sim.clearAll()
    except:
        pass
        
        #for file in files:
    if file.endswith('_data.pkl'):
        #data_path = os.path.join(root, file)
        #print(data_path)
        print(file)
        # plot sim
        from RBS_network_models.sim_analysis import process_simulation
        from RBS_network_models.Organoid_RTT_R270X.DIV112_WT.src.conv_params import conv_params, mega_params
        from RBS_network_models.utils.cfg_helper import import_module_from_path
        feature_module = import_module_from_path(feature_path)
        fitnessFuncArgs = feature_module.fitnessFuncArgs
        #sim_data_path = data_path
        sim_data_path = file
        #reference_data_path = os.path.join(ultimate_dir, 'best_data.pkl')
        try:
            process_simulation(
            sim_data_path, 
            reference_data_path,
            DEBUG_MODE=False,
            conv_params = conv_params,
            mega_params = mega_params,
            fitnessFuncArgs = fitnessFuncArgs,
            )        
            #break
        except Exception as e:
            print(e)
            continue

