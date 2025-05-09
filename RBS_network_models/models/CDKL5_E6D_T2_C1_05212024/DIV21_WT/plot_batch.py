import glob
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_network_metrics_v2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import re
import os
import json

def load_metrics_file(fpath):
    metrics = np.load(fpath, allow_pickle=True).item()
    print(f'Loaded {fpath}')
    return metrics

if __name__ == "__main__":
    
    # ─── configure your batch paths here ────────────────────────────────────────────    
    batch_paths = [
        #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-21',
        
        # aw 2025-04-22 23:35:03
        #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-22/gen_1'
        
        # aw 2025-04-23 08:31:42
        #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-23/'

        ## aw 2025-04-26 10:50:48
        #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26/'

        # aw 2025-04-29 09:29:54
        #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-28/'

        # aw 2025-05-09 12:03:42
        #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-28'
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-09'
    ]

    # get all files ending in _metrics.npy
    metrics_files = []
    for batch_path in batch_paths:
        metrics_files += glob.glob(f'{batch_path}/**/*_metrics.npy', recursive=True)

    # get corresponding fitness jsons for each metrics file
    # paired_files = []
    # for fpath in metrics_files:
    #     json_path = fpath.replace('_metrics.npy', '_fitness.json')
    #     if os.path.exists(json_path):
    #         paired_files.append(json_path)
            
    # get a list of all fitness values for each metrics file, filter out to top n files
    fitness_values = []
    for fpath in metrics_files:
        json_path = fpath.replace('_metrics.npy', '_fitness.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f).get('fit', 1000) # Default to 1000 if not found
                fitness_values.append(data)
                #fitness_values.append(data['fit'])

    # sort the metrics files by fitness values - store idx to filter paired_files later
    fitness_values = np.array(fitness_values)
    #sorted_idx = np.argsort(fitness_values)[::-1]
    
    # sort min to max
    sorted_idx = np.argsort(fitness_values)
    
    # get the top n metrics files
    top_n = 256
    #top_n = 1
    metrics_files = np.array(metrics_files)[sorted_idx][:top_n]
    #paired_files = np.array(paired_files)[sorted_idx][:top_n]
    fitness_values = fitness_values[sorted_idx][:top_n]
    
    #
    num_workers = 256
    #num_workers = 1
    # metrics_files is your list of .npy paths
    with Pool(processes=num_workers) as pool:
        npy_list = pool.map(load_metrics_file, metrics_files)

    kwargs = {}    
    plot_network_metrics_v2(npy_list, kwargs, parallel=True, num_workers=num_workers)
    print('Done plotting network metrics')
    #print(f'Found {len(metrics_files)} metrics files')


'''
salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=1 --image=adammwea/axonkilo_docker:v7
shifter --image adammwea/axonkilo_docker:v7 bash
'''