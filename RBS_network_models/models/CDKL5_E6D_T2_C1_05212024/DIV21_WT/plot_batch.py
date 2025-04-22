import glob
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_network_metrics_v2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

batch_paths = [
    '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-21',
]

# get all files ending in _metrics.npy
metrics_files = []
for batch_path in batch_paths:
    metrics_files += glob.glob(f'{batch_path}/**/*_metrics.npy', recursive=True)

# # load the metrics files
# npy_list = []    
# for metrics_file in metrics_files:
#     # for debug
#     # if 'gen_0_cand_0' in metrics_file:
#     #     pass #i know this one works well
#     # elif 'gen_1_cand_0' in metrics_file:
#     #     pass #this one is a bit weird
#     # else:
#     #     continue
    
#     metrics = np.load(metrics_file, allow_pickle=True).item()
#     npy_list.append(metrics)
#     print(f'Loaded {metrics_file}')
#     #break #for testing


def load_metrics_file(fpath):
    metrics = np.load(fpath, allow_pickle=True).item()
    print(f'Loaded {fpath}')
    return metrics

# # assume metrics_files is your list of paths
# with ThreadPoolExecutor(max_workers=16) as executor:
#     npy_list = list(executor.map(load_metrics_file, metrics_files))

if __name__ == "__main__":
    # metrics_files is your list of .npy paths
    with Pool(processes=16) as pool:
        npy_list = pool.map(load_metrics_file, metrics_files)

kwargs = {}    
plot_network_metrics_v2(npy_list, kwargs, parallel=True, num_workers=16)
print('Done plotting network metrics')
#print(f'Found {len(metrics_files)} metrics files')