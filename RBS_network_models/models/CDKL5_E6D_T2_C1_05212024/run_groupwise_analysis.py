# This script requires paths for network_metrics analysis objects.

import numpy as np
import os
import glob


# helper functions ============================================================
def load_metrics_file(fpath):
    try:
        metrics = np.load(fpath, allow_pickle=True).item()
        #sim_data_path = metrics.get('sim_data_path', None)
        #if sim_data_path is None:
            # sim_data_dir = os.path.dirname(os.path.dirname(fpath))
            # # find a file in sim_data_dir that ends with '_data.pkl'
            # sim_data_files = glob.glob(os.path.join(sim_data_dir, '*_data.pkl'))
            # if sim_data_files:
            #     sim_data_path = sim_data_files[0]
            # else:
            #     raise FileNotFoundError(f'No simulation data file found in {sim_data_dir} matching *_data.pkl')
            # #sim_data_path = fpath.replace('_metrics.npy', '_data.pkl')
            # metrics['sim_data_path'] = sim_data_path
        print(f'Loaded {fpath}')
        return metrics
    except Exception as e:
        print(f'Error loading {fpath}: {e}')
        return None

# NOTE: This script appears to be the correct identities of these data and wells... # aw 2025-06-26 11:53:55 fix names later
data_paths = {
    'FxHET': {
        'DIV21': [
            '/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well005/metrics.npy',            
        ]
        },
    'MxWT': {
        'DIV21': [
            '/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well002/metrics.npy',
            
        ]
        },
}

# 
data_struct = {}
for genotype, div_data in data_paths.items():
    data_struct[genotype] = {}
    for div, paths in div_data.items():
        data_struct[genotype][div] = []
        for path in paths:
            #metrics = ef.load_metrics_file(path)
            # try:
            #     metrics = np.load(path, allow_pickle=True).item()
            # except Exception as e:
            #     print(f"Error loading {path}: {e}")
            #     metrics = None
            metrics = load_metrics_file(path)
            if metrics is not None:
                data_struct[genotype][div].append(metrics)
                
print("Data structure loaded successfully.")