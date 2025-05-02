print('Initializing...')
# =================== #
# Import Libraries
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import multiprocessing
import time
import random
import pandas as pd
# =================== #

# Main Test Function
# if __name__ == "__main__": # aw 2025-02-23 17:08:59 - older implementation of dtw_analysis_dynamic

#     # init lock
#     # Global Lock (outside function)
#     lock = None  # Initialize as None so it doesn't get pickled

#     # Run Tests
#     print('Running test_dtw.py')
#     print('Loading test data')    
#     path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/tests/dtw_test_data/'
#     sequence_stacks = np.load(path + 'sequence_stacks.npy', allow_pickle=True).item()

#     # test dtw # aw 2025-02-23 14:52:22 - this probably won't work anymore due to changes of input data
#         # print('Running dtw')
#         # dtw_results = dtw_analysis_v3(
#         #     time_sequence_dict, cat_sequence_dict, sequence_stacks, bursting_data
#         # )

#     # test dynamic dtw
#     print('Running dynamic dtw')
#     #previous_computed_data = np.load(path + 'dtw_analysis_dynamic_results.npy', allow_pickle=True).item()
#     data_path = path
#     mean_dtw, std_dtw, variance_dtw, global_matrix = dtw_analysis_dynamic(sequence_stacks, data_path,
#                                                                         #confidence_threshold=0.99, min_samples=500
#                                                                         )
# aw 2025-02-23 17:09:19 - newer implementation of dtw_analysis_dynamic
if __name__ == "__main__":
    print('Running test_dtw.py')
    print('Loading test data')

    path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/tests/dtw_test_data/'
    sequence_stacks = np.load(path + 'sequence_stacks.npy', allow_pickle=True).item()

    print('Running dynamic DTW...')
    data_path = path
    mean_dtw, std_dtw, variance_dtw, global_matrix = dtw_analysis_dynamic_v2(sequence_stacks, 
                                                                             data_path,
                                                                             cv_threshold=0.05,
                                                                             moving_avg_window=500,
                                                                             moving_avg_threshold=0.01,
                                                                             )
    print('DTW Analysis Complete!')
'''
salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=1 --image=adammwea/axonkilo_docker:v7
module load conda
conda activate netsims_env
pip install -e /pscratch/sd/a/adammwea/workspace/RBS_network_models
pip install -e /pscratch/sd/a/adammwea/workspace/MEA_Analysis
python /pscratch/sd/a/adammwea/workspace/RBS_network_models/tests/test_dtw.py
'''