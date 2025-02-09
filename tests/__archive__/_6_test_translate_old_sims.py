global PROGRESS_SLIDES_PATH, SIMULATION_RUN_PATH, REFERENCE_DATA_NPY, CONVOLUTION_PARAMS, DEBUG_MODE
from DIV21.utils.sim_helper import *
import json
# ===================================================================================================
# sim_data_path = (
#     '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/'
#     'CDKL5_DIV21/tests/outputs/test_run_a_simulation/test_run_5_data.pkl')
# output_dir =(
#     '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/'
#     'CDKL5_DIV21/tests/outputs/test_sensitivity_analysis')
old_sim_data_dir = (
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/'
    'CDKL5_DIV21/outputs')
output_dir = (
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/'
    'CDKL5_DIV21/tests/outputs/test_translation_of_old_sims')
reference_data_path = (
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/'
    'CDKL5_DIV21/src/experimental_reference_data/CDKL5-E6D_T2_C1_05212024_240611_M06844_Network_000076_network_metrics_well000.npy')
# ===================================================================================================
#walk through output_dir and get all .pkl file paths
#sim_data_paths = []
excluded_file_types = [
    '.py', 
    #'.npy', 
    #'_params.pkl'
    '_batch.json',
    '.csv',
    '.seed',
    '_cfg.json',    #   NOTE: ironically this might not be the best way to get cfg information because of the way the cfg is saved.
                    #   at least, it doesn't seem to retain the file path information
    ]
data_paths = []
for root, dirs, files in os.walk(old_sim_data_dir):
    for file in files:
        #if file.endswith('_data.pkl'):
        #sim_data_paths.append(os.path.join(root, file))
        if not any([file.endswith(ext) for ext in excluded_file_types]):
            data_paths.append(os.path.join(root, file))
            
#data_paths = sorted(data_paths)

for path in data_paths:
    if path.endswith('.json'):
        print(path)
        
        # # _cfg.json
        # if '_cfg' in path:
            
        #     def handle_json_cfg(path):
        #         with open(path, 'r') as f:
        #             cfg = json.load(f)
        #         #print(cfg)
        #         if 'simConfig' in cfg:
        #             old_cfg = cfg['simConfig']
        #         else:
        #             raise Exception('Unexpected JSON format')
        #         from CDKL5_DIV21.src.cfg import cfg as template_cfg
                
                                
        #         if 'simulation_run_path' in cfg:
        #             print('simulation_run_path:', cfg['simulation_run_path'])
        #         if 'reference_data_path' in cfg:
        #             print('reference_data_path:', cfg['reference_data_path'])
        #         if 'convolution_params_path' in cfg:
        #             print('convolution_params_path:', cfg['convolution_params_path'])
        #         if 'debug_mode' in cfg:
        #             print('debug_mode:', cfg['debug_mode'])
        #         if 'progress_slides_path' in cfg:
        #             print('progress_slides_path:', cfg['progress_slides_path'])
        #         if 'output_dir' in cfg:
        #             print('output_dir:', cfg['output_dir'])
        #         if 'duration_seconds' in cfg:
        #             print('duration_seconds:', cfg['duration_seconds'])
        #         if 'save_data' in cfg:
        #             print('save_data:', cfg['save_data'])
        #         if 'overwrite_cfgs' in cfg:
        #             print('overwrite_cfgs:', cfg['overwrite_cfgs'])
        #         print()
        #     handle_json_cfg(path)
        
        # _data.json - try to get _data.json files first. The contain the most information from these older sims.
        if '_data' in path:
            def handle_json_data(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                #print(data)
                if 'simData' in data:
                    old_data = data['simData']
                else:
                    raise Exception('Unexpected JSON format')
                from DIV21.src.cfg import cfg as template_cfg
            handle_json_data(path)
    elif path.endswith('.pkl'):
        print(path)
    elif path.endswith('.npy'):
        print(path)

# NOTE: 2025-01-03 13:28:25
# aw - on second thought this script is not necessary. I can just use the old data files directly.