# CDKL5-E6D_T2_C1_DIV21 tuning script
# Modeled after the network_metric_targets dictionary in modules/analysis/simulation_fitness_functions/calculate_fitness.py

'''Setup Python environment '''
import setup_environment
import sys
import os
setup_environment.set_pythonpath()

'''Imports'''
import os
import spikeinterface.sorters as ss
import workspace.RBS_network_simulations.modules.mea_processing_library as mea_lib
import numpy as np
from modules.analysis.analyze_network_activity import get_experimental_network_activity_metrics
import pickle
import dill

'''functions'''
def get_base_path(recording_path):
    return os.path.dirname(recording_path).replace('xRBS_input_data', 'yRBS_spike_sorted_data')

def get_presumed_sorting_output_path(recording_path, stream):
    base_path = get_base_path(recording_path)
    
    #generate initial path
    path = os.path.join(base_path, f"well{str(stream).zfill(3)}/sortings/sorter_output")
    
    #replace xInputs with yThroughput
    path = path.replace('xInputs', 'yThroughput')
    path = path.replace('xRBS_input_data', 'yRBS_spike_sorted_data')
    
    #assert replacements have been made
    assert 'xInputs' not in path, f"Error: 'xInputs' not replaced in path: {path}"
    assert 'xRBS_input_data' not in path, f"Error: 'xRBS_input_data' not replaced in path: {path}"    

    return path

def load_spike_sorted_data(path):
    # sorting_object_list = []
    # for path in paths:
    #sorter_output_folder = os.path.abspath(path)
    sorter_output_folder = path
    assert os.path.exists(sorter_output_folder), f"Error: path does not exist: {sorter_output_folder}"
    sorting_object = ss.Kilosort2Sorter._get_result_from_folder(sorter_output_folder)
    #sorting_object_list.append(sorting_object)
    return sorting_object

def make_json_serializable(obj):
    # Handle NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle NumPy scalar types
    if isinstance(obj, (np.int64, np.float64)):
        return obj.item()
    # Raise error for unsupported types
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def convert_keys_to_str(data):
    if isinstance(data, dict):
        return {str(k): convert_keys_to_str(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys_to_str(i) for i in data]
    else:
        return data

def sort_data(recording_path, stream_num, sorting_output_path):
    #alloc to enable running in perlmutter:
    #salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 1 --account m2043_g --image=docker:adammwea/axonkilo_docker:v7
    #shifter --image=docker:adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_1_derive_features_from_experimental_data/derive_CDKL5-E6D_T2_C1_DIV21_features.py

    #spike sort using kilosort2
    #from modules.mea_processing_library import load_recordings, kilosort2_wrapper
    #print module location
    print(f"Using kilosort2_wrapper from {mea_lib.__file__}")
    
    #load recording
    h5_file_path = recording_path
    stream_select = stream_num
    recording_object = mea_lib.load_recordings(h5_file_path, stream_select=stream_select)
    stream_objs = recording_object[1]
    
    #sort data
    for wellid, stream_obj in stream_objs.items():
        # don't set sorting params, just do defualt.
        maxwell_recording_extractor = stream_obj['recording_segments'][0] #note: network scans should only have one recording segment
        recording = maxwell_recording_extractor
        output_path = sorting_output_path
        sorting_params = ss.Kilosort2Sorter.default_params()
        sorting_object=mea_lib.kilosort2_wrapper(recording, output_path, sorting_params)
        print(f"Sorting complete for {wellid}")    

'''main script'''
if __name__ == '__main__':
    
    '''Args'''
    stream_select = None  # Set to a list of well indices if needed, e.g., [0, 2, 4]
    stream_select = 0

    '''Recordings'''
    recordings = [
        #"/pscratch/sd/a/adammwea/xRBS_input_data/CDKL5-E6D_T2_C1_05212024/240611/M06844/Network/000076/data.raw.h5",
        "/pscratch/sd/a/adammwea/workspace/xInputs/xRBS_input_data/CDKL5-E6D_T2_C1_05212024/240611/M06844/Network/000076/data.raw.h5"
    ]
    
    '''Convolution Params''' #going to be tweaking this as we develop from now on, so I'll need to keep this up to date
    #convolution_params_path = "/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/convolution_params/_standard_convolution_params.py"
    convolution_params_path = "/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/convolution_params/241202_convolution_params.py"
        #tweaking binsize to be smaller to get better resolution of network activity for shorter simulations.
        
    '''options'''
    #default options
    spike_sort = False
    get_metrics = False
    
    #comment out as needed
    #spike_sort = True
    get_metrics = True
    
    '''sort data'''
    if spike_sort:
        #sort data if needed
        for recording_path in recordings:
            stream_nums = [0, 1, 2, 3, 4, 5]
            if stream_select is not None: stream_nums = [stream_select]
            for stream_num in stream_nums:
                #get sorting output path
                sorting_output_path = get_presumed_sorting_output_path(recording_path, stream_num)
                sorting_output_path = os.path.dirname(sorting_output_path)
                os.makedirs(sorting_output_path, exist_ok=True)
                print(f"Sorting output path: {sorting_output_path}")
                sorting_object = sort_data(recording_path, stream_num, sorting_output_path)
    
    '''get metrics'''
    if get_metrics:
    #load recordings and sorting data
        for recording_path in recordings:
            stream_nums = [0, 1, 2, 3, 4, 5]
            if stream_select is not None: stream_nums = [stream_select]
            for stream_num in stream_nums:    
                #load recordings and sorting data
                MaxID, recording_object, expected_well_count, rec_counts = mea_lib.load_recordings(recording_path, stream_select=stream_select)
                sorting_output_path = get_presumed_sorting_output_path(recording_path, stream_select)
                stream_sorting_object = load_spike_sorted_data(sorting_output_path)
                
                #get recording segment - note: network scans should only have one recording segment
                well_id = f'well{str(0).zfill(2)}{stream_select}'
                network_recording_segment = recording_object[well_id]['recording_segments'][0]  #note: network scans should only have one recording segment
                
                #build results dictionary
                kwargs = {
                    'recording_object': network_recording_segment,
                    'sorting_object': stream_sorting_object,
                }
                
                #get network activity metrics
                #import conv_params from convolution_params_path
                with open(convolution_params_path, 'r') as f:
                    exec(f.read())
                    kwargs['conv_params'] = conv_params
                network_metrics = get_experimental_network_activity_metrics(**kwargs)
                
                # save network metrics
                print("Saving network metrics as pickle...")
                current_path = os.path.dirname(os.path.abspath(__file__))
                save_path = os.path.join(current_path, 'network_metrics', f"network_metrics_well00{stream_num}.pkl")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump(network_metrics, f)
                print(f"Saved network metrics to {save_path}")
                
                print("Saving network metrics as numpy...")
                #save_path = os.path.join(get_base_path(recording_path), f"well{str(stream_select).zfill(3)}/sortings/sorter_output/network_metrics.npy")
                # save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"network_metrics_well00{stream_num}.npy")
                save_path = os.path.join(current_path, 'network_metrics', f"network_metrics_well00{stream_num}.npy")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, network_metrics)
                print(f"Saved network metrics to {save_path}")