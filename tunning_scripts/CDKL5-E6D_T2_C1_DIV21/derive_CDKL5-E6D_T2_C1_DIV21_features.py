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
import tunning_scripts.mea_processing_library as mea_lib
import numpy as np
from modules.analysis.analyze_network_activity import get_experimental_network_activity_metrics
import pickle
import dill

'''Args'''
stream_select = None  # Set to a list of well indices if needed, e.g., [0, 2, 4]
stream_select = 0

'''Recordings'''
recordings = [
    "/pscratch/sd/a/adammwea/xRBS_input_data/CDKL5-E6D_T2_C1_05212024/240611/M06844/Network/000076/data.raw.h5"
]

#get base path and sorting output path
def get_base_path(recording_path):
    return os.path.dirname(recording_path).replace('xRBS_input_data', 'yRBS_spike_sorted_data')

# get sorting output path
def get_presumed_sorting_output_path(recording_path, stream):
    base_path = get_base_path(recording_path)
    #paths = []
    #for i in range(6):
    path = os.path.join(base_path, f"well{str(stream).zfill(3)}/sortings/sorter_output")
    # if os.path.exists(path):
    #     paths.append(path)
    return path

#load spike sorted data
def load_spike_sorted_data(path):
    # sorting_object_list = []
    # for path in paths:
    sorter_output_folder = os.path.abspath(path)
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

# Convert dictionary keys to strings
def convert_keys_to_str(data):
    if isinstance(data, dict):
        return {str(k): convert_keys_to_str(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys_to_str(i) for i in data]
    else:
        return data

if __name__ == '__main__':
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
            network_metrics = get_experimental_network_activity_metrics(**kwargs)
            
            # save network metrics
            # save_path = os.path.join(get_base_path(recording_path), f"well{str(stream_select).zfill(3)}/sortings/sorter_output/network_metrics.pkl")
            # save path = current path
            print("Saving network metrics as pickle...")
            current_path = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(current_path, f"network_metrics_well00{stream_num}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(network_metrics, f)
            print(f"Saved network metrics to {save_path}")
            
            # #also save as dill
            # #save_path = os.path.join(get_base_path(recording_path), f"well{str(stream_select).zfill(3)}/sortings/sorter_output/network_metrics.dill")
            # save_path = os.path.join(current_path, f"network_metrics_well00{stream_num}.dill")
            # with open(save_path, 'wb') as f:
            #     dill.dump(network_metrics, f)
            # print(f"Saved network metrics to {save_path}")
            
            #save file as json
            # #save_path = os.path.join(get_base_path(recording_path), f"well{str(stream_select).zfill(3)}/sortings/sorter_output/network_metrics.json")
            # print("Saving network metrics as JSON...")
            # import json
            # current_path = os.path.dirname(os.path.abspath(__file__))
            # save_path = os.path.join(current_path, f"network_metrics_well00{stream_num}.json")
            # # Save your network_metrics dictionary to JSON
            # #Prepare dictionary for JSON serialization
            # network_metrics_serializable = convert_keys_to_str(network_metrics)
            # with open(save_path, 'w') as json_file:
            #     json.dump(network_metrics_serializable, json_file, default=make_json_serializable, indent=4)
            # # with open('network_metrics.json', 'w') as json_file:
            # #     json.dump(network_metrics, json_file, default=convert_to_serializable, indent=4)
            # print(f"Saved network metrics to {save_path}")
            
            #save as numpy file?
            print("Saving network metrics as numpy...")
            #save_path = os.path.join(get_base_path(recording_path), f"well{str(stream_select).zfill(3)}/sortings/sorter_output/network_metrics.npy")
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"network_metrics_well00{stream_num}.npy")
            np.save(save_path, network_metrics)
            print(f"Saved network metrics to {save_path}")
            
            
            
    #it's convenient to precalc some of the metrics
    #timeVector = get_time_vector(results)
    
    
    
    
    # network_metric_targets = {
    #     #General Data
    #     'source': 'experimental', # 'simulated' or 'experimental'
    #     'timeVector': timeVector,
        
    #     #Spiking Data
    #     'spiking_data': {
    #         'spike_times': get_spike_times(results),
    #         'spiking_times_by_unit': get_spike_times_by_unit(results),
    #         #'spiking_data_by_unit': None,
    #         'spiking_summary_data': {
    #             #'spike_times': None,
    #             'MeanFireRate': {
    #                 #'target': 2, #TODO: these are placeholder values
    #                 'min': get_min_fr(results),
    #                 'max': get_max_fr(results),
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #             'CoVFireRate': {
    #                 #'target': 1000, #TODO: these are placeholder values
    #                 'min': get_min_CoV_fr(results),
    #                 'max': get_max_CoV_fr(results),
    #                 'weight': 1, #TODO: update these with Nfactors
    #                 },
    #             'MeanISI': {
    #                 #'target': 1000, #TODO: these are placeholder values
    #                 'min': get_min_mean_isi(results),
    #                 'max': get_max_mean_isi(results),
    #                 'weight': 1, #TODO: update these with Nfactors
    #                 },
    #             'CoV_ISI': {
    #                 #'target': 1000, #TODO: these are placeholder values
    #                 'min': get_min_CoV_isi(results),
    #                 'max': get_max_CoV_isi(results),
    #                 'weight': 1, #TODO: update these with Nfactors
    #                 },         
    #         },
    #     },
        
    #     #Bursting Data
    #     'bursting_data': {
    #         'bursting_summary_data': {
    #             'baseline': {
    #                 'target': 0, #TODO: these are placeholder values
    #                 'min': None,
    #                 'max': None,
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #             'MeanWithinBurstISI': {
    #                 'target': 1000, #TODO: these are placeholder values
    #                 'min': None,
    #                 'max': None,
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #             'CovWithinBurstISI': {
    #                 'target': 1000, #TODO: these are placeholder values
    #                 'min': None,
    #                 'max': None,
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #             'MeanOutsideBurstISI': {
    #                 'target': 1000, #TODO: these are placeholder values
    #                 'min': None,
    #                 'max': None,
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #             'CoVOutsideBurstISI': {
    #                 'target': 1000, #TODO: these are placeholder values
    #                 'min': None,
    #                 'max': None,
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #             'MeanNetworkISI': {
    #                 'target': 1000, #TODO: these are placeholder values
    #                 'min': None,
    #                 'max': None,
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #             'CoVNetworkISI': {
    #                 'target': 1000, #TODO: these are placeholder values
    #                 'min': None,
    #                 'max': None,
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #             'Number_Bursts': {
    #                 'target': 1000, #TODO: these are placeholder values
    #                 'min': None,
    #                 'max': None,
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #             'mean_IBI': {
    #                 'target': 1000, #TODO: these are placeholder values
    #                 'min': None,
    #                 'max': None,
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #             'cov_IBI': {
    #                 'target': 1000, #TODO: these are placeholder values
    #                 'min': None,
    #                 'max': None,
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #             'mean_Burst_Peak': {
    #                 'target': 1000, #TODO: these are placeholder values
    #                 'min': None,
    #                 'max': None,
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #             'cov_Burst_Peak': {
    #                 'target': 1000, #TODO: these are placeholder values
    #                 'min': None,
    #                 'max': None,
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #             'fano_factor': {
    #                 'target': 1000, #TODO: these are placeholder values
    #                 'min': None,
    #                 'max': None,
    #                 'weight': 1, #TODO: update these with Nfactors
    #             },
    #         },
    #         #'bursting_data_by_unit': None,
    #     }
    # }
  