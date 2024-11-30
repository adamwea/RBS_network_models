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
import modules.mea_processing_library as mea_lib
import numpy as np
from modules.analysis.analyze_network_activity import get_experimental_network_activity_metrics
import pickle
import dill
import json

import os
import h5py
import logging

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

def get_base_path(recording_path):
    return os.path.dirname(recording_path).replace('xRBS_input_data', 'yRBS_spike_sorted_data')

def get_presumed_sorting_output_path(recording_path, stream):
    base_path = get_base_path(recording_path)
    #paths = []
    #for i in range(6):
    path = os.path.join(base_path, f"well{str(stream).zfill(3)}/sortings/sorter_output")
    # if os.path.exists(path):
    #     paths.append(path)
    return path

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

def convert_keys_to_str(data):
    if isinstance(data, dict):
        return {str(k): convert_keys_to_str(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys_to_str(i) for i in data]
    else:
        return data

def save_network_data_pkl(recordings, stream_select, network_metrics_output_dir):
    for recording_path in recordings:
        stream_nums = [0, 1, 2, 3, 4, 5]
        if stream_select is not None: 
            stream_nums = [stream_select]
        for stream_num in stream_nums:    
            # Load recordings and sorting data
            MaxID, recording_object, expected_well_count, rec_counts = mea_lib.load_recordings(recording_path, stream_select=stream_select)
            sorting_output_path = get_presumed_sorting_output_path(recording_path, stream_select)
            stream_sorting_object = load_spike_sorted_data(sorting_output_path)
            
            # Get recording segment - note: network scans should only have one recording segment
            well_id = f'well{str(0).zfill(2)}{stream_select}'
            network_recording_segment = recording_object[well_id]['recording_segments'][0]  # Note: network scans should only have one recording segment
            
            # Build results dictionary
            kwargs = {
                'recording_object': network_recording_segment,
                'sorting_object': stream_sorting_object,
            }
            
            # Get network activity metrics
            network_metrics = get_experimental_network_activity_metrics(**kwargs)
            
            # Ensure the output directory exists
            os.makedirs(network_metrics_output_dir, exist_ok=True)
            
            # Save network metrics as pickle
            print("Saving network metrics as pickle...")
            save_path = os.path.join(network_metrics_output_dir, f"network_metrics_well00{stream_num}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(network_metrics, f)
            print(f"Saved network metrics to {save_path}")
            
            # Save network metrics as numpy file
            print("Saving network metrics as numpy...")
            save_path = os.path.join(network_metrics_output_dir, f"network_metrics_well00{stream_num}.npy")
            np.save(save_path, network_metrics)
            print(f"Saved network metrics to {save_path}")
            
def check_hdf5_plugin(h5_full_path):
    plugin_path = '/opt/maxwell_hdf5_plugin/Linux'
    
    # Check if the plugin directory exists
    if not os.path.isdir(plugin_path):
        logger.error("Error: HDF5 plugin directory is not available in the environment.")
        raise Exception("Error: HDF5 plugin directory is not available in the environment.")
    
    # #check if the plugin is loaded
    # plugin_loaded = False
    # for plugin in h5py.get_config().plugin_path:
    #     if plugin == plugin_path:
    #         plugin_loaded = True
    #         break

    # if not plugin_loaded:
    #     logger.error("Error: HDF5 plugin is not loaded in the Python environment.")
    #     raise Exception("Error: HDF5 plugin is not loaded in the Python environment.")
    
    #check python os environment
    if 'HDF5_PLUGIN_PATH' not in os.environ:
        logger.error("Error: HDF5_PLUGIN_PATH is not set in the Python environment.")
        raise Exception("Error: HDF5_PLUGIN_PATH is not set in the Python environment.")
    else:
        print(f"HDF5_PLUGIN_PATH: {os.environ['HDF5_PLUGIN_PATH']}")   
    
    
    # # Check if the file is an HDF5 file
    # if not h5py.is_hdf5(h5_full_path):
    #     logger.error("Error: File is not an HDF5 file.")
    #     raise Exception("Error: File is not an HDF5 file.")

def generate_network_bursting_plot(network_data, bursting_plot_path, bursting_fig_path):
    """Generate and save the network bursting plot."""
    #fig = network_data['bursting_data']['bursting_summary_data']['fig']
    ax = network_data['bursting_data']['bursting_summary_data']['ax']
    ax_old = ax
    
    # Create a new figure with shared x-axis
    fig, ax = plt.subplots(figsize=(16, 4.5))
    
    #copy ax features to new ax
    ax.set_xlim(ax_old.get_xlim())
    ax.set_ylim(ax_old.get_ylim())
    #ax.set_ylabel(ax_old.get_ylabel())
    ax.set_ylabel('Firing Rate (Hz)')
    #ax.set_xlabel(ax_old.get_xlabel())
    ax.set_xlabel('Time (s)')
    ax.set_title(ax_old.get_title())
    #ax.plot(ax_old.get_lines()[0].get_xdata(), ax_old.get_lines()[0].get_ydata(), color='royalblue')
    ax.plot(ax_old.get_lines()[0].get_xdata(), ax_old.get_lines()[0].get_ydata(), color='royalblue')
    ax.plot(ax_old.get_lines()[1].get_xdata(), ax_old.get_lines()[1].get_ydata(), 'or')
    

    fig.savefig(bursting_plot_path) #save as pdf
    print(f"Bursting plot saved to {bursting_plot_path}")

    # Save bursting plot as pickle
    with open(bursting_fig_path, 'wb') as f:
        pickle.dump((fig, ax), f)
    print(f"Bursting plot data saved to {bursting_fig_path}")

def plot_raster_plot_experimental(ax, spiking_data_by_unit):
    """Plot a raster plot for spiking data."""
    
    # Calculate the average firing rate for each unit
    firing_rates = {}
    for gid in spiking_data_by_unit:
        spike_times = spiking_data_by_unit[gid]['spike_times']
        spike_times = [spike_times] if isinstance(spike_times, (int, float)) else spike_times
        firing_rate = len(spike_times) / (max(spike_times) - min(spike_times)) if len(spike_times) > 1 else 0
        firing_rates[gid] = firing_rate
    
    # Sort the units based on their average firing rates
    sorted_units = sorted(firing_rates, key=firing_rates.get)
    
    # Create a mapping from original gid to new y-axis position
    gid_to_ypos = {gid: pos for pos, gid in enumerate(sorted_units)}
    
    # Plot the units in the sorted order
    for gid in sorted_units:
        spike_times = spiking_data_by_unit[gid]['spike_times']
        spike_times = [spike_times] if isinstance(spike_times, (int, float)) else spike_times
        ax.plot(spike_times, [gid_to_ypos[gid]] * len(spike_times), 'b.', markersize=2)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Unit ID (sorted by firing rate)')
    ax.set_title('Raster Plot')
    plt.tight_layout()
    return ax

def generate_raster_plot_experimental(network_data, raster_plot_path, raster_fig_path):
    """Generate and save the raster plot."""
    spiking_data_by_unit = network_data['spiking_data']['spiking_data_by_unit']

    fig, ax = plt.subplots(figsize=(16, 4.5))

    ax = plot_raster_plot_experimental(ax, spiking_data_by_unit)
    fig.savefig(raster_plot_path) #save as pdf
    print(f"Raster plot saved to {raster_plot_path}")

    # Save raster plot as pickle
    with open(raster_fig_path, 'wb') as f:
        pickle.dump((fig, ax), f)
    print(f"Raster plot data saved to {raster_fig_path}")

if __name__ == '__main__':
    
    '''Args'''
    stream_select = None  # Set to a list of well indices if needed, e.g., [0, 2, 4]
    stream_select = 0

    '''Recordings'''
    recordings = [
        #"/pscratch/sd/a/adammwea/xInputs/xRBS_input_data/CDKL5-E6D_T2_C1_05212024/240611/M06844/Network/000076/data.raw.h5",
        "/pscratch/sd/a/adammwea/workspace/xInputs/xRBS_input_data/CDKL5-E6D_T2_C1_05212024/240611/M06844/Network/000076/data.raw.h5",
    ]
    
    '''Save network metrics'''
    switch = False
    if switch:
        check_hdf5_plugin(recordings[0])
        network_metrics_output_dir = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/1_derive_features_from_experimental_data/network_metrics'
        save_network_data_pkl(recordings, stream_select, network_metrics_output_dir)
    
    
    '''load network metrics, save ax for followup plotting'''
    switch = True
    if switch:
        #load network metrics
        #network_metrics_path = 'workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/1_derive_features_from_experimental_data/network_metrics/network_metrics_well000.npy'
        network_metrics_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_1_derive_features_from_experimental_data/network_metrics/network_metrics_well000.npy'
        network_metrics = np.load(network_metrics_path, allow_pickle=True).item()
        
        #raster
        #raster_plot_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/1_derive_features_from_experimental_data/network_metrics/network_metrics_well000_raster_plot.png'
        #raster_fig_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/1_derive_features_from_experimental_data/network_metrics/network_metrics_well000_raster_fig.pkl'
        raster_plot_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_1_derive_features_from_experimental_data/network_metrics/network_metrics_well000_raster_plot.png'
        raster_fig_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_1_derive_features_from_experimental_data/network_metrics/network_metrics_well000_raster_fig.pkl'
        generate_raster_plot_experimental(network_metrics, raster_plot_path, raster_fig_path)
        
        #bursting
        #bursting_plot_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/1_derive_features_from_experimental_data/network_metrics/network_metrics_well000_bursting_plot.png'
        #bursting_fig_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/1_derive_features_from_experimental_data/network_metrics/network_metrics_well000_bursting_fig.pkl'
        bursting_plot_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_1_derive_features_from_experimental_data/network_metrics/network_metrics_well000_bursting_plot.png'
        bursting_fig_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_1_derive_features_from_experimental_data/network_metrics/network_metrics_well000_bursting_fig.pkl'
        generate_network_bursting_plot(network_metrics, bursting_plot_path, bursting_fig_path)
               