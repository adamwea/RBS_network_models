from simulate._temp_files.temp_user_args import *
import numpy as np
from matplotlib import pyplot as plt
#import netpyne
#import json
#from helper_functions import load_clean_sim_object
#import os

def plot_experimental_raster(data_path, save_fig=None, sampling_rate=10000, xlim_start=30000, xlim_end=300000, figsize=None, dpi=600):
    """
    Plot a raster plot of spike trains with the top 30% of firing neurons in gold and the rest in light blue.
    
    Parameters:
    - real_spike_data: List of lists or numpy array, each inner list/array represents a spike train.
    - sampling_rate: Sampling rate in Hz, default is 10000.
    - xlim_start: Start of the x-axis in milliseconds, default is 30000 (30 seconds).
    - xlim_end: End of the x-axis in milliseconds, default is 300000 (300 seconds).
    """

    real_spike_data=np.load(data_path, allow_pickle = True)
    real_spike_data=real_spike_data['spike_array']
    print('data loaded')
    
    # Ensure real_spike_data is a numpy array
    real_spike_data = np.array(real_spike_data)
    
    # Sort the spike trains by firing rate, highest to lowest
    real_spike_data = sorted(real_spike_data, key=lambda x: np.sum(x), reverse=False)
    
    # Convert the sorted list back to a numpy array for plotting
    real_spike_data = np.array(real_spike_data)
    
    # Plot raster plot of spike trains
    if figsize is None: figsize = (20, 10)
    else: pass #if figsize is provided, use it
    fig, ax = plt.subplots(figsize=figsize)
    
    # Loop through each spike train and plot the spikes
    for neuron_id, spike_train in enumerate(real_spike_data):
        spike_times = np.where(spike_train == 1)[0] / sampling_rate #* 1000  # Convert to milliseconds
        color = '#ADD8E6'
        ax.vlines(spike_times, neuron_id + 0.5, neuron_id + 1.5, color=color)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('IPNs')
    ax.set_title('Spike Raster Plot')
    ax.set_ylim(0, len(real_spike_data) + 0.5)  # Ensure correct y-axis direction
    ax.set_xlim(xlim_start, xlim_end)  # Set x-axis limits based on parameters
    
    raster_fig= plt.figure()
    return raster_fig