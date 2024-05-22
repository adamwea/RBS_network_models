'''imports'''
import os
import netpyne
import subprocess
from fitness_functions import fitnessFunc
from fitness_config import *
import pandas as pd
import json
import os
from reportlab.pdfgen import canvas
from PIL import Image
import sys, os
import datetime
#from USER_INPUTS import *
from reportlab.lib.pagesizes import letter
import numpy as np

'''functions'''
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
def calc_fitness(simData, fitness_save_path):
    avgScaledFitness = fitnessFunc(simData, plot = True, data_file_path = None, fitness_save_path = fitness_save_path, exp_mode = True, **kwargs)
    return avgScaledFitness
def create_simulated_sim_obj(exp_data_files):       
    elite_paths = {}        
    if verbose == True: enablePrint()
    for file in exp_data_files:
        #get absolute path
        file = os.path.abspath(file)
        print(file)
        if '.xlsx' in file:
            xlsx_file = file
        if '.npz' in file:
            npz_file = file    
    fitness_save_path = os.path.dirname(xlsx_file)
    xlsx_data = pd.read_excel(xlsx_file)
    avgRate = xlsx_data['firing_rate'].values
    avgRate = np.nanmean(avgRate)
    #get all rows where firing rate is in the top 30% of all firing rates
    I_cells = xlsx_data[xlsx_data['firing_rate'] > xlsx_data['firing_rate'].quantile(0.7)]
    #get the remaining rows
    E_cells = xlsx_data[xlsx_data['firing_rate'] <= xlsx_data['firing_rate'].quantile(0.7)]
    #get the avg firing rate of both I and E cells
    avgRate_I = I_cells['firing_rate'].values
    avgRate_E = E_cells['firing_rate'].values
    avgRate_I = np.nanmean(avgRate_I)
    avgRate_E = np.nanmean(avgRate_E)
    real_spike_data = npz_file
    #load npz file
    real_spike_data = np.load(real_spike_data, allow_pickle = True)
    simData = real_spike_data['spike_array']
    # this data is in the form of a 2D array, where each row is a neuron and each column is boolean for whether or not the neuron spiked
    # we need to convert this into a dictionary with time domain 't', spike times 'spkt', and spike indices 'spkid'
    # i.e. if multiple neurons fire at the same time, 'spkt' will have multiple indices in sequence with the same time value
    # we will also need to convert the time domain to ms: time is samples at 10kHz, so 1s = 10,000 samples
    #flatten 2D array to 1D array, where each index counts spikes at that time
    # Assuming simData is a 2D NumPy array
    spikes_per_time = np.sum(simData, axis=0)
    #bin the array from 3million samples to 300
    binned_spikes = np.array([spikes_per_time[i:i+1000].sum() for i in range(0, len(spikes_per_time), 1000)])    
    # Assuming simData is a 2D NumPy array
    time_indices = np.where(spikes_per_time > 0)
    time_indices = time_indices[0]
    #create array spkt, where each index is the time of a spike
    
    spkt = np.array([i for i in time_indices for j in range(spikes_per_time[i])])

    # Create an array representing all times in simData
    all_times = np.arange(simData.shape[1])
    spike_dict = {
        'avgRate': avgRate,
        'popRates': {'E': avgRate_E, 'I': avgRate_I},
        'soma_volatage': None, #TODO: add method to get soma voltage  
        'spkid': np.arange(len(spkt)),  # Generate a sequence of spike IDs
        'spkt': spkt / 10000,  # Convert the time indices to ms
        't': all_times / 10000,        
    }
    #Sort spkt min to max
    sorted_indices = np.argsort(spike_dict['spkt'])
    spike_dict['spkt'] = spike_dict['spkt'][sorted_indices]
    
    simData = spike_dict

    return simData
def fit_exp_data(exp_dir):

    #job_dir = os.path.abspath(job_dir)
    exp_data_files = [f.path for f in os.scandir(exp_dir)]
    #assert that there are 2 files, one is an xlsx file, the other is a .npz file
    #assert len(exp_data_files) == 2, f"Error: {exp_dir} does not contain 2 files."
    xlsx_file = None
    npz_file = None

    for file in exp_data_files:
        if '.xlsx' in file:
            xlsx_file = file
        elif '.npz' in file:
            npz_file = file

    assert xlsx_file is not None, "Error: No xlsx file found."
    #assert npz_file is not None, "Error: No npz file found."

    simData = create_simulated_sim_obj(exp_data_files)
    fitness_save_path = os.path.abspath(xlsx_file)
    fitness_save_path = os.path.dirname(xlsx_file)
    avgScaledFitness = calc_fitness(simData, fitness_save_path = fitness_save_path)                            

if __name__ == '__main__':
    kwargs = fitnessFuncArgs
    
    #surpress all print statements
    blockPrint()
    
    #set to True to print verbose output
    verbose = True

    exp_dirs = [
        './BigData/wt',
        ]
        
    #run plot_elites    
    for exp_dir in exp_dirs:
        try: fit_exp_data(exp_dir)
        except Exception as e: 
            enablePrint()
            print(f'An error occurred while plotting {os.path.basename(exp_dir)}')
            print(f"Error: {e}")
            blockPrint()
            
            