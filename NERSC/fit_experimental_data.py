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
def calc_fitness(data_file_path):
    #assert that data_file_path exists, if it doesnt, probably wokring with locally instead of on NERSC
    try: assert os.path.exists(data_file_path), f"Error: {data_file_path} does not exist."
    except: return None, None, None, None, None
    
    print(f"Calculating Fitness for {os.path.basename(data_file_path)}")
    #load the data file using netpyne loadall
    netpyne.sim.loadAll(data_file_path)
    simData = netpyne.sim.allSimData
    batch_saveFolder = netpyne.sim.cfg.saveFolder
    simLabel = netpyne.sim.cfg.simLabel

    #pathing
    NERSC_path = os.path.dirname(os.path.realpath(__file__))
    repo_path = os.path.dirname(NERSC_path)
    cwd = repo_path
    cwd_basename = os.path.basename(cwd)
    batch_saveFolder = f'{cwd}{batch_saveFolder.split(cwd_basename)[1]}'                        
    fitness_save_path = os.path.dirname(data_file_path)

    #measure fitness
    avgScaledFitness = fitnessFunc(
        simData, plot = False, simLabel = simLabel, 
        data_file_path = data_file_path, batch_saveFolder = batch_saveFolder, 
        fitness_save_path = fitness_save_path, **kwargs)
    
    return avgScaledFitness, simData, batch_saveFolder, fitness_save_path, simLabel
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
    data_file_path = xlsx_file
    real_spike_data = npz_file
    #load npz file
    real_spike_data = np.load(real_spike_data, allow_pickle = True)
    simData = real_spike_data['spike_array']
    # this data is in the form of a 2D array, where each row is a neuron and each column is boolean for whether or not the neuron spiked
    # we need to convert this into a dictionary with time domain 't', spike times 'spkt', and spike indices 'spkid'
    # i.e. if multiple neurons fire at the same time, 'spkt' will have multiple indices in sequence with the same time value
    # we will also need to convert the time domain to ms: time is samples at 10kHz, so 1s = 10,000 samples
    # Initialize the dictionary
    spike_dict = {'t': [], 'spkt': [], 'spkid': []}

    # Convert the 2D array to the dictionary format
    spike_id = 0
    for neuron_index, neuron_data in enumerate(simData):
        for time_index, spike in enumerate(neuron_data):
            spike_dict['t'].append(time_index)  # Add the time index to 't'
            if spike==1:  # If the neuron spiked at this time
                time_ms = time_index/1000  # Convert the time index to ms  
                spike_dict['spkid'].append(spike_id)  # Add the neuron index to 'spkid'              
                spike_dict['spkt'].append(time_ms)  # Add the time index to 'spkt'
                spike_id += 1  # Increment the spike id


    avgScaledFitness = fitnessFunc(simData, plot = False, data_file_path = data_file_path, fitness_save_path = fitness_save_path, exp_mode = True, **kwargs)
    sys.exit()

    #create corresponding data file path
    fit_file_path = os.path.join(root, file.replace('_data', '_Fitness'))
    data_file_path = os.path.join(root, file)
    
    #skip if gen_dir is less than start_gen
    if cand is not None:
        if int(data_file_path.split('_')[-2]) != cand: 
            enablePrint()
            print(f"Skipping {os.path.basename(data_file_path)}")
            blockPrint()
            #continue

        # enablePrint()    
        # print(data_file_path)
        # blockPrint()
        # if verbose == True: enablePrint()

        #temp
        #if '.archive' in data_file_path: continue
        
        #check if data file_path exists
        if os.path.exists(data_file_path): pass
        #else: continue

        try: 
            avgScaledFitness, simData, batch_saveFolder, fitness_save_path, simLabel = recalc_fitness(data_file_path)
            elite_paths[simLabel] = {'avgScaledFitness': avgScaledFitness, 
                                        'data_file_path': data_file_path,
                                        'batch_saveFolder': batch_saveFolder,
                                        'fitness_save_path': fitness_save_path,
                                        'simData': simData,
                                        }
        except: pass
    return elite_paths
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

    create_simulated_sim_obj(exp_data_files)                              

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
            
            