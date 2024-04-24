import os
import netpyne
import subprocess
from fitness_functions import fitnessFunc
from fitness_config import *
kwargs = fitnessFuncArgs
#from USER_INPUTS import *

#get output directory
output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')

import pandas as pd
import json

# Initialize an empty DataFrame
df = pd.DataFrame()

#os walk through output path and find all .json files with containing fitness in the name
for root, dirs, files in os.walk(output_path):
    for file in files:
        if '.json' in file and 'Fitness' in file:
            #create corresponding data file path
            fit_file_path = os.path.join(root, file)
            data_file_path = os.path.join(root, file.replace('_Fitness', '_data'))
            
            #check if data file_path exists
            if os.path.exists(data_file_path): pass
            else: continue

            #load the data file using netpyne loadall
            netpyne.sim.loadAll(data_file_path)
            simData = netpyne.sim.allSimData
            batch_saveFolder = netpyne.sim.cfg.saveFolder
            #get cwd
            cwd = os.getcwd()
            cwd_basename = os.path.basename(cwd)
            #make bach_saveFolder relative after basename in batch_saveFolder
            batch_saveFolder = f'{cwd}{batch_saveFolder.split(cwd_basename)[1]}'
            #batch_saveFolder = batch_saveFolder[1:] #remove leading '/'
            simLabel = netpyne.sim.cfg.simLabel
            fitness_save_path = os.path.dirname(data_file_path)
            avgScaledFitness = fitnessFunc(
                simData, plot = False, simLabel = simLabel, batch_saveFolder = batch_saveFolder, 
                fitness_save_path = None, **kwargs)
            if avgScaledFitness < 500:
                avgScaledFitness = fitnessFunc(
                simData, plot = True, simLabel = simLabel, batch_saveFolder = batch_saveFolder, 
                fitness_save_path = fitness_save_path, **kwargs)
                print('Plots saved to: ...')
            #print('Average Scaled Fitness:', avgScaledFitness)

            # ## get info from .csv
            # indiv_csv_dir = os.path.dirname(os.path.dirname(data_file_path))
            # for root, dirs, files in os.walk(indiv_csv_dir):
            #     for file in files:
            #         if 'indiv.csv' in file:
            #             csv_file_path = os.path.join(root, file)
            #             # Read the CSV file line by line
            #             with open(csv_file_path, 'r') as f:
            #                 for line in f:
            #                     #get first line to define columns
            #                     if 'gen' in line:
            #                         columns = line.split(',')
            #                         df = pd.DataFrame(columns=columns)
            #                         print('Columns:', columns)
            #                     # Check if avgScaledFitness is in the line
            #                     if str(avgScaledFitness) in line:
            #                         print('Found line with average scaled fitness:', line)
            #                         # Convert the line to a DataFrame and append it to df
            #                         #line_df = pd.DataFrame([line.split(',')])
            #                         line_df = pd.DataFrame([line.split(',')], columns=columns)
            #                         #df = df.append(line_df, ignore_index=True)
            #                         # # Extend row with fitness values at data at fit_file_path, name new columns by .json key names
            #                         # with open(fit_file_path, 'r') as f:
            #                         #     fitness_data = json.load(f)
            #                         #     for key, value in fitness_data.items():
            #                         #         df[key] = value
            #                         # break

            # # Save the DataFrame to a CSV file
            # #plots_path = '/home/adamm/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/plots'
            # #df.to_csv(os.path.join(plots_path, 'fitness_data.csv'), index=False)
            # print('Saved fitness data to:', os.path.join(indiv_csv_dir, 'fitness_data.csv'))
            
            