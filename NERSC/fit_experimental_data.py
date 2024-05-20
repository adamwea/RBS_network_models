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

'''functions'''
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
def calc_fitness(data_file_path):
    #assert that data_file_path exists, if it doesnt, probably wokring with locally instead of on NERSC
    try: assert os.path.exists(data_file_path), f"Error: {data_file_path} does not exist."
    except: return None, None, None, None, None
    
    print(f"Recalculating Fitness for {os.path.basename(data_file_path)}")
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
        print(file)    
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
    assert len(exp_data_files) == 2, f"Error: {exp_dir} does not contain 2 files."
    xlsx_file = None
    npz_file = None

    for file in exp_data_files:
        if '.xlsx' in file:
            xlsx_file = file
        elif '.npz' in file:
            npz_file = file

    assert xlsx_file is not None, "Error: No xlsx file found."
    assert npz_file is not None, "Error: No npz file found."

    create_simulated_sim_obj(exp_data_files)                              

if __name__ == '__main__':
    kwargs = fitnessFuncArgs
    
    #surpress all print statements
    blockPrint()
    
    #set to True to print verbose output
    verbose = True

    exp_dirs = [
        '/pscratch/sd/a/adammwea/2DNetworkSimulations/BigData/wt',
        ]
        
    #run plot_elites    
    for exp_dir in exp_dirs:
        try: fit_exp_data(exp_dir)
        except Exception as e: 
            enablePrint()
            print(f'An error occurred while plotting {os.path.basename(exp_dir)}')
            print(f"Error: {e}")
            blockPrint()
            
            