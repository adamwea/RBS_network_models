'''
Collection of useful functions written for batch processing of the HD-MEA simulations using NetPyNE
'''

#imports
import os
from netpyne import sim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks
from scipy.stats import norm
#import svgutils.transform as sg
from scipy.signal import butter, filtfilt
from scipy import stats
#import cairosvg

### Batch Functions
import os
import shutil

# (I dont understand how this works, figure it out later)
import inspect

import netpyne

'''helper functions'''
def load_clean_sim_object(data_file_path):
        # #netpyne.sim.initialize() #initialize netpyne
        # print('Loading data from:', data_file_path)
        # netpyne.sim.loadAll(data_file_path)
        print('clearing all data')
        try: netpyne.sim.clearAll() #clear all sim data
        except: pass
        print('loading all data')
        netpyne.sim.loadAll(data_file_path)
        #print('test concluded')
   

def get_walltime_per_sim(USER_walltime_per_gen, USER_pop_size, USER_nodes):
    USER_walltime_per_gen_hours = int(USER_walltime_per_gen.split(':')[0])
    USER_walltime_per_gen_hours = USER_walltime_per_gen_hours / USER_nodes
    USER_walltime_per_gen_minutes = int(USER_walltime_per_gen.split(':')[1])
    USER_walltime_per_gen_minutes = USER_walltime_per_gen_minutes / USER_nodes
    USER_walltime_per_gen_seconds = int(USER_walltime_per_gen.split(':')[2])
    USER_walltime_per_gen_seconds = USER_walltime_per_gen_seconds / USER_nodes
    
    USER_walltime_per_gen_seconds += USER_walltime_per_gen_minutes*60
    USER_walltime_per_gen_seconds += USER_walltime_per_gen_hours*3600
    USER_walltime_per_sim_seconds = USER_walltime_per_gen_seconds/USER_pop_size

    # Convert back to hh:mm:ss
    hours, remainder = divmod(USER_walltime_per_sim_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

def find_batch_object_and_sim_label():
    # Get the current frame
    current_frame = inspect.currentframe()  
    # Iterate through the call stack
    while current_frame:
        caller_frame = inspect.getouterframes(current_frame, 3)#[1][0]
        # Check each local variable in the frame
        #for name, obj in list(current_frame.f_locals.items())[::-1]:
        for name, obj in list(caller_frame[1][0].f_locals.items())[::-1]:
            # If the object is of type Batch, return it and its simLabel
            if name == '_':
                simLabel = caller_frame[1][0].f_locals['_']
                batch = caller_frame[2][0].f_locals['batch']
                return batch, simLabel
        # If not, move to the next frame
        current_frame = current_frame.f_back
    # If no Batch object is found, print a message and return None
    print("Batch object not found in the caller frames.")
    return None, None

def move_btr_files():
    try:
        # List all files in the current working directory
        files_in_cwd = os.listdir('.')
        # Filter out the .btr files
        btr_files = [file for file in files_in_cwd if file.endswith('.btr')]
        
        # Check if there are any .btr files to move
        if not btr_files:
            print("No .btr files found.")
            return
        
        # Create a subfolder for .btr files if it doesn't exist
        btr_folder = 'btr files'
        if not os.path.exists(btr_folder):
            os.makedirs(btr_folder)
        
        # Move each .btr file to the subfolder
        for btr_file in btr_files:
            shutil.move(btr_file, os.path.join(btr_folder, btr_file))
        
        print(".btr files successfully moved.")
        
    except Exception as e:
        print("Failed to move files:", e)

def get_batchrun_info(root, file):
    file_path = os.path.join(root, file)
    batchrun_folder = os.path.basename(os.path.normpath(root))    
    batch_key = file.split(batchrun_folder)[1].split('data')[0]
    #hot fix for evol_batchruns
    if "gen_" in batchrun_folder and batch_key == '/':
        batch_key = file.split(f"{batchrun_folder}/")[1].split('data')[0]
    return file_path, batchrun_folder, batch_key
