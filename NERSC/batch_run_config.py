import os
import pickle
import json
import shutil
import sys
import json

## Logger
import logging
#set up logging
script_dir = os.path.dirname(os.path.realpath(__file__))
log_file = f'{script_dir}/batchRun.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
# Define a logger
logger = logging.getLogger(__name__)

'''
USER INPUT
'''
## Edit the following parameters as needed ##
## Edit params in the batch_run_files folder as needed (with care) ##

## Batch Params ##
method = 'evol' #evolutionary algorithm
# method = 'grid' #grid search

## Selecte a candidate configuration to start the evolution from ##
selected_cand_cfg = None
# selected_cand_cfg = '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5.1_optimizing_EEonly/output/24-3-24_5sec_EEsearch/gen_5/gen_5_cand_29_cfg.json'

## Parallelization Parameters ##
#Cores
#pop_per_core = 4
core_num = 8 #cores/node
#pop_size = pop_per_core * core_num
#NERSC_cores_per_node = 64
pop_size = 4      #128 popsize -> 
                        #8 cpus/task = (4 cores * 2 Threads)/task. 
                        #128 pop/4 nodes = 32 pop/node.
                        #8*cpus/task*32 pop/node = 256 cpus/node  
num_nodes = 4 #keep this value at 1 for now
num_elite_percent = 10/100 # top 10% of the population will be copied to the next generation, this is considered high-medium elitism
num_elites = int(num_elite_percent * pop_size)
#duration_seconds = 5

## Overwrite Parameters ##
overwrite_run = False #True will overwrite any existing run with the same name
continue_run = False #True will continue most recent run
skip = True #True will skip the simulation if it already exists
#make sure batch_run_path does not exist. Avoid overwriting data from any Run
#overwrite_run = True #comment out later, this is for debugging

'''
Initialize
'''

import datetime

# Get current date in YYMMDD format
current_date = datetime.datetime.now().strftime('%y%m%d')

# Edit Run_Name to a unique name for the batch run
try: 
    run_label = sys.argv[1] #batch_run_files folder name
    #print(f'run_name: {run_name}')
except: run_label = 'unnamed_run' ### Change this to a unique name for the batch run

# Prepare Batch_Run_Folder and Initial Files
script_path = os.path.dirname(os.path.realpath(__file__))
output_path = script_path+'/output'

# Get list of existing runs for the day
try: existing_runs = [run for run in os.listdir(output_path) if run.startswith(current_date)]
except: existing_runs = []

# Find the highest run number for the day
if existing_runs:
    highest_run_number = max(int(run.split('_Run')[1].split('_')[0]) for run in existing_runs)
else:
    highest_run_number = 0

# Increment the run number for the new run
new_run_number = highest_run_number + 1
prev_run_number = new_run_number - 1

# Update run_name with new format
run_name = f'{current_date}_Run{new_run_number}_{run_label}'
prev_run_name = f'{current_date}_Run{prev_run_number}_{run_label}'   

# Get unique run path
run_path = f'{output_path}/{run_name}'
prev_run_path = f'{output_path}/{prev_run_name}'

# Mediate Overwrite
if overwrite_run or continue_run:
    if prev_run_name in existing_runs:
        assert not (overwrite_run and continue_run), 'overwrite_run and continue_run cannot both be True'
        # Manage batch_run path
        #if overwrite_run or continue_run:
        if overwrite_run and os.path.exists(prev_run_path):
            shutil.rmtree(prev_run_path)
            run_path = prev_run_path   
            logger.info(f'Overwriting existing batch_run: {os.path.basename(run_path)}')
        elif continue_run and os.path.exists(prev_run_path):
            run_path = prev_run_path
            logger.info(f'Continuing existing batch_run: {os.path.basename(run_path)}')      
        else:
            logger.info(f'Creating new batch_run: {os.path.basename(run_path)}')
            #assert False, 'Run already exists for today. Overwrite and continue options are False. This Error shouldnt happen'

# Create a directory to save the batch_run files
if not os.path.exists(run_path):
    os.makedirs(run_path)

'''
Generate Config
'''
#these updated params are based on the results from the following path:
if selected_cand_cfg is not None: 
    #selected_cand_cfg = 'batch_run_files/evol_params/evol_params_2021-07-07_16-00-00.json'
    with open(selected_cand_cfg, 'r') as f:
        initCfg = json.load(f)
else: initCfg = None

##Specify initFile, cfgFile and netParamsFile paths
initFile = f'{script_path}/init.py'
cfgFile = f'{script_path}/cfg.py'
netParamsFile = f'{script_path}/netParams.py'

assert os.path.exists(initFile), f'initFile does not exist: {initFile}'
assert os.path.exists(cfgFile), f'cfgFile does not exist: {cfgFile}'
assert os.path.exists(netParamsFile), f'netParamsFile does not exist: {netParamsFile}'

##Create a dictionary with the given variables and their values
batch_config_options = {
    "run_path": run_path,
    'batchLabel': os.path.basename(run_path),
    "method": method,
    "core_num": core_num, #256 cpus/node
    "nodes": num_nodes, #4 nodes
    "script": initFile,
    #"pop_per_core": 1,
    #"duration_seconds": 5,
    "pop_size": pop_size,   #128 popsize -> 8 cpus/task = (4 cores * 2 Threads)/task. 
                            #128 pop/4 nodes = 32 pop/node.
                            #8*cpus/task*32 pop/node = 256 cpus/node  
    #"max_generations": 2000,
    "max_generations" : 5,
    "time_sleep": 10, #seconds
    "maxiter_wait": 6*20, #15 minutes per gen?
    "skip": skip,
    "num_elites": num_elites,
    "cfgFile": cfgFile,
    "netParamsFile": netParamsFile,
    "initCfg": initCfg,
}

# # Write the dictionary to a JSON file
# with open("batch_config_options.json", "w") as f:
#     json.dump(batch_config_options, f)