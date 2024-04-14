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

# ## Edit Run_Name to a unique name for the batch run ##
# try: run_name = sys.argv[1] #batch_run_files folder name
# except: run_name = 'NERSC_Test' ### Change this to a unique name for the batch run

## Batch Params ##
method = 'evol' #evolutionary algorithm
# method = 'grid' #grid search

## Selecte a candidate configuration to start the evolution from ##
selected_cand_cfg = None
# selected_cand_cfg = '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5.1_optimizing_EEonly/output/24-3-24_5sec_EEsearch/gen_5/gen_5_cand_29_cfg.json'

## Parallelization Parameters ##
pop_per_core = 1
core_num = 4
num_nodes = 1
pop_size = pop_per_core * core_num
num_elite_percent = 10/100 # top 10% of the population will be copied to the next generation, this is considered high-medium elitism
num_elites = int(num_elite_percent * pop_size)
#duration_seconds = 5

## Overwrite Parameters ##
overwrite_run = True #True will overwrite any existing run with the same name
continue_run = False #True will continue most recent run
skip = True #True will skip the simulation if it already exists
#make sure batch_run_path does not exist. Avoid overwriting data from any Run
#overwrite_run = True #comment out later, this is for debugging

'''
Initialize
'''
##Prepare Batch_Run_Folder and Initial Files
#output_path = nb_path.replace('batch_run_files', 'output')'
script_path = os.path.dirname(os.path.realpath(__file__))
output_path = script_path+'/output'
#get unique run path
i = 0
first_run = True
unique_run_path = f'{output_path}/run{i}_{run_name}'
while os.path.exists(unique_run_path):
    first_run = False
    i += 1
    unique_run_path = f'{output_path}/run{i}_{run_name}'

## Mediate Overwrite
if first_run is False:
    assert not (overwrite_run and continue_run), 'overwrite_run and continue_run cannot both be True'
    #Manage batch_run path
    if overwrite_run or continue_run:
        i = i-1
        run_path = f'{output_path}/run{i}_{run_name}'
    else:
        run_path = unique_run_path    
    #if continue-run is True, skip must be True
    if continue_run is True:
        assert skip is True, 'skip must be True if continue_run is True'
        logger.info(f'Continuing most recent batch_run: {os.path.basename(run_path)}')    
    #if overwrite-run is True, delete the existing run_path
    if overwrite_run and os.path.exists(run_path):
        #assert skip is False, 'skip must be False if overwrite_run is True'
        shutil.rmtree(run_path)   
        logger.info(f'Overwriting existing batch_run: {os.path.basename(run_path)}')      
else:
    run_path = unique_run_path

## Create a directory to save the batch_run files
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
    "core_num": core_num,
    "nodes": num_nodes,
    "script": initFile,
    #"pop_per_core": 1,
    #"duration_seconds": 5,
    "pop_size": pop_size,
    "max_generations": 2000,
    "time_sleep": 10, #seconds
    "maxiter_wait": 6*40, #6itersx10sec/itersx30sec/min = 30 minutes per gen?
    "skip": skip,
    "num_elites": num_elites,
    "cfgFile": cfgFile,
    "netParamsFile": netParamsFile,
    "initCfg": initCfg,
}

# # Write the dictionary to a JSON file
# with open("batch_config_options.json", "w") as f:
#     json.dump(batch_config_options, f)