import os
import pickle
import json
import shutil
import sys
import json
import datetime

## Logger
import logging
script_dir = os.path.dirname(os.path.realpath(__file__))
log_file = f'{script_dir}/batchRun.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

'''
USER INPUT
'''
from USER_INPUTS import *
## Batch Params ##
assert USER_method, 'method_user must be specified in USER_INPUTS.py'
method = USER_method #evolutionary algorithm

## Selecte a candidate configuration to start the evolution from ##
selected_cand_cfg = None
# selected_cand_cfg = '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5.1_optimizing_EEonly/output/24-3-24_5sec_EEsearch/gen_5/gen_5_cand_29_cfg.json'

## Parallelization Parameters ##
assert USER_cores_per_node, 'USER_cores_per_node must be specified in USER_INPUTS.py'
cores_per_node = USER_cores_per_node #cores/node
assert USER_pop_size, 'USER_pop_size must be specified in USER_INPUTS.py'
pop_size = USER_pop_size
assert USER_nodes, 'USER_nodes must be specified in USER_INPUTS.py'
num_nodes = USER_nodes #keep this value at 1 for now

## Genetic Algorithm Parameters ##
assert USER_frac_elites, 'USER_frac_elites must be specified in USER_INPUTS.py'
num_elite_percent = 100*USER_frac_elites # top 10% of the population will be copied to the next generation, this is considered high-medium elitism
num_elites = int(num_elite_percent * pop_size)
try: assert num_elites > 0, "num_elites rounded to 0. num_elites must be greater than 0. Setting num_elites = 1."
except: num_elites = 1
assert USER_max_generations, 'USER_max_generations must be specified in USER_INPUTS.py'
max_generations = USER_max_generations
assert USER_time_sleep, 'USER_time_sleep must be specified in USER_INPUTS.py'
time_sleep = USER_time_sleep
assert USER_maxiter_wait_minutes, 'USER_maxiter_wait_minutes must be specified in USER_INPUTS.py'
maxiter_wait = USER_maxiter_wait_minutes*60/time_sleep

## Overwrite Parameters ##
assert USER_overwrite_run is not None, 'USER_overwrite_run must be specified in USER_INPUTS.py'
assert USER_continue_run is not None, 'USER_continue_run must be specified in USER_INPUTS.py'
overwrite_run = USER_overwrite_run #True will overwrite any existing run with the same name
continue_run = USER_continue_run #True will continue most recent run
skip = False #True will skip the simulation if it already exists
if continue_run: skip = True

# Edit Run_Name to a unique name for the batch run
assert USER_run_label, 'USER_run_label must be specified in USER_INPUTS.py'
run_label = USER_run_label #batch_run_files folder name

'''
Initialize
'''
# from mpi4py import MPI
# # Initialize the MPI environment
# comm = MPI.COMM_WORLD
# # Get the rank of the current process
# rank = comm.Get_rank()
# # Finalize the MPI environment
# MPI.Finalize()
# print("Rank:", rank)
import time
# Get the rank of the current process
rank = os.environ.get('OMPI_COMM_WORLD_RANK')

# Check if the environment variable is set
if rank is not None:
    rank = int(rank)  # Convert the rank to an integer
    logger.info(f"I am process: {rank}")
else:
    logger.info("OMPI_COMM_WORLD_RANK is not set")
    rank = 0

# Get current date in YYMMDD format
current_date = datetime.datetime.now().strftime('%y%m%d')
# Prepare Batch_Run_Folder and Initial Files
script_path = os.path.dirname(os.path.realpath(__file__))
output_path = script_path+'/output'

# Get list of existing runs for the day
try: existing_runs = [run for run in os.listdir(output_path) if run.startswith(current_date)]
except: existing_runs = []

# Find the highest run number for the day
if existing_runs: highest_run_number = max(int(run.split('_Run')[1].split('_')[0]) for run in existing_runs)
else: highest_run_number = 0

# Increment the run number for the new run
new_run_number = highest_run_number + 1
prev_run_number = new_run_number - 1

# Update run_name with new format
run_name = f'{current_date}_Run{new_run_number}_{run_label}'
prev_run_name = f'{current_date}_Run{prev_run_number}_{run_label}'
#print(f'Run Name: {run_name}')
#sys.exit()   

# Get unique run path
run_path = f'{output_path}/{run_name}'
prev_run_path = f'{output_path}/{prev_run_name}'

#logger.info(f'Rank: {rank}')
#logger.info(f'Run Path: {run_path}')
#logger.info(f'Previous Run Path: {prev_run_path}')
# Mediate Overwrite
if overwrite_run or continue_run:
    if prev_run_name in existing_runs:
        assert not (overwrite_run and continue_run), 'overwrite_run and continue_run cannot both be True'
        # Manage batch_run path
        # if USER_MPI_run_keep and os.path.exists(prev_run_path):
        #     #shutil.rmtree(prev_run_path)
        #     run_path = prev_run_path
        #     logger.info('MPI')   
        #     #logger.info(f'Overwriting existing batch_run: {os.path.basename(run_path)}')
        if overwrite_run and os.path.exists(prev_run_path):
            shutil.rmtree(prev_run_path)
            run_path = prev_run_path   
            logger.info(f'Overwriting existing batch_run: {os.path.basename(run_path)}')
        elif continue_run and os.path.exists(prev_run_path):
            run_path = prev_run_path
            logger.info(f'Continuing existing batch_run: {os.path.basename(run_path)}')      
else: logger.info(f'Creating new batch_run: {os.path.basename(run_path)}')

# Create a directory to save the batch_run files
#logger.info(f'Run Path: {run_path}')
if not os.path.exists(run_path) and rank == 0:
    os.makedirs(run_path)
else:
    # Wait for the directory to be created by rank 0
    if not os.path.exists(run_path):
        logger.info(f'Rank {rank} waiting for run_path to be created: {run_path}')
        while not os.path.exists(run_path):
            time.sleep(1)
#sys.exit()

'''
Generate Config
'''
## If a candidate configuration is selected, load it
# example: selected_cand_cfg = 'batch_run_files/evol_params/evol_params_2021-07-07_16-00-00.json'
if selected_cand_cfg is not None:    
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
    "cores_per_node": cores_per_node, #256 cpus/node
    "nodes": num_nodes, #4 nodes
    "script": initFile,
    #"pop_per_core": 1,
    #"duration_seconds": 5,
    "pop_size": pop_size,   #128 popsize -> 8 cpus/task = (4 cores * 2 Threads)/task. 
                            #128 pop/4 nodes = 32 pop/node.
                            #8*cpus/task*32 pop/node = 256 cpus/node  
    "max_generations" : max_generations,
    "time_sleep": time_sleep, #seconds
    "maxiter_wait": maxiter_wait, #iterations of time_sleep
    "skip": skip,
    "num_elites": num_elites,
    "cfgFile": cfgFile,
    "netParamsFile": netParamsFile,
    "initCfg": initCfg,
}