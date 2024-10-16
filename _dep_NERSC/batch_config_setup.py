import os
import pickle
import json
import shutil
import sys
import json
import datetime
import time

## Logger
import logging
script_dir = os.path.dirname(os.path.realpath(__file__))
log_file = f'{script_dir}/batchRun.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

## User Inputs
from USER_INPUTS import *

'''
Initialize
'''
# Get the rank of the current process
rank = os.environ.get('OMPI_COMM_WORLD_RANK')
if rank is None: rank = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK')
if rank is None: rank = os.environ.get('OMPI_COMM_WORLD_NODE_RANK')
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
run_name = f'{current_date}_Run{new_run_number}_{USER_run_label}'
prev_run_name = f'{current_date}_Run{prev_run_number}_{USER_run_label}'
#print(f'Run Name: {run_name}')
#sys.exit()   

# Get unique run path
run_path = f'{output_path}/{run_name}'
prev_run_path = f'{output_path}/{prev_run_name}'

if USER_overwrite or USER_continue:
    if prev_run_name in existing_runs:
        assert not (USER_overwrite and USER_continue), 'overwrite_run and continue_run cannot both be True'
        if USER_overwrite and os.path.exists(prev_run_path):
            shutil.rmtree(prev_run_path)
            run_path = prev_run_path   
            logger.info(f'Overwriting existing batch_run: {os.path.basename(run_path)}')
        elif USER_continue and os.path.exists(prev_run_path):
            run_path = prev_run_path
            logger.info(f'Continuing existing batch_run: {os.path.basename(run_path)}')

if not os.path.exists(run_path) and rank == 0:
    logger.info(f'Creating new batch_run: {os.path.basename(run_path)}')
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
#load HOF of previous runs
selected_cand_cfg = None
if selected_cand_cfg is not None:    
    with open(selected_cand_cfg, 'r') as f:
        initCfg = json.load(f)
        initCfg = initCfg['simConfig']
else: initCfg = None

##Create a dictionary with the given variables and their values
batch_config_options = {
    "run_path": run_path,
    'batchLabel': os.path.basename(run_path),
    #"initCfg": initCfg,
}