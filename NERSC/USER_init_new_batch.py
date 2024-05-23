''' NOTE: it's important thiat this script only prints run_path and nothing else.'''

import os
import sys
import datetime
import shutil
import time
import json

#from USER_INPUTS import USER_run_label
# from USER_INPUTS import USER_overwrite
# from USER_INPUTS import USER_continue
USER_continue = True #Continue the previous run without deleting it
USER_overwrite = False #Delete the previous run and start a new one
try: USER_run_label = sys.argv[-1] ### Change this to a unique name for the batch run
except: USER_run_label = 'USER_int_debug' ### Change this to a unique name for the batch run

def init_new_batch(USER_run_label, run_path_only = False):
    '''
    Initialize
    '''
    # from mpi4py import MPI
    # mpi_rank = MPI.COMM_WORLD.Get_rank()
    # mpi_size = MPI.COMM_WORLD.Get_size()
    # rank = mpi_rank
    # #rank = os.environ.get('OMPI_COMM_WORLD_RANK')
    # if not rank: rank = 0
    # rank = int(rank)

    # Get current date in YYMMDD format
    current_date = datetime.datetime.now().strftime('%y%m%d')
    # Prepare Batch_Run_Folder and Initial Files
    script_path = os.path.dirname(os.path.realpath(__file__))
    output_path = script_path+'/output'
    #print(f'Output Path: {output_path}')

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
                if not run_path_only: 
                    #print(rank)
                    shutil.rmtree(prev_run_path)
                run_path = prev_run_path   
                #logger.info(f'Overwriting existing batch_run: {os.path.basename(run_path)}')
                #print(f'Overwriting existing batch_run: {os.path.basename(run_path)}')
            elif USER_continue and os.path.exists(prev_run_path):
                run_path = prev_run_path
                #logger.info(f'Continuing existing batch_run: {os.path.basename(run_path)}')
                #print(f'Continuing existing batch_run: {os.path.basename(run_path)}')

    if not os.path.exists(run_path):# and rank == 0:
        #logger.info(f'Creating new batch_run: {os.path.basename(run_path)}')
        #print(f'Creating new batch_run: {os.path.basename(run_path)}')
        os.makedirs(run_path)
    # else:
    # Wait for the directory to be created by rank 0
    # if not os.path.exists(run_path):
    #     #print(f'Rank {rank} waiting for run_path to be created: {run_path}')
    #     #logger.info(f'Rank {rank} waiting for run_path to be created: {run_path}')
    #     while not os.path.exists(run_path):
    #         time.sleep(1)
    #sys.exit()
    return run_path, run_name, USER_run_label

if __name__ == '__main__':

    
    #print('Initializing new batch run...')
    assert not (USER_overwrite and USER_continue), 'overwrite_run and continue_run cannot both be True'
    #print(f'USER_run_label: {USER_run_label}')
    #print(f'USER_overwrite: {USER_overwrite}')
    #print(f'USER_continue: {USER_continue}')
    run_path, run_name, USER_run_label = init_new_batch(USER_run_label)
    #print(f'Run Path: {run_path}')
    #print(f'Run Name: {run_name}')
    #save a .json file at run_path with run_name and run_path
    # with open(f'{run_path}/run_info.json', 'w') as f:
    #     json.dump({'run_name': run_name, 'run_path': run_path}, f)
    #sys.exit()  
    print(run_path)