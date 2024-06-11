''' NOTE: it's important thiat this script only prints run_path and nothing else.'''

import os
import datetime
import shutil
from USER_INPUTS import USER_run_label, USER_overwrite, USER_continue

def init_batch_run(USER_run_label=None, run_path_only = False):
    '''
    Initialize
    '''
    if USER_run_label is None: from USER_INPUTS import USER_run_label
    
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

    # Get unique run path
    run_path = f'{output_path}/{run_name}'
    prev_run_path = f'{output_path}/{prev_run_name}'

    '''Check if run exists and if it should be overwritten or continued'''
    #from USER_INPUTS import USER_overwrite, USER_continue
    assert not (USER_overwrite and USER_continue), 'overwrite_run and continue_run cannot both be True'
    if USER_overwrite or USER_continue:
        if prev_run_name in existing_runs:
            assert not (USER_overwrite and USER_continue), 'overwrite_run and continue_run cannot both be True'
            if USER_overwrite and os.path.exists(prev_run_path):
                if not run_path_only: 
                    shutil.rmtree(prev_run_path)
                run_path = prev_run_path   
            elif USER_continue and os.path.exists(prev_run_path):
                run_path = prev_run_path
    if not os.path.exists(run_path):# and rank == 0:
        os.makedirs(run_path)
    return run_path, run_name, USER_run_label

if __name__ == '__main__':
    #from USER_INPUTS import USER_run_label
    run_path, run_name, USER_run_label = init_batch_run(USER_run_label)
    
    #save run_path to .txt file
    script_path = os.path.dirname(os.path.realpath(__file__))
    temp_path = f'{script_path}/temp'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    with open(f'{temp_path}/run_path.txt', 'w') as file:
        file.write(run_path)

