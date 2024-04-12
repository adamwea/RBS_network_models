import os
import pickle
import json
import shutil
import sys
import json

'''
USER INPUT
'''
## Edit the following parameters as needed ##
## Edit params in the batch_run_files folder as needed (with care) ##

## Edit Run_Name to a unique name for the batch run ##
if sys.argv[1] is not None: run_name = sys.argv[1] #batch_run_files folder name
else: run_name = 'NERSC_Test' ### Change this to a unique name for the batch run

## Batch Method ##
method = 'evol' #evolutionary algorithm
# method = 'grid' #grid search

## Selecte a candidate configuration to start the evolution from ##
selected_cand_cfg = None
# selected_cand_cfg = '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5.1_optimizing_EEonly/output/24-3-24_5sec_EEsearch/gen_5/gen_5_cand_29_cfg.json'

## Parallelization Parameters ##
pop_per_core = 12
core_num = 4
num_nodes = 1
pop_size = pop_per_core * core_num
num_elite_percent = 10/100 # top 10% of the population will be copied to the next generation, this is considered high-medium elitism
num_elites = int(num_elite_percent * pop_size)
#duration_seconds = 5

## Overwrite Parameters ##
overwrite_run = False #True will overwrite any existing run with the same name
continue_ = True #True will continue from the last generation of the run
#make sure batch_run_path does not exist. Avoid overwriting data from any Run
#overwrite_run = True #comment out later, this is for debugging

'''
Initialize
'''
##Prepare Batch_Run_Folder and Initial Files
#output_path = nb_path.replace('batch_run_files', 'output')'
script_path = os.path.dirname(os.path.realpath(__file__))
output_path = script_path+'/output/'
run_path = output_path + run_name

## Mediate Overwrite
if continue_ is False:
    assert overwrite_run or not os.path.exists(run_path), f"Run {run_path} already exists. Set overwrite_run to True to overwrite."
if overwrite_run and os.path.exists(run_path):    
    shutil.rmtree(run_path)

# Create a directory to save the batch files
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

##Create a dictionary with the given variables and their values
batch_config_options = {
    "run_path": run_path,
    'batchLabel': os.path.basename(run_path),
    "method": method,
    "core_num": core_num,
    "nodes": num_nodes,
    #"pop_per_core": 1,
    #"duration_seconds": 5,
    "pop_size": pop_size,
    "max_generations": 2000,
    "time_sleep": 10, #seconds
    "maxiter_wait": 6*40, #6itersx10sec/itersx30sec/min = 30 minutes per gen?
    "skip": True,
    "num_elites": num_elites,
    "initCfg": initCfg,
}

# Write the dictionary to a JSON file
with open("batch_config_options.json", "w") as f:
    json.dump(batch_config_options, f)