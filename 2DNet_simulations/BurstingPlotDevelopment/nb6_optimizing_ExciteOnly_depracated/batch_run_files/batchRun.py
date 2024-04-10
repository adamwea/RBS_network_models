# This file is a *highly* modified version of the file batchRun.py from the NetPyNE tutorial 9.

# General Imports
import os
import shutil
import json
import pickle
import glob
import sys

# NetPyne Imports
sys.path.insert(0, '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/netpyne')
from netpyne import specs
from netpyne.batch import Batch

# # Get the directory of the current script
# script_dir = os.path.dirname(os.path.realpath(__file__))
# print(f'Script directory: {script_dir}')

# # Change the working directory to the script directory
# os.chdir(script_dir)

# print(f'Changed working directory to: {script_dir}')
#  #get current working directory
# output_path = os.path.dirname(script_dir)
# output_path = f'{output_path}/output' 
# print(f'Output path: {output_path}')

''' Example of evolutionary algorithm optimization of a network using NetPyNE
To run use: mpiexec -np [num_cores] nrniv -mpi batchRun.py
'''

def batchRun(batchLabel = 'batchRun', method = 'evol', skip = True, batch_config = None):
    
    #Get batch_config as needed         
    if batch_config is None:
        # Load the dictionary from the JSON file
        with open('batch_config.pickle', 'rb') as f:
            batch_config = pickle.load(f)
        assert 'method' in batch_config, 'method must be specified in batch_config'
        assert 'skip' in batch_config['runCfg'], 'skip must be specified in batch_config'
        assert 'batchLabel' in batch_config, 'batchLabel must be specified in batch_config'
        #after batch_config is loaded, delete it, there's a copy in the batch_run_path
        #os.remove('batch_config.pickle')
    
    #extract_batch params
    batch_run_path =  batch_config['saveFolder']

    #load param_space from batch_run_path json
    assert os.path.exists(f'{batch_run_path}/param_space.pickle'), 'params.json does not exist in run_path'
    with open(f'{batch_run_path}/param_space.pickle', 'rb') as f:
        params = pickle.load(f)   

	# create Batch object with paramaters to modify, and specifying files to use
    batch = Batch(params=params)
    batch.method = batch_config['method']
    batch.batchLabel = batch_config['batchLabel']
    batch.saveFolder = batch_config['saveFolder']

    #prepare run and evol configuration
    batch.runCfg = batch_config['runCfg']
    batch.evolCfg = batch_config['evolCfg']    

    #run batch
    batch.run()

# Main code
if __name__ == '__main__':
    cwd = os.getcwd()
    print(f'Current working directory: {cwd}')
    
    # Load the dictionary from the JSON file
    with open('batch_config.pickle', 'rb') as f:
        batch_config = pickle.load(f)
    assert 'method' in batch_config, 'method must be specified in batch_config'
    assert 'skip' in batch_config['runCfg'], 'skip must be specified in batch_config'
    assert 'batchLabel' in batch_config, 'batchLabel must be specified in batch_config'
    #after batch_config is loaded, delete it, there's a copy in the batch_run_path
    # os.remove('batch_config.pickle')
    
    batchRun(
        batchLabel = batch_config['batchLabel'], 
        #method = 'grid', 
        method = batch_config['method'],
        skip = batch_config['runCfg']['skip'],
        batch_config = batch_config
        ) 
