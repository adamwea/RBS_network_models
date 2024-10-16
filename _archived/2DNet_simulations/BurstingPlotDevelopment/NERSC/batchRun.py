# This file is a modified version of the file batchRun.py from the NetPyNE tutorial 9.

# General Imports
import os
import shutil
import json
import pickle
import glob
import sys
import logging

# Set up logging
logging.basicConfig(filename='logfile.log', level=logging.INFO)

# Define a logger
logger = logging.getLogger(__name__)


# NetPyne Imports
#sys.path.insert(0, '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/netpyne')
from netpyne import specs
from netpyne.batch import Batch


# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
logging.info(f'Script directory: {script_dir}')

# Change the working directory to the script directory ! might create problems in nersc
os.chdir(script_dir)

logging.info(f'Changed working directory to: {script_dir}')
 #get current working directory

output_path = f'{script_dir}/output/' 
# Check if the output directory exists, if not, create it
if not os.path.exists(output_path):
    os.makedirs(output_path)

logging.info(f'Output path: {output_path}')

''' Example of evolutionary algorithm optimization of a network using NetPyNE
To run use: mpiexec -np [num_cores] nrniv -mpi batchRun.py
'''

# if folder "aw_grid" exists, delete it
# if os.path.exists('aw_grid'):
#     shutil.rmtree('aw_grid')

def batchRun(batchLabel = 'batchRun', method = 'grid', params=None, skip = False):

    if params is None:
        params = specs.ODict()

    batch_obj = Batch(params=params, cfgFile='nersccfg.py', netParamsFile='netParams.py')
    batch_obj.batchLabel = batchLabel
    batch_obj.method = method
    batch_obj.saveFolder = f"{output_path}/{batchLabel}"
    batch_obj.runCfg = {
                'type': 'mpi_bulletin',#'hpc_slurm',
        'script': 'init.py',
        'skip': skip,
        'cores': 16,


    }
    
    batch_obj.run()

# Main code
if __name__ == '__main__':
	#batchEvol('simple')  # 'simple' or 'complex'
    
    batchLabel = sys.argv[1]

    if batchLabel == "evol":
        #need to load the params.
        
        with open('nersc_evol_param.json', 'r') as f:
            params = json.load(f)

        batchRun(
            batchLabel = batchLabel, 
            method = 'evol', 
            params=params, 
            skip = True
            ) 
    else:


        with open('nersc_gridvar_param.json', 'r') as f:
            grid_params = json.load(f)
        params = specs.ODict()
        params[batchLabel] = grid_params[batchLabel]
        #params['duration_seconds'] = grid_params['duration_seconds']
        batchRun(
            batchLabel = batchLabel+'_grid', 
            #method = 'grid', 
            method = 'grid',
            params=params, 
            skip = True
            ) 
