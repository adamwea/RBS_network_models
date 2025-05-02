import matplotlib; matplotlib.use('Agg')
import os
#output = os.environ.pop('GPU_SUPPORT_ENABLED', None)
import sys
from netpyne import sim
import json
#===================================================================================================
# Get and test MPI rank
# from mpi4py import MPI
# mpi_rank = MPI.COMM_WORLD.Get_rank()
# mpi_size = MPI.COMM_WORLD.Get_size()
# rank = mpi_rank
# print("Initiating Rank:", rank)
#get current script path and set as working directory
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
print('CWD:', os.getcwd())
#===================================================================================================

# simCfg = './cfg_vlatest.py'
# netParams = './netParams_vlatest.py'
simCfg = './cfg.py'
netParams = './netParams.py'

#NOTE:  these paths are used in the event that the arguments provided to the function in command line are not valid, and the default paths are used instead
        # generally, these paths are not used - but provided they match the paths in the batch config file, they will be copied to the batch folder and run by each simulation
        # in each rank
        
# initialize cfg runtime kwargs
cfg_runtime_kwargs_path = './cfg_kwargs.json'
#target_script_path = '../fitting/experimental_data_features/fitness_args_20241205_022033.py'
#param_script_path = '../fitting/evol_parameter_spaces/adjusted_evol_params_241202.py'
#param_script_path = '../_3_analysis/evol_parameter_spaces/adjusted_evol_params_241202.py'
#target_script_path = '../_3_analysis/experimental_data_features/fitness_args_20241205_022033.py'
evol_params_path = 'evol_params.py'

duration = 1
def init_cfg_runtime_kwargs(cfg_runtime_kwargs_path, duration_seconds, target_script_path, param_script_path):
    cfg_runtime_kwargs_path = os.path.abspath(cfg_runtime_kwargs_path)
    target_script_path = os.path.abspath(target_script_path)
    param_script_path = os.path.abspath(param_script_path)
    if os.path.exists(cfg_runtime_kwargs_path):
        os.remove(cfg_runtime_kwargs_path)
    cfg_runtime_kwargs = {
        'duration_seconds': duration_seconds,
        'target_script_path': target_script_path,
        'param_script_path': param_script_path,
    }
    with open(cfg_runtime_kwargs_path, 'w') as f:
        json.dump(cfg_runtime_kwargs, f, indent=4)
    print('cfg_runtime_kwargs initialized successfully.')
    return cfg_runtime_kwargs
#init_cfg_runtime_kwargs(cfg_runtime_kwargs_path, duration, target_script_path, param_script_path)

# initialize simConfig and netParams objects
simConfig, netParams = sim.readCmdLineArgs(
    simConfigDefault=simCfg,
    netParamsDefault=netParams,
)

print("simConfig and netParams objects initialized successfully.")

# Create network and run simulation
#sim.createSimulateAnalyze(netParams=netParams, simConfig=simConfig)