'''Setup Python environment for running the script'''
# from pprint import pprint
# try:
#     import setup_environment as setup_environment
#     setup_environment.set_pythonpath()
# except:
#     #import workspace.RBS_network_simulations.workspace.optimization_projects.CDKL5_DIV21._2_batchrun_optimization.setup_environment as setup_environment
#     import workspace.optimization_projects.CDKL5_DIV21._2_batchrun_optimization.setup_environment as setup_environment
#     setup_environment.set_pythonpath()
# # import workspace.RBS_network_simulations.workspace.optimization_projects.CDKL5_DIV21._2_batchrun_optimization.setup_environment as setup_environment
# # setup_environment.set_pythonpath()

import matplotlib; matplotlib.use('Agg')
import sys
#sys.path.insert(0, 'submodules/netpyne')
#from netpyne import netpyne
from netpyne import sim
from mpi4py import MPI
#get rank of current process and print
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
rank = mpi_rank
print("Initiating Rank:", rank)
import os
import sys
#sys.path.insert(0, 'simulate_config_files')
#from simulate_config_files import *

# read cfg and netParams from command line arguments if available; otherwise use default
#simConfig, netParams = sim.readCmdLineArgs(simConfigDefault='simConfig.py', netParamsDefault='netParams.py')
# if 'NERSC' not in os.getcwd():
#     os.chdir('NERSC')
print('CWD:', os.getcwd())
simConfig, netParams = sim.readCmdLineArgs(simConfigDefault='cfg.py', netParamsDefault='netParams.py')


# Create network and run simulation
sim.createSimulateAnalyze(netParams=netParams, simConfig=simConfig)