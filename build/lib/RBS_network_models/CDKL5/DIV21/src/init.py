import matplotlib; matplotlib.use('Agg')
import os
from netpyne import sim
import json
#===================================================================================================
# Get and test MPI rank
from mpi4py import MPI
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
rank = mpi_rank
print("Initiating Rank:", rank) 
script_dir = os.path.dirname(os.path.realpath(__file__)) #get current script path and set as working directory
os.chdir(script_dir)
print('CWD:', os.getcwd())
#===================================================================================================

#set paths to simulation configuration files
simCfg = 'cfg.py'
netParams = 'netParams.py'

#NOTE:  these paths are used in the event that the arguments provided to the function in command line are not valid, and the default paths are used instead
        # generally, these paths are not used - but provided they match the paths in the batch config file, they will be copied to the batch folder and run by each simulation
        # in each rank

# initialize simConfig and netParams objects
simConfig, netParams = sim.readCmdLineArgs(
    simConfigDefault=simCfg,
    netParamsDefault=netParams,
)

#print("simConfig and netParams objects initialized successfully.")
sim.createSimulateAnalyze(simConfig = simConfig, netParams = netParams)