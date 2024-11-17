import matplotlib; matplotlib.use('Agg')
from netpyne import sim
from mpi4py import MPI
#get rank of current process and print
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
rank = mpi_rank
print("Initiating Rank:", rank)
import os
import sys

# read cfg and netParams from command line arguments if available; otherwise use default
#simConfig, netParams = sim.readCmdLineArgs(simConfigDefault='simConfig.py', netParamsDefault='netParams.py')
if 'NERSC' not in os.getcwd():
    os.chdir('NERSC')
print('CWD:', os.getcwd())
simConfig, netParams = sim.readCmdLineArgs(simConfigDefault='cfg.py', netParamsDefault='netParams.py')
#sys.exit()


# Create network and run simulation
sim.createSimulateAnalyze(netParams=netParams, simConfig=simConfig)