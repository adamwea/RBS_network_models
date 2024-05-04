import matplotlib; matplotlib.use('Agg')
from netpyne import sim

# read cfg and netParams from command line arguments if available; otherwise use default
#simConfig, netParams = sim.readCmdLineArgs(simConfigDefault='simConfig.py', netParamsDefault='netParams.py')
simConfig, netParams = sim.readCmdLineArgs(simConfigDefault='/mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/InhibNeuronDebug/output/240430_Run21_debug_run/gen_0/gen_0_cand_1_cfg.json', 
                                           netParamsDefault='/mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/InhibNeuronDebug/output/240430_Run21_debug_run/240430_Run21_debug_run_netParams.py')

# Create network and run simulation
sim.createSimulateAnalyze(netParams=netParams, simConfig=simConfig)