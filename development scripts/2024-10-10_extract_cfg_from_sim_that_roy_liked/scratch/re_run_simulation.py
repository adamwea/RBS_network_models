import sys
import os
from copy import copy, deepcopy
import random
# Add paths to the system for importing
sys.path.insert(0, 'simulate_local')
sys.path.insert(0, 'simulate_config_files')
sys.path.insert(0, 'submodules/netpyne')
from netpyne import sim, specs
import netpyne
from netpyne import sim

#extracted_data_path = '/app/discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/gen_1_cand_29_extracted_data.json'

# Load the simulation data
from static_param_extraction_functions import extract_static_params
data_path = os.path.abspath('discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/simulation_with_only_cand_of_interest/gen_1/gen_1_cand_29_data.json')
temp_netParams_path = 'discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/temp_netParams.json'
extracted_data_path = 'discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/extracted_data.json'
cfg_path = os.path.abspath('discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/simulation_with_only_cand_of_interest/gen_1/gen_1_cand_29_cfg.json')
_ , _, sim_obj = extract_static_params(data_path, cfg_path, temp_netParams_path, extracted_data_path, save_json=True)
sim=sim_obj # This is a hack to make the code below work

#Run the simulation
#simConfig = sim.cfg
#netParams = sim.net.params
#sim.createSimulate(netParams=netParams, simConfig=simConfig)


#(pops, cells, conns, rxd, stims, simData) = sim.create(netParams, simConfig, output=True)
sim_obj.simulate()