import sys
import os
import math
import ast
from copy import copy
import random
import json

# Import necessary NetPyNE modules
from netpyne import sim
from netpyne import specs
import netpyne

def normal(mean, stddev):
    """Generate a random number from a normal (Gaussian) distribution."""
    return random.gauss(mean, stddev)

def eval_string_funcs_in_netParams(data):
    """Recursively traverse netParams, evaluating strings that begin with 'abs(' as mathematical expressions."""
    
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = eval_string_funcs_in_netParams(value)  # Recurse into nested dicts
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = eval_string_funcs_in_netParams(data[i])  # Recurse into lists
    elif isinstance(data, str):
        # Only evaluate expressions that begin with 'abs('
        if data.startswith('abs('):
            try:
                return eval(data, {"normal": normal, "abs": abs})
            except Exception as e:
                print(f"Failed to evaluate expression: {data}, Error: {e}")
                return data  # Return original string if eval fails
    elif isinstance(data, tuple): pass
    elif isinstance(data, set): pass
    elif isinstance(data, float): pass
    elif isinstance(data, int): pass
    else:
        data_dict = copy(data).__dict__
        for key, value in data_dict.items():
            data_dict[key] = eval_string_funcs_in_netParams(value)  # Recurse into nested dicts
        data = data_dict
    # Return the data unchanged if it's not a string or doesn't match the criteria
    return data

def recursive_update_skeleton(new_params, skeleton):
    """Recursively update skeleton params with values from new_params."""
    
    for key, value in new_params.items():
        # If the skeleton has the key, update it
        if hasattr(skeleton, key):
            current_attr = getattr(skeleton, key)

            # If both are dict-like structures, recurse further
            if isinstance(value, dict) and isinstance(current_attr, dict):
                recursive_update_skeleton(value, current_attr)
            
            # If one is a dict but the other is not, overwrite the skeleton's attribute
            elif isinstance(value, dict) and not isinstance(current_attr, dict):
                setattr(skeleton, key, copy.deepcopy(value))
            
            # For lists, try to merge or replace (simple case: overwrite the skeleton list)
            elif isinstance(value, list) and isinstance(current_attr, list):
                setattr(skeleton, key, value)
            
            # If value is not a dict, simply replace the skeleton's attribute
            else:
                setattr(skeleton, key, value)

        # If the skeleton does not have the key, go deeper
        # else:
        #     # If the value is a dict, create a new object and recurse
        #     if isinstance(value, dict):
        #         setattr(skeleton, key, recursive_update_skeleton(value, copy(getattr(skeleton, key))))
        #     # If the value is not a dict, simply replace the skeleton's attribute
        #     else:
        #         setattr(skeleton, key, value)

        # If skeleton does not have the key, create it (assuming it's valid for skeleton)
        else:
            setattr(skeleton, key, copy.deepcopy(value))

    return skeleton

# Add paths to the system for importing
sys.path.insert(0, 'simulate_local')
sys.path.insert(0, 'simulate_config_files')
sys.path.insert(0, 'submodules/netpyne')



# Define the path to the data file containing netParams
data_path = 'discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/simulation_with_only_cand_of_interest/gen_1/gen_1_cand_29_data.json'
data_path = os.path.abspath(data_path)

# Assert that the file exists
assert os.path.exists(data_path), f'File not found: {data_path}'

# Load the simulation data
print(f"Loading file {data_path} ... ")
sim.load(data_path, output=True)
sim.loadNetParams(data_path)
#sim.loadAll(data_path)
network = sim.net
simConfig = sim.cfg


# Inside netParams, some values are expressed as string-funcs...
# so we need to evaluate them using math in the same way NEURON would
# Because of this, we aren't sure we can perfectly recreate the original netParams...
# but we can save the netParams from this reloaded data - we will compare them to the original netParams

# Remove redundant keys from netParams
# Evaluate string functions in netParams
network_new = copy(network)
evaluated_cells_in_net = eval_string_funcs_in_netParams(network_new.cells)
network_new.cells = evaluated_cells_in_net
# netParams_new = {
#     'net': {
#         'params': {
#             'cellParams': network_new.cells,
#             #connParams': network_new.conns
#         }
#     }
# }
new_cells = network_new.cells
#cheeck type of new_cells
print(type(new_cells))
#convert list to dict
new_cells = {f'cell_{i}': new_cells[i] for i in range(len(new_cells))}
netParams_new = {'cellParams': new_cells}
print(type(netParams_new))

# Skeleton NetParams object (empty or with minimal structure)
# Recursively update the skeleton with matching keys from netParams_new
#netParams_new_and_structured = recursive_update_skeleton(netParams_new, skeleton = specs.NetParams())
netParams_new_and_structured = specs.NetParams()
netParams_new_and_structured.cellParams = netParams_new['cellParams']

# Save the netParam_skeleton to a file
new_path = 'discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/temp_netParams.json'
abs_path = os.path.abspath(new_path)
netParams_new_and_structured.save(abs_path)


# # Save netParams as regular json file first
# with open('discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/extracted_netParams.json', 'w') as f:
#     json.dump(netParams_new, f, indent=4, default=str)  # Use default=str to handle non-serializable objects

#Clear the sim object
netpyne.sim.clearAll() #clear all sim data
#sim.load(data_path, output=True) #reload the sim data
sim.loadSimCfg(data_path) #load the new simConfig
sim.loadNetParams(abs_path) #load the new netParams

#loop through sim.net and assert that there are no string functions with 'abs('
valid_net = copy(sim.net) #copy the sim.net object
valid_net = valid_net.__dict__ #convert to dict
for key, value in valid_net.items():
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, str):
                        assert 'abs(' not in v2, f"String function with 'abs(' found in {key}.{k}.{k2}: {v2}"
            elif isinstance(v, str):
                assert 'abs(' not in v, f"String function with 'abs(' found in {key}.{k}: {v}"
    elif isinstance(value, str):
        assert 'abs(' not in value, f"String function with 'abs(' found in {key}: {value}"
    else:
        continue

#save extracted params as new data.json
target_filepath = 'discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/extracted_data.json'
abs_target_filepath = os.path.abspath(target_filepath)
sim.cfg.filename = abs_target_filepath
sim.saveData() #save the data

#clean up
temp_path = abs_path
os.remove(abs_path) #remove the temp netParams file