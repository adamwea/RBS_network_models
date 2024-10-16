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

def update_sim_obj_filename(obj, new_filename):
    """
    Recursively traverse the object (dict or any other object with __dict__) and update 
    any 'filename' keys with the new filename.
    """
    #assert obj is or has a dict or object
    assert isinstance(obj, (dict, object)), f"Object is not a dict or object: {obj}"
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == 'filename':
                obj[key] = new_filename
            elif isinstance(value, (dict)):
                update_sim_obj_filename(value, new_filename)
    elif hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            if key == 'filename':
                obj.__dict__[key] = new_filename
            elif isinstance(value, (dict)):
                update_sim_obj_filename(value, new_filename)

# def validate_lack_of_abs_string_funcs(data):
#     for key, value in data.items():
#         if isinstance(value, dict):
#             for k, v in value.items():
#                 if isinstance(v, dict):
#                     for k2, v2 in v.items():
#                         if isinstance(v2, str):
#                             assert 'abs(' not in v2, f"String function with 'abs(' found in {key}.{k}.{k2}: {v2}"
#                 elif isinstance(v, str):
#                     assert 'abs(' not in v, f"String function with 'abs(' found in {key}.{k}: {v}"
#         elif isinstance(value, str):
#             assert 'abs(' not in value, f"String function with 'abs(' found in {key}: {value}"

def validate_lack_of_abs_string_funcs(data, path=""):
    """
    Recursively checks for the presence of 'abs(' in string values within
    the nested data structure. Raises an AssertionError if found.
    
    Parameters:
    - data: The data structure to validate (can be dict, list, or any other type)
    - path: A string that tracks the current position within the nested structure (for better error messages)
    """
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else str(key)
            validate_lack_of_abs_string_funcs(value, current_path)
    
    elif isinstance(data, list):
        for index, item in enumerate(data):
            current_path = f"{path}[{index}]"
            validate_lack_of_abs_string_funcs(item, current_path)
    
    elif isinstance(data, str):
        # Check if the string contains 'abs('
        assert 'abs(' not in data, f"String function with 'abs(' found at {path}: {data}"
    
    # For other types (int, float, etc.), no need to check further
    # These will be ignored automatically

import numpy as np
# Helper function to generate positive values based on a normal distribution
def positive_normal(mean, std):
    return abs(np.random.normal(mean, std))

def extract_static_params(data_path, cfg_path, temp_netParams_path, extracted_data_path, save_json=False):
    
    # Assert that the file exists
    assert os.path.exists(data_path), f'File not found: {data_path}'

    # Load the simulation data
    print(f"Loading file {data_path} ... ")
    sim.load(data_path, simConfig=cfg_path, output=True)
    sim.loadSimCfg(cfg_path)
    sim.loadNetParams(data_path)
    #old_cell_params = copy(sim.net.params.cellParams)
    #new_cell_params = copy(sim.net.params.cellParams)
    cells = copy(sim.net.cells)
    new_cell_params = copy(sim.net.params.cellParams)
    new_cell_params.pop('Erule')
    new_cell_params.pop('Irule')
    for i, cell in enumerate(cells):
        cfg = sim.cfg
        cellType = cells[i].tags['cellType']
        #new_cellRule = {'conds': {'cellType': cellType}, 'secs': cells[i].secs}
        new_cellRule = {'conds': {'cellType': cellType}, 'secs': {}}
        new_cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}
        new_cellRule['secs']['soma']['geom'] = {    
                'L': positive_normal(cfg.E_L_mean if cellType == 'E' else cfg.I_L_mean, 
                                    cfg.E_L_stdev if cellType == 'E' else cfg.I_L_stdev),  # Length
                'diam': positive_normal(cfg.E_diam_mean if cellType == 'E' else cfg.I_diam_mean, 
                                        cfg.E_diam_stdev if cellType == 'E' else cfg.I_diam_stdev),  # Diameter
                'Ra': positive_normal(cfg.E_Ra_mean if cellType == 'E' else cfg.I_Ra_mean, 
                                    cfg.E_Ra_stdev if cellType == 'E' else cfg.I_Ra_stdev),  # Axial resistance
            }
        new_cellRule['secs']['soma']['mechs']['hh'] = {
            'gnabar': positive_normal(cfg.gnabar_E if cellType == 'E' else cfg.gnabar_I, cfg.gnabar_E_std if cellType == 'E' else cfg.gnabar_I_std),  # Sodium conductance
            'gkbar': positive_normal(cfg.gkbar_I if cellType == 'I' else cfg.gkbar_E, cfg.gkbar_I_std if cellType == 'I' else cfg.gkbar_E_std),  # Potassium conductance
            'gl': 0.003,  # Leak conductance
            'el': -70,  # Leak reversal potential
            }
        new_cell_params[f'{cellType}{i}rule'] = new_cellRule
        
        #cellType = cells[i].tags['cellType']
        #new_cell_params[f'{cellType}{i}rule'] ={
        #    'conds': {'cellType': cellType},
        #    'secs': cells[i].secs,
        #    #'soma': cells[i].soma,
        #}
        #---
        # cellRule = {'conds': {'cellType': cellType}, 'secs': {}}
        # cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}
        # cellRule['secs']['soma']['geom'] = {    
        #         'L': positive_normal(cfg.E_L_mean if cellType == 'E' else cfg.I_L_mean, 
        #                             cfg.E_L_stdev if cellType == 'E' else cfg.I_L_stdev),  # Length
        #         'diam': positive_normal(cfg.E_diam_mean if cellType == 'E' else cfg.I_diam_mean, 
        #                                 cfg.E_diam_stdev if cellType == 'E' else cfg.I_diam_stdev),  # Diameter
        #         'Ra': positive_normal(cfg.E_Ra_mean if cellType == 'E' else cfg.I_Ra_mean, 
        #                             cfg.E_Ra_stdev if cellType == 'E' else cfg.I_Ra_stdev),  # Axial resistance
        #     }
        # cellRule['secs']['soma']['mechs']['hh'] = {
        #     'gnabar': positive_normal(cfg.gnabar_mean_E if cellType == 'E' else cfg.gnabar_mean_I, cfg.gnabar_stdev),  # Sodium conductance
        #     'gkbar': positive_normal(cfg.gkbar_mean_I if cellType == 'I' else cfg.gkbar_mean_E, cfg.gkbar_stdev),  # Potassium conductance
        #     'gl': 0.003,  # Leak conductance
        #     'el': -70,  # Leak reversal potential
        #     }

    #old_cells = copy(sim.net.cells)
    #new_cellParams = copy(sim.net.cells)
    #new_cell_params = eval_string_funcs_in_netParams(new_cell_params)
    #new_cellParams = {f'compartCell{i}': new_cellParams[i] for i in range(len(new_cellParams))}
    sim.net.params.cellParams = new_cell_params
    sim.cfg.filename = extracted_data_path
    sim.cfg.saveFolder = os.path.dirname(extracted_data_path)
    sim.cfg.simLabel = sim.cfg.simLabel + '_extracted'
    sim.create(simConfig=sim.cfg, netParams=sim.net.params, output=True)
    
    # # Evaluate string functions in netParams
    # network_new = copy(sim.net)
    # network_new.cells = eval_string_funcs_in_netParams(network_new.cells)

    # # Convert the list of cells to a dictionary for saving
    # new_cells = {f'cell_{i}': network_new.cells[i] for i in range(len(network_new.cells))}
    # netParams_new = {'cellParams': new_cells}



    # Validate that no 'abs(' string functions exist in sim.net
    valid_net = copy(sim.net.cells)
    valid_net_dict = {f'cell_{i}': valid_net[i] for i in range(len(valid_net))}
    validate_lack_of_abs_string_funcs(valid_net_dict)
    valid_net = copy(sim.cfg).__dict__
    validate_lack_of_abs_string_funcs(valid_net)

    sim.simulate()

    # Set the filename for saving the extracted data and save it
    sim.cfg.filename = extracted_data_path
    sim.cfg.saveFolder = os.path.dirname(extracted_data_path)
    sim.cfg.simLabel = sim.cfg.simLabel + '_extracted'
    simConfig = sim.cfg
    netParams = sim.net.params
    print("Extraction complete.")
    return simConfig, netParams, sim

if __name__ == "__main__":
    # Define paths
    data_path = os.path.abspath('discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/simulation_with_only_cand_of_interest/gen_1/gen_1_cand_29_data.json')
    temp_netParams_path = 'discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/temp_netParams.json'
    extracted_data_path = 'discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/extracted_data.json'
    cfg_path = os.path.abspath('discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/simulation_with_only_cand_of_interest/gen_1/gen_1_cand_29_cfg.json')
    extract_static_params(data_path, cfg_path, temp_netParams_path, extracted_data_path, save_json=True)