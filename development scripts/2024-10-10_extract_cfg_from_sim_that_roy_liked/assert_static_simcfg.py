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
import numpy as np

def normal(mean, stddev):
    """Generate a random number from a normal (Gaussian) distribution."""
    return random.gauss(mean, stddev)

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

# Helper function to generate positive values based on a normal distribution
def positive_normal(mean, std):
    return abs(np.random.normal(mean, std))

def revise_cell_params(data_path, cfg_path, temp_netParams_path, extracted_data_path, save_json=False):
    
    # Assert that the file exists
    assert os.path.exists(data_path), f'File not found: {data_path}'

    # Load the simulation data
    print(f"Loading file {data_path} ... ")
    sim.load(data_path, simConfig=cfg_path, output=True)
    sim.loadSimCfg(cfg_path)
    sim.loadNetParams(data_path)
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
        
    sim.net.params.cellParams = new_cell_params
    sim.cfg.filename = extracted_data_path
    sim.cfg.saveFolder = os.path.dirname(extracted_data_path)
    sim.cfg.simLabel = sim.cfg.simLabel + '_extracted'
    #sim.create(simConfig=sim.cfg, netParams=sim.net.params, output=True)
    
    # Validate that no 'abs(' string functions exist in sim.net
    valid_net = copy(sim.net.cells)
    valid_net_dict = {f'cell_{i}': valid_net[i] for i in range(len(valid_net))}
    validate_lack_of_abs_string_funcs(valid_net_dict)
    valid_net = copy(sim.cfg).__dict__
    validate_lack_of_abs_string_funcs(valid_net)

    print("Cell Params Made Static.")

    return sim

if __name__ == "__main__":
    # Define paths
    data_path = os.path.abspath('discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/simulation_with_only_cand_of_interest/gen_1/gen_1_cand_29_data.json')
    temp_netParams_path = 'discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/temp_netParams.json'
    extracted_data_path = 'discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/extracted_data.json'
    cfg_path = os.path.abspath('discrete_development_tasks/extract_cfg_from_sim_that_roy_liked/simulation_with_only_cand_of_interest/gen_1/gen_1_cand_29_cfg.json')
    revise_cell_params(data_path, cfg_path, temp_netParams_path, extracted_data_path, save_json=True)