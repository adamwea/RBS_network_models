'''
sensitivity_analysis.py
This script is designed to run sensitivity analysis on a network model simulations.
It generates a standard output directory for the sensitivity analysis results,
creates a symlink to the original simulation data file, and validates the original simulation data.
'''

from RBS_network_models.utils import netpyne_helpers as nph
from MEA_Analysis.NetworkAnalysis_aw import network_metrics_helper as nmh
from pprint import pprint
from netpyne import sim, specs
import numpy as np
import os
from time import sleep
import pickle
import json

def _generate_permutations(perm_levels, origin_cfg_path, origin_net_params, perm_output_dir, evol_params, overwrite=False):
    """
    Generate permutations of the configuration parameters based on the provided levels.
    
    input:
    - perm_levels: dict, where keys are parameter names and values are lists of levels to permute.
    - origin_cfg_path: str, path to the original configuration file.
    - perm_output_dir: str, directory to save the generated permutations.
    - overwrite: bool, if True, overwrite existing permutation files.
    """
    
    import os
    import json
    
    
    # Load the original configuration file
    origin_cfg = nph.load_sim_config_from_json(origin_cfg_path)
    origin_netParams = nph.load_net_params_from_json(origin_net_params)
    
    # Create output directory if it doesn't exist
    os.makedirs(perm_output_dir, exist_ok=True)

    # Create a dictionary to hold all permutations
    permutations = {}
    
    # Loop through each parameter and its levels
    for param, levels in perm_levels.items():
        if param not in origin_cfg:
            raise ValueError(f"Parameter '{param}' not found in the original configuration file.")
        
        # set up list of sim_labels based on the param currently being permuted
        num_levels = len(levels)
        list_of_levels = list(range(-num_levels//2, num_levels//2 + 1))  # e.g. [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        negative_labels = [f"{param}{li}" for li in list_of_levels if li < 0]
        positive_labels = [f"{param}+{li}" for li in list_of_levels if li > 0]
        list_of_labels = negative_labels + positive_labels
        assert len(list_of_labels) == num_levels, \
            f"Number of labels ({len(list_of_labels)}) does not match number of levels ({num_levels}) for parameter '{param}'."
        
        # Create a new configuration for each level
        for i, level in enumerate(levels):
            
            # Check if the permutation already exists
            simLabel = list_of_labels[i]
            expected_dir = os.path.join(perm_output_dir, simLabel)
            if os.path.exists(expected_dir) and not overwrite:
                expected_netParam_path = os.path.join(expected_dir, f"{simLabel}_netParams.json")
                expected_cfg_path = os.path.join(expected_dir, f"{simLabel}_cfg.json")
                if os.path.exists(expected_netParam_path) and os.path.exists(expected_cfg_path):
                    permutations[simLabel] = {
                        'netParams': expected_netParam_path,
                        'cfg': expected_cfg_path
                    }
                    print(f"Permutation for {param} at level {simLabel} already exists: {expected_dir}. Skipping...")
                    continue
            
            # Mutate the original configuration and network parameters
            netParams, simConfig = nph.mutate_sim_and_net_params(
                netParams=origin_netParams,
                simConfig=origin_cfg,
                param=param,
                set_value=level,
                evol_params=evol_params,
                #sim_label=list_of_labels[i],
                #saveFolder=perm_output_dir,
                #save=True,  # save the mutated configuration
                #verbose=True,
            )
            
            # Save the new configuration and network parameters
            net_params_path, sim_cfg_path = nph.save_sim_cfg_and_net_params(
                netParams,
                simConfig,
                saveFolder=perm_output_dir,
                simLabel=list_of_labels[i],
                overwrite=overwrite,  # overwrite existing files if necessary
            )

            # Store the paths in the permutations dictionary
            permutations[list_of_labels[i]] = {
                'netParams': net_params_path,
                'cfg': sim_cfg_path
            }

            # reload the origin_cfg and origin_netParams to deal with persistent changes
            origin_cfg = nph.load_sim_config_from_json(origin_cfg_path)
            origin_netParams = nph.load_net_params_from_json(origin_net_params)
            
            print(f"Generated permutation for {param} at level {level}: {list_of_labels[i]}")
            
            #print(f"Generating permutation for {param} at level {level}...")
            # new_cfg = origin_cfg.copy()
            # new_cfg[param] = level
            
            # # Define the output file name
            # output_file = os.path.join(perm_output_dir, f"{param}_{level}.json")
            
            # # Check if the file already exists and overwrite if necessary
            # if os.path.exists(output_file) and not overwrite:
            #     print(f"File {output_file} already exists. Skipping...")
            #     continue
            
            # # Save the new configuration to a JSON file
            # with open(output_file, 'w') as f:
            #     json.dump(new_cfg, f, indent=4)
            
            # print(f"Generated permutation for {param} at level {level}: {output_file}")
    print(f"Generated {len(levels)} permutations for {param}.")
    return permutations

def _define_permutation_levels(evol_params, origin_cfg_path, levels=None):
    """
    Extract each param and its range from the evolution parameters.
    1. Loop through each param in evol_params.
    2. Find matching param in origin_cfg_path.
    3. Get the currently set value of the param in the original cfg file. Set that as the center of the range.
    4. Use linspace to generate levels to the left and right of center value, for a total of `levels` values + 1 (center value).
    5. Return a dictionary with the param as key and the list of levels as value.
    """
    
    #import json
    import numpy as np
    
    if levels is None:
        levels = 10  # default number of sensitivity analysis levels
    
    # Load the original configuration file
    # with open(origin_cfg_path, 'r') as f:
    #     origin_cfg = json.load(f)
    origin_cfg = nph.load_sim_config_from_json(origin_cfg_path)
    
    perm_levels = {}
    
    for param, param_range in evol_params.items():
        if param not in origin_cfg:
            raise ValueError(f"Parameter '{param}' not found in the original configuration file.")
        
        assert isinstance(param_range, (list, tuple)) and len(param_range) == 2, \
            f"Parameter range for '{param}' must be a list or tuple of two values, got {param_range}."
        
        # Get the current value of the parameter in the original configuration
        current_value = origin_cfg[param]
        
        # Generate levels around the current value
        left_bound = param_range[0]
        right_bound = param_range[1]
        left_levels = np.linspace(left_bound, current_value, num=levels // 2, endpoint=False)
        right_levels = np.linspace(current_value, right_bound, num=levels // 2 + 1)[1:]  # Exclude the center value
        
        # Combine left and right levels
        perm_levels[param] = np.concatenate((left_levels, right_levels)).tolist()
    
    return perm_levels

def _symlink_to_origin_simulation(origin_sim_path, sa_run_dir):
    """
    Create a symlink to the original simulation data file.
    
    input:
    - origin_sim_path: str, path to the original simulation data file. Pkl or json file.
    """
    import os
    import shutil
    
    if not os.path.exists(origin_sim_path):
        raise FileNotFoundError(f"Original simulation data file not found: {origin_sim_path}")
    
    # Create a symlink in the current directory
    link_name = "_origin"
    origin_sim_dir = os.path.dirname(origin_sim_path)
    origin_symlink_path = os.path.join(sa_run_dir, link_name)
    
    # Check if the symlink already exists
    if os.path.islink(origin_symlink_path):
        # If it exists, check if it points to the correct target
        if os.readlink(origin_symlink_path) == origin_sim_dir:
            print(f"Symlink already exists and points to the correct target: {origin_symlink_path}")
            return origin_symlink_path
        else:
            # If it points to a different target, remove the old symlink
            print(f"Removing existing symlink: {origin_symlink_path}")
            os.remove(origin_symlink_path)
    
    # create the symlink if it does not exist
    os.symlink(origin_sim_dir, origin_symlink_path)
    print(f"Created symlink to original simulation data: {link_name}")
    
    return os.path.join(sa_run_dir, link_name)

def _generate_standard_sa_run_dirs(origin_sim_path, proj_output_dir, overwrite=False):
    """
    Generate a standard output directory for sensitivity analysis results.
    
    sa_output_dir/year-month/day-hh-mm/
    
    input:
    - origin_sim_path: str, path to the original simulation data file. Pkl or json file.
    - sa_output_dir: str, optional, base directory for the sensitivity analysis output. If None, it will be derived from the origin_sim_path.
    """
    import os
    from datetime import datetime
    
    # if sa_output_dir is None:
    #     # Derive the output directory from the original simulation path
    #     sa_output_dir = os.path.dirname(origin_sim_path)
    
    # Create a timestamp for the output directory
    #timestamp = datetime.now().strftime("%Y%m/%d")
    timestamp = datetime.now().strftime("%y%m%d")
    
    # Construct the full output directory path
    full_output_dir = os.path.join(proj_output_dir, timestamp, "sensitivity_analysis")
    # Always create a new run directory (run0000, run0001, ...)
    run_base_dir = full_output_dir
    os.makedirs(run_base_dir, exist_ok=True)
    subdirs = [d for d in os.listdir(run_base_dir) if os.path.isdir(os.path.join(run_base_dir, d)) and d.startswith("run")]
    if subdirs:
        run_numbers = [int(d[3:]) for d in subdirs if d[3:].isdigit()]
        next_run_number = max(run_numbers) + 1
    else:
        next_run_number = 0
        
    if overwrite and next_run_number > 0:
        # If overwrite is True, just subtract 1 from the next run number
        print(f"Overwriting the latest sensitivity analysis run: run{next_run_number:04d}")
        next_run_number -= 1
        # NOTE: This will overwrite the latest run directory, so be careful with this option.
        # - With individual simulations, I will include options to either overwrite or keep simulation data in the latest run directory.

    full_output_dir = os.path.join(run_base_dir, f"run{next_run_number:04d}")
    # Ensure the directory exists
    os.makedirs(full_output_dir, exist_ok=True)
    
    # make subdir for permutations
    perm_output_dir = os.path.join(full_output_dir, "permutations")
    os.makedirs(perm_output_dir, exist_ok=True)
    # full_output_dir = os.path.join(full_output_dir, "permutations")
    # os.makedirs(full_output_dir, exist_ok=True)
    
    # make symlink to original simulation data
    origin_symlink_path = _symlink_to_origin_simulation(origin_sim_path, full_output_dir)
    
    return full_output_dir, origin_symlink_path, perm_output_dir

def run_sensitivity_analysis(origin_sim_path, proj_output_dir, **kwargs):
    """
    Run sensitivity analysis on the network model.
    This function is a placeholder for the actual sensitivity analysis logic.
    
    input:
    - origin_sim_path: str, path to the original simulation data file. Pkl or json file.
    """
    print("Running sensitivity analysis...")
    
    # init

    #validate that origin path has _data.pkl or _data.json file, _cfg.json file, and _netParams.json file.
    print(f"Validating original simulation data at: {origin_sim_path}")
    nph.validate_sim_cfg_and_net_params(origin_sim_path)
    
    # Generate the standard output directory for sensitivity analysis results
    print(f"Generating standard output directory for sensitivity analysis results in: {proj_output_dir}")
    overwrite = kwargs.get('overwrite', False)  # default to False, create a new run directory
    sa_run_dir, origin_symlink_path, perm_output_dir = _generate_standard_sa_run_dirs(origin_sim_path, proj_output_dir, overwrite=overwrite)
    print(f"Sensitivity analysis run directory created: {sa_run_dir}")
    print(f"Symlink to original simulation data created: {origin_symlink_path}")
    print(f"Permutations output directory created: {perm_output_dir}")
    #_symlink_to_origin_simulation(origin_sim_path, proj_output_dir)
    
    # unpack evolution parameters and levels
    #evol_params = kwargs.get('evol_params', None)
    assert kwargs.get('evol_path', None) is not None, "Evolution parameters path must be provided."
    evol_params = nph.import_evol_params(kwargs.get('evol_path', None))  # path to the evolution parameters file, if not specified, it will be derived from the simulation data path
    assert evol_params is not None, "Evolution parameters must be provided."
    print(f"Using evolution parameters and ranges:")
    pprint(evol_params)
    levels = kwargs.get('levels', None)  # default to 10 levels if not specified
    if levels is None:
        levels = 10  # default number of sensitivity analysis levels
        print("No levels specified, defaulting to 10 levels.")
    print(f"Number of sensitivity analysis levels: {levels}")
    
    # permute cfg and netParams
    print("Permuting cfg and netParams based on evolution parameters...")
    origin_cfg_path = origin_sim_path.replace("_data.pkl", "_cfg.json") if "_data.pkl" in origin_sim_path else origin_sim_path.replace("_data.json", "_cfg.json")
    origin_net_params_path = origin_sim_path.replace("_data.pkl", "_netParams.json") if "_data.pkl" in origin_sim_path else origin_sim_path.replace("_data.json", "_netParams.json")
    perm_levels = _define_permutation_levels(evol_params, origin_cfg_path, levels=levels)
    permutations = _generate_permutations(perm_levels, origin_cfg_path, origin_net_params_path, perm_output_dir, evol_params, 
                           #overwrite=overwrite
                           overwrite=False # easier for debugging rn.
                           )
    
    # unpack simulaiton runtime parameters
    run_sims = kwargs.get('run_sims', True)  # default to True, run simulations
    parallel = kwargs.get('parallel', True)  # default to True, run simulations in parallel
    mpi = kwargs.get('mpi', False)  # default to False
    slurm = kwargs.get('slurm', False)  # default to False
    
    # run permuted simulations
    init_path = kwargs.get('init_path', None)  # path to the init.py file that initializes the simulation environment, if not specified, it will be derived from the simulation data path
    tasks_per_node = kwargs.get('tasks_per_node', 1)  # default to 1 task per node, can be adjusted based on the number of cores available
    if run_sims:
        #print("Running permuted simulations...")
        print("Running simulations in parallel..." if parallel else "Running simulations sequentially...")
        print("Using MPI..." if mpi else "Not using MPI.")
        print("Using SLURM..." if slurm else "Not using SLURM.")
        nph.run_sims(
            permutations,
            perm_output_dir,
            #run_sims=run_sims,
            parallel=parallel,
            init_path=init_path,  # required for MPI or SLURM, path to the init.py file that initializes the simulation environment, if not specified, it will be derived from the simulation data path
            mpi=mpi,
            slurm=slurm,
            tasks_per_node=tasks_per_node,  # default to 1 task per node, can be adjusted based on the number of cores available
            overwrite=overwrite,  # overwrite existing simulation data
        )
    
    
    # analyze permuted simulations
    
    # reports
    
    #just a stopping point for debugging
    print()