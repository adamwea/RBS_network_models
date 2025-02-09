import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from concurrent.futures import ProcessPoolExecutor, as_completed
from netpyne import sim, specs
from RBS_network_models.CDKL5.DIV21.src.evol_params import params
from RBS_network_models.sim_analysis import process_simulation

'''run sensitivity analysis'''
def run_sensitivity_analysis(
    
    # input parameters
    sim_data_path, 
    output_path,
    reference_data_path = None,
    
    #plotting parameters
    plot = False,
    conv_params = None,
    fitnessFuncArgs = None,
    
    #sensitivity analysis parameters
    lower_bound=0.2,
    upper_bound=1.8,
    levels=2,
    duration_seconds=1,
    option='serial', #NOTE: options are 'serial' or 'parallel'
    num_workers=None, #NOTE: specify number of workers for parallel option, if None, will allocate as many as possible and distribute threads evenly
    debug=False,
    try_loading=True,
    ):
    
    #simulation output path
    output_path = os.path.join(output_path, 'simulations')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # generate permutations
    def generate_permutations(
        sim_data_path, 
        #evol_params_path, 
        saveFolder,
        lower_bound=0.2, 
        upper_bound=1.8, 
        levels=2,
        duration_seconds=1,
        verbose = False
        ):
        #from copy import deepcopy
        #from netpyne import sim    
        
        #cfg_permutations
        cfg_permutations = []
        
        #apparently need to modify simcfg before loading
        simConfig = sim.loadSimCfg(sim_data_path, setLoaded=False)
        
        #modify shared runtime options
        duration_seconds = duration_seconds
        simConfig.duration = 1e3 * duration_seconds  # likewise, I think it's necessary to modify netParams, not net.params or net
        simConfig.verbose = False
        #simConfig.verbose = True # NOTE: during connection formation, this will be VERY verbose
        simConfig.validateNetParams = True
        #simConfig.coreneuron = True
        simConfig.saveFolder=saveFolder
        simConfig.saveJson = False
        simConfig.savePickle = True
        #simConfig.coreneuron = True
        simConfig.cvode_active = False # make sure variable time step is off...not sure why it was ever on.
        simConfig.simLabel = '_'+simConfig.simLabel # NOTE this is only applied to the original simConfig 
                                                    # - only because it isnt overwritten later when generating permutations.
        
        # turn recordings off
        # remove recordCells from simConfig
        if hasattr(simConfig, 'recordCells'):
            delattr(simConfig, 'recordCells')
        
        # append original simConfig to permutations so that it is also run, plot, and saved with the others.
        # structure is maintained as a tuple to match the structure of the other permutations
        cfg_permutations.append((
            simConfig.__dict__.copy(), #cfg
            None, #permuted param
            None, #original value
            ))
        
        # load evol_params
        # evol_params_module = import_module_from_path(evol_params_path)
        # evol_params = evol_params_module.params
        #from DIV21.src.evol_params import params
        from RBS_network_models.CDKL5.DIV21.src.evol_params import params
        evol_params = params
        
        # generate permutations
        for evol_param, evol_val in evol_params.items():
            # for cfg_param, cfg_val in simConfig.items():
            #     if evol_param == cfg_param:
            if hasattr(simConfig, evol_param):
                cfg_param = evol_param
                cfg_val = getattr(simConfig, evol_param)
                #if evol_val is a list, then it's a range from min to max allowed for the parameter
                if isinstance(evol_val, list):
                    
                    #NOTE: it has occured to me that modifying certain params this way just isnt very practical or useful
                    # for example, modifying std of distribution for a param, would require getting all the values of
                    # the param, and somehow remapping them to the new std for each cell. 
                    # I dont think this would be very useful, and would be pretty complicated to implement.
                    # by contrast, if the mean of the distribution was modified, it would be much simpler to just 
                    # shift all the values by the same proportion.
                    
                    #the following if statement will skip over these kinds of params
                    excepted_param_keys = [
                        'std', # see rationale above
                        #'probLengthConst', # NOTE: this is included into a string that passed and evaluated in hoc,
                                            # 'probability': 'exp(-dist_3D / {})*{}'.format(cfg.probLengthConst, spec['prob']),
                                            # it's easier to modify probability to get a sense of how it affects the network
                                            
                                            # NOTE: nvm I figured it out. I can just modify the string directly.
                        ]
                    if any([key in cfg_param for key in excepted_param_keys]):
                        if verbose: print(f'Skipping permutations for {cfg_param}...')
                        continue
                    
                    if verbose: print(f'Generating permutations for {cfg_param}...')
                    
                    
                    # 2025-01-10 10:09:03 aw - original code. 2 levels of permutations, hard coded.
                    # going to implement code that will allow for any number of levels of permutations 
                    # between the lower and upper bounds.
                    
                    # #create two permutations of cfg in this param, 0.2 of the original, and 1.8 of the original
                    # cfg_permutation_1 = simConfig.__dict__.copy()
                    # cfg_permutation_1[cfg_param] = cfg_val * lower_bound
                    # cfg_permutation_1['simLabel'] = f'{cfg_param}_reduced'
                    # cfg_permutations.append((
                    #     cfg_permutation_1, #cfg
                    #     cfg_param, #permuted param
                    #     cfg_val, #original value                    
                    #     ))
                    
                    # cfg_permutation_2 = simConfig.__dict__.copy()
                    # cfg_permutation_2[cfg_param] = cfg_val * upper_bound
                    # cfg_permutation_2['simLabel'] = f'{cfg_param}_increased'
                    # cfg_permutations.append((
                    #     cfg_permutation_2,
                    #     cfg_param,
                    #     cfg_val,                    
                    #     ))
                    
                    
                    # 2025-01-10 10:10:08 aw new code. any number of levels of permutations between lower and upper bounds.
                    def append_permutation_levels(cfg_permutations, 
                                                  simConfig, 
                                                  cfg_param, 
                                                  cfg_val, 
                                                  lower_bound, 
                                                  upper_bound, 
                                                  levels):
                        """
                        Append permutations to the list of cfg permutations for a given parameter.
                        """
                        def permute_param(cfg_permutation, cfg_param, upper_bound, lower_bound, level, levels):                         
                            """
                            Handle special cases where the parameter should not be permuted.
                            """
                            # if verbose: print(f'Skipping permutations for {cfg_param}...')
                            # return cfg_permutations
                            # special cases
                            #print(f'Generating levels for {cfg_param}...')
                            if 'LengthConst' in cfg_param:
                                # got to typical case, this isnt actually a probability based param
                                # TODO: rename this to just length constant later.
                                upper_value = cfg_val * upper_bound
                                lower_value = cfg_val * lower_bound                       
                            elif 'prob' in cfg_param:
                                #modify upper and lower bounds such that probability based params 
                                #dont go below 0 or above 1
                                upper_value = cfg_val * upper_bound
                                lower_value = cfg_val * lower_bound
                                if upper_value > 1:
                                    upper_value = 1
                                if lower_value < 0:
                                    lower_value = 0
                                
                                # #calculate new upper and lower bounds to be used in the permutations
                                # upper_bound = 1 / cfg_val
                                # lower_bound = 0 / cfg_val
                            else:
                                #typical case
                                upper_value = cfg_val * upper_bound
                                lower_value = cfg_val * lower_bound
                                
                            # do two linspaces and stitch them together to ensure cfg_val is centered.
                            permuted_vals_1 = np.linspace(lower_value, cfg_val, levels // 2 + 1)[:-1] # returns all but the last value (exclude cfg_val)
                            permuted_vals_2 = np.linspace(cfg_val, upper_value, levels // 2 + 1)[1:] ## returns all but the first value (exclude cfg_val)
                            permuted_vals = np.concatenate((permuted_vals_1, permuted_vals_2))
                            
                            # quality
                            assert permuted_vals.size == levels, f'Expected {levels} permuted values, got {permuted_vals.size}'
                            assert np.all(np.diff(permuted_vals) > 0), f'Permuted values are not in ascending order: {permuted_vals}'
                            assert cfg_val > permuted_vals[levels//2-1], f'Permuted value {permuted_vals[levels//2-1]} is not less than original value {cfg_val}'
                            assert cfg_val < permuted_vals[levels//2], f'Permuted value {permuted_vals[levels//2]} is not greater than original value {cfg_val}'
                                
                            # return permuted value
                            #permuted_vals = np.linspace(lower_value, upper_value, levels)
                            permuted_val = permuted_vals[level]
                            cfg_permutation[cfg_param] = permuted_val
                            return cfg_permutation, permuted_vals
                        #print(f'Generating levels for {cfg_param}...')
                        for i in range(levels):
                            original_cfg = simConfig.__dict__.copy()
                            cfg_permutation = simConfig.__dict__.copy()
                            #temp_upper, temp_lower = upper_bound, lower_bound # save original values for future iterations
                            cfg_permutation, permuted_vals = permute_param(cfg_permutation, cfg_param, upper_bound, lower_bound, i, levels)
                            # cfg_permutation[cfg_param] = cfg_val * (lower_bound + (i * (upper_bound - lower_bound) / (levels - 1)))
                            # upper_bound, lower_bound = temp_upper, temp_lower # set back to original values for next iteration
                            cfg_permutation['simLabel'] = f'{cfg_param}_{i}'
                            cfg_permutations.append((
                                cfg_permutation,
                                cfg_param,
                                cfg_val,
                                ))
                            
                            # aw 2025-01-17 08:38:27 - adding validation controls to ensure permutations are correct
                            # seems like there's no issue in this step - however there may be issues in the net preprocessing
                            assert cfg_permutation[cfg_param] == permuted_vals[i], f'Failed to permute {cfg_param} to {permuted_vals[i]}'
                            assert original_cfg != cfg_permutation, f'Failed to permute {cfg_param} to {permuted_vals[i]}'
                        #print(f'Generated permuted vals: {permuted_vals}')
                        return cfg_permutations
                    
                    cfg_permutations = append_permutation_levels(cfg_permutations,
                                                simConfig,
                                                cfg_param,
                                                cfg_val,
                                                lower_bound,
                                                upper_bound,
                                                levels)
                    
                    if verbose: print(f'Permutations generated for {cfg_param}!')
        #debug
        #only keep cfgs where the simLabel contains 'gk' or 'gna'
        # #TODO: figure out the issue with these params
        # cfg_permutations = [cfg_permutation for cfg_permutation in cfg_permutations if 'gk' in cfg_permutation[0]['simLabel'] or 'gna' in cfg_permutation[0]['simLabel']]
        
        print(f'Generated {len(cfg_permutations)} cfg permutations.')
        return cfg_permutations
    cfg_permutations = generate_permutations(sim_data_path, 
                                             output_path,
                                             #evol_params_path, 
                                             #sensitivity_analysis_output_path, 
                                             lower_bound=lower_bound, 
                                             upper_bound=upper_bound,
                                             levels=levels,
                                             duration_seconds=duration_seconds
                                             )

    # run permutation, test individual perms as needed
    if option == 'serial':
        for perm_simConfig in cfg_permutations:
            try:
                run_permutation(
                    sim_data_path,
                    reference_data_path = reference_data_path,
                    plot = plot,
                    conv_params=conv_params,
                    fitnessFuncArgs=fitnessFuncArgs,
                    debug = debug, 
                    *perm_simConfig
                    )
            except Exception as e:
                print(f'Error running permutation {perm_simConfig["simLabel"]}: {e}')
                print('Continuing to next permutation...')
            #break
    elif option == 'parallel':
        # run all permutations, parallelized        
        # TODO: validate this function
        # NOTE: this one seems to work, the ones above, do not.
                # running this one for now to see if it works.
        def run_all_permutations(
            sim_data_path, 
            cfg_permutations, 
            plot=None, 
            reference_data_path=None, 
            num_workers=None,
            debug=False,
            conv_params=None,
            fitnessFuncArgs=None,
            try_loading=True      
            ):
            """
            Run all configuration permutations in parallel, limited by logical CPU availability.

            Args:
                sim_data_path (str): Path to simulation data.
                cfg_permutations (list): List of configuration permutations to run.
                plot: Plot-related information for `run_permutation` (optional).
                reference_data_path (str): Reference data path for `run_permutation` (optional).
            """
            import os
            from concurrent.futures import ProcessPoolExecutor, as_completed

            available_cpus = os.cpu_count()
            
            # set number of workers (i.e. processes, simultaneously running simulations)
            if num_workers is None:
                num_workers = min(len(cfg_permutations), available_cpus)
            else:
                num_workers = min(num_workers, available_cpus)
            print(f'Using {num_workers} workers out of {available_cpus} available CPUs.')

            # Evenly distribute threads among processes
            threads_per_worker = max(1, available_cpus // num_workers)
            os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
            os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
            os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
            os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)

            # Prepare tasks
            tasks = []
            for cfg_tuple in cfg_permutations:
                if isinstance(cfg_tuple, tuple) and len(cfg_tuple) == 3:
                    cfg, cfg_param, cfg_val = cfg_tuple
                    tasks.append((sim_data_path, 
                                  cfg, cfg_param, 
                                  cfg_val, 
                                  reference_data_path, 
                                  plot, 
                                  debug, 
                                  conv_params, 
                                  fitnessFuncArgs,
                                  try_loading,
                                  ))
                else:
                    raise ValueError(f"Unexpected structure in cfg_permutations: {cfg_tuple}")

            # Run tasks in parallel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(run_permutation, *task): task for task in tasks}

                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        future.result()  # This will raise any exceptions from the worker
                    except Exception as e:
                        cfg = task[1]  # Access the configuration from the task
                        sim_label = cfg.get("simLabel", "unknown") if isinstance(cfg, dict) else "unknown"
                        print(f"Unhandled exception in permutation {sim_label}: {e}")
        run_all_permutations(
            sim_data_path, 
            cfg_permutations, 
            plot=plot, 
            reference_data_path=reference_data_path, 
            num_workers=num_workers,
            debug=debug,
            conv_params=conv_params,
            fitnessFuncArgs=fitnessFuncArgs,
            try_loading=try_loading
            )

def run_permutation(
    sim_data_path, 
    cfg, 
    cfg_param, 
    cfg_val,
    reference_data_path = None,
    plot = False,
    debug = False,
    conv_params = None,
    fitnessFuncArgs = None,
    try_loading = True,
    *args
    ):
    
    if plot: assert reference_data_path is not None, "Reference data path must be provided for plotting."
    
    try: 
        simLabel = cfg['simLabel']
        print(f'Running permutation {simLabel}...')
        
        if cfg_param is not None and cfg_val is not None: #if none, then it's the original simConfig            
            prepare_permuted_sim(sim_data_path, 
                                 cfg, 
                                 cfg_param, 
                                 cfg_val, 
                                 try_loading=try_loading)
            
            # if try_loading is true, check if perpare_permuted_sim succeeded in loading all sim data
            # this should save a little time.
            if try_loading and hasattr(sim, 'allSimData'):
                pass
            else:            
                sim.runSim()                        # run parallel Neuron simulation
                sim.gatherData()                    # gather spiking data and cell info from each node
        elif not debug: # i.e. if original simConfig is being re-run for official 
                        # sensitivity analysis - confirming that the basic load and re-run method does
                        # in fact generate identical results.
            # #sim.clearAll()
            # if not debug_permuted_sims:
            sim.load(sim_data_path, simConfig=cfg)
            sim.runSim()                        # run parallel Neuron simulation
            sim.gatherData()                    # gather spiking data and cell info from each node
        elif debug: 
            print(f'Runing in debug mode. Only loading simConfig from {sim_data_path}')
            print(f'Simulation will not be re-run. it will, however, be saved to the expected location.')
            #if sim is None: #assume sim is already loaded
            sim.load(sim_data_path, simConfig=cfg)
            #else: pass
            print()
        else:
            print('Not sure how you got here... Something went wrong.')
                
        permuted_data_paths = sim.saveData()                      # save params, cell info and sim output to file (pickle,mat,txt,etc)
        assert len(permuted_data_paths) == 1, "Expected only one data path, the .pkl file. Got more."
        perm_sim_data_path = permuted_data_paths[0]
    except Exception as e:
        print(f'Error running permutation {simLabel}: {e}')
        return
        
    if plot:
        assert conv_params is not None, "Conversion parameters must be provided for plotting."
        assert fitnessFuncArgs is not None, "Fitness function arguments must be provided for plotting."
        try:        
            process_simulation(
                perm_sim_data_path, 
                reference_data_path, 
                DEBUG_MODE=debug,
                conv_params=conv_params,
                fitnessFuncArgs=fitnessFuncArgs,
                #try_loading=try_loading
                )
        except Exception as e:
            print(f'Error processing permutation {simLabel}: {e}')
            return
        
    print(f'Permutation {simLabel} successfully ran!')  

def prepare_permuted_sim(
    sim_data_path, 
    cfg, 
    cfg_param, 
    cfg_val,
    try_loading=True):
    
    #try loading sim data if possible
    if try_loading:
        try:
            expected_save_path = os.path.join(cfg['saveFolder'], f'{cfg["simLabel"]}_data.pkl')
            exists = os.path.exists(expected_save_path)
            if exists:
                print(f'Simulation data for {cfg["simLabel"]} already exists at {expected_save_path}. Attempting to load...')
                sim.load(expected_save_path)
                assert hasattr(sim, 'net'), "Simulation data loaded but 'net' attribute is missing."
                #assert hasattr(sim, 'netParams'), "Simulation data loaded but 'netParams' attribute is missing."
                assert hasattr(sim, 'cfg'), "Simulation data loaded but 'simConfig' attribute is missing."
                #assert cfg == sim.cfg, "Loaded simulation data does not match the expected configuration."
                assert cfg[cfg_param]==sim.cfg[cfg_param], "Loaded simulation data does not match the expected configuration."
                assert hasattr(sim, 'simData'), "Simulation data loaded but 'simData' attribute is missing."
                assert hasattr(sim, 'allSimData'), "Simulation data loaded but 'allSimData' attribute is missing."
                #assert hasattr(sim, 'allCellData'), "Simulation data loaded but 'allCellData' attribute is missing."
                print(f'Simulation data for {cfg["simLabel"]} loaded successfully.')
                return
        except Exception as e:
            print(f'Error loading simulation data for {cfg["simLabel"]}: {e}')
            print(f'Will attempt to run the simulation instead.')
            try: sim.clearAll()
            except: pass   # continue to run the simulation if loading fails
    
    # load netparams and permute
    sim.load(sim_data_path, simConfig=cfg)
    simConfig = specs.SimConfig(simConfigDict=cfg)
    netParams = sim.loadNetParams(sim_data_path, setLoaded=False)
    
    #Typical case:
    strategy = 'by_value'
    #SPECIAL CASES: gnabar, gkbar, L, diam, Ra
    handle_by_name = ['gnabar', 'gkbar', 'L', 'diam', 'Ra']
    if any([name in cfg_param for name in handle_by_name]): 
        if '_' in cfg_param:
            elements = cfg_param.split('_')
            for element in elements:
                if any([name==element for name in handle_by_name]):
                    strategy = 'by_name'
                    break
        elif any([name==cfg_param for name in handle_by_name]):
            strategy = 'by_name'

    cfg_to_netparams_mapping = map_cfg_to_netparams(
        {cfg_param: cfg_val}, 
        netParams.__dict__.copy(),
        strategy=strategy
        )
    mapped_paths = cfg_to_netparams_mapping[cfg_param]
    
    if mapped_paths is None:
        print(f"WARNING: mapped paths is None.")
        print(f"No paths found for {cfg_param} = {cfg_val}")
        return

    # update permuted params
    def getNestedParam(netParams, paramLabel):
        if '.' in paramLabel: 
            paramLabel = paramLabel.split('.')
        if isinstance(paramLabel, list ) or isinstance(paramLabel, tuple):
            container = netParams
            for ip in range(len(paramLabel) - 1):
                if hasattr(container, paramLabel[ip]):
                    container = getattr(container, paramLabel[ip])
                else:
                    container = container[paramLabel[ip]]
            return container[paramLabel[-1]]
    for mapped_path in mapped_paths:    
        current_val = getNestedParam(netParams, mapped_path)
        #assert cfg_val == current_val, f"Expected {cfg_val} but got {current_val}"
        try:
            if isinstance(current_val, str): #handle hoc strings
                assert str(cfg_val) in current_val, f"Expected {cfg_val} to be in {current_val}"
                updated_func = current_val.replace(str(cfg_val), str(cfg[cfg_param]))                    
                #netParams.setNestedParam(mapped_path, updated_func)
                before_val = getNestedParam(netParams, mapped_path)
                netParams.setNestedParam(mapped_path, updated_func)
                after_val = getNestedParam(netParams, mapped_path)
                assert before_val != after_val, f"Failed to update {mapped_path} from {before_val} to {after_val}"
                print(f"Updated {mapped_path} from {before_val} to {after_val}")
            elif strategy == 'by_name': #special case
                #assert cfg_val == current_val, f"Expected {cfg_val} but got {current_val}"
                original_val = cfg_val
                permuted_val = cfg[cfg_param]
                modifier = permuted_val / original_val  # NOTE: this should end up equal to one of the 
                                                        #       level multipliers
                #proportion = cfg[cfg_param] / current_val
                
                #adjust mean proportinally 
                #netParams.setNestedParam(mapped_path, current_val * modifier)
                
                before_val = getNestedParam(netParams, mapped_path)
                netParams.setNestedParam(mapped_path, current_val * modifier)
                after_val = getNestedParam(netParams, mapped_path)
                assert before_val != after_val, f"Failed to update {mapped_path} from {before_val} to {after_val}"
                print(f"Updated {mapped_path} from {before_val} to {after_val}")
            else:
                assert cfg_val == current_val, f"Expected {cfg_val} but got {current_val}"
                before_val = getNestedParam(netParams, mapped_path)
                netParams.setNestedParam(mapped_path, cfg[cfg_param])
                after_val = getNestedParam(netParams, mapped_path)
                assert before_val != after_val, f"Failed to update {mapped_path} from {before_val} to {after_val}"
                print(f"Updated {mapped_path} from {before_val} to {after_val}")  
        except:
            print(f'Error updating {mapped_path}: {e}')
            continue

    # remove previous data
    sim.clearAll()

    #remove mapping from netParams #TODO: figure out how to actually take advantage of this
    # if hasattr(netParams, 'mapping'):
    #     del netParams.mapping
    netParams.mapping = {}

    # run simulation
    # Create network and run simulation
    sim.initialize(                     # create network object and set cfg and net params
            simConfig = simConfig,          # pass simulation config and network params as arguments
            netParams = netParams)
    sim.net.createPops()                # instantiate network populations
    sim.net.createCells()               # instantiate network cells based on defined populations
    sim.net.connectCells()              # create connections between cells based on params
    sim.net.addStims()                  # add stimulation (usually there are none)
    sim.setupRecording()                # setup variables to record for each cell (spikes, V traces, etc)

def map_cfg_to_netparams(simConfig, netParams, strategy='by_value'):
    """
    Map attributes in simConfig to their corresponding locations in netParams based on values.
    
    Parameters:
        simConfig (dict): The configuration dictionary (cfg).
        netParams (object): The network parameters object.
    
    Returns:
        dict: A mapping from simConfig parameters to their paths in netParams.
    """
    def find_value_in_netparams(value, netParams, current_path=""):
        """
        Recursively search for the value in netParams and return a list of matching paths.
        
        Parameters:
            value (any): The value to search for.
            netParams (object): The network parameters object.
            current_path (str): The current path in the recursive search.
        
        Returns:
            list: A list of paths to the matching value.
        """
        stack = [(netParams, current_path)]  # Stack for backtracking, contains (current_object, current_path)
        matching_paths = []  # To store all matching paths

        while stack:
            current_obj, current_path = stack.pop()
            
            # if 'connParams' in current_path:  # Debugging: specific context output
            #     print('found connParams')
            #     if 'I->E' in current_path:
            #         print('found I->E')

            if isinstance(current_obj, dict):
                for key, val in current_obj.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    if val == value:
                        matching_paths.append(new_path)
                    elif isinstance(val, str):  # Handle HOC string matches
                        if str(value) in val:
                            matching_paths.append(new_path)
                    elif isinstance(val, (dict, list)):
                        stack.append((val, new_path))  # Push deeper layer onto stack

            elif isinstance(current_obj, list):
                for i, item in enumerate(current_obj):
                    new_path = f"{current_path}[{i}]"
                    if item == value:
                        matching_paths.append(new_path)
                    elif isinstance(item, str):  # Handle HOC string matches
                        if str(value) in item:
                            matching_paths.append(new_path)
                    elif isinstance(item, (dict, list)):
                        stack.append((item, new_path))  # Push list item onto stack

        return matching_paths  # Return all matching paths
    
    def find_name_in_netparams(name, netParams, current_path=""):
        """
        Recursively search for the name in netParams and return a list of matching paths.
        
        Parameters:
            name (str): The name to search for.
            netParams (object): The network parameters object.
            current_path (str): The current path in the recursive search.
        
        Returns:
            list: A list of paths to the matching name.
        """
        stack = [(netParams, current_path)]
        
        if '_' in name:
            elements = name.split('_')
            try: assert 'E' in elements or 'I' in elements
            except: elements = None
        else:
            elements = None
        
        matching_paths = []
        while stack:
            current_obj, current_path = stack.pop()
            
            # if 'cellParams' in current_path:  # Debugging: specific context output
            #     print('found cellParams')
                
            # if 'gnabar' in current_path:  # Debugging: specific context output
            #     print('found gnabar')
                
            # elements=None
            # #if _ in 
            
            if elements is not None:
                if isinstance(current_obj, dict):
                    for key, val in current_obj.items():
                        new_path = f"{current_path}.{key}" if current_path else key
                        if all([element in new_path for element in elements]):
                            matching_paths.append(new_path)
                        elif isinstance(val, (dict, list)):
                            stack.append((val, new_path))
                elif isinstance(current_obj, list):
                    for i, item in enumerate(current_obj):
                        new_path = f"{current_path}[{i}]"
                        if all([element in new_path for element in elements]):
                            matching_paths.append(new_path)
                        elif isinstance(item, (dict, list)):
                            stack.append((item, new_path))
                
            elif isinstance(current_obj, dict):
                for key, val in current_obj.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    #if key == name:
                    if key == name:
                        matching_paths.append(new_path)
                    elif isinstance(val, (dict, list)):
                        stack.append((val, new_path))
            elif isinstance(current_obj, list):
                for i, item in enumerate(current_obj):
                    new_path = f"{current_path}[{i}]"
                    if item == name:
                        matching_paths.append(new_path)
                    elif isinstance(item, (dict, list)):
                        stack.append((item, new_path))
        return matching_paths
        
    # Generate the mapping
    mapping = {}
    for param, value in simConfig.items():
        if strategy == 'by_name':
            #paths = find_value_in_netparams(param, netParams)
            paths = find_name_in_netparams(param, netParams)
        elif strategy == 'by_value':
            #paths = find_name_in_netparams(value, netParams)
            paths = find_value_in_netparams(value, netParams)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        mapping[param] = paths if paths else None  # Assign None if no path is found

    return mapping

''' plot sensitivity analysis results'''
def plot_sensitivity_analysis(
    og_simulation_data_path, 
    sensitvity_analysis_output_dir,
    num_workers=None,
    burst_rates=None,
    original_burst_rate=None,
    format_option='long',
    levels=6,
    plot_grid=True,
    plot_heatmaps=True
    ):
    
    plot_sensitivity_grid_plots(
        og_simulation_data_path,
        sensitvity_analysis_output_dir,
        num_workers=num_workers,
        levels=levels,
        plot_grid=plot_grid,
        plot_heatmaps=plot_heatmaps,      
    )

def load_network_metrics(input_dir, og_simulation_data_path, num_workers=None):
    network_metrics_files = glob.glob(input_dir + '/*network_metrics.npy')
    #burst_rates = {}
    network_metrics_data = {}
    
    # set number of workers (i.e. processes, simultaneously running simulations)
    available_cpus = os.cpu_count()
    if num_workers is None:
        num_workers = min(len(network_metrics_files), available_cpus)
    else:
        num_workers = min(num_workers, len(network_metrics_files), available_cpus)
    print(f'Using {num_workers} workers to load {len(network_metrics_files)} network metrics files.')

    # Evenly distribute threads among processes
    threads_per_worker = max(1, available_cpus // num_workers)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
        
    total_files = len(network_metrics_files)
    completed_files = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(metrics_loader, file) for file in network_metrics_files]
        for future in as_completed(futures):
            result = future.result()
            completed_files += 1
            print(f"Completed {completed_files} out of {total_files} processes") #NOTE: there's a delay between the completion of the process and return of the result
        results = [future.result() for future in futures]
        
    # aw 2025-01-22 11:39:49 - does this work beetter if I do threads instead of processes?
    # from concurrent.futures import ThreadPoolExecutor
    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [executor.submit(metrics_loader, file) for file in network_metrics_files]
    #     for future in as_completed(futures):
    #         result = future.result()
    #         completed_files += 1
    #         print(f"Completed {completed_files} out of {total_files} processes")
    #     #results = list(executor.map(metrics_loader, network_metrics_files))
    #     results = [future.result() for future in futures]
    
    # aw 2025-01-13 11:14:09 - return network metrics instead of burst rates
    for i, (data, summary_plot, basename) in enumerate(results):
        if data is not None:
            network_metrics_data[basename] = {'summary_plot': summary_plot, 'data': data}
            
    return network_metrics_data    

def metrics_loader(network_metrics_file):
    """
    Helper function to process network metrics files and extract burst rate information.
    """
    try:
        start = time.time()
        basename = os.path.basename(network_metrics_file)
        #remove file extension from basename
        basename = os.path.splitext(basename)[0]
        #print('Loading', basename, '...')
        data = np.load(network_metrics_file, allow_pickle=True).item()
        #mean_burst_rate = data['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Rate']
        summary_plot = network_metrics_file.replace('network_metrics.npy', 'summary_plot.png')
        
        # #just for debug
        # from netpyne import sim
        # sim_data_path = network_metrics_file.replace('_network_metrics.npy', '_data.pkl')
        # # netParams = sim.loadNetParams(sim_data_path, setLoaded=False)
        # # net = sim.loadNet(sim_data_path)
        # sim.load(sim_data_path)
        # netParams = sim.net.params
        # #just for debug #TODO: not sure if some changes to netparams didnt get applied before running the simulation
        # # TODO: Seems like some changes to netParams didn't get applied before running the simulation - at least tau params are not being applied
        
        # aw 2025-01-13 11:44:00 - sending the whole data object back is going really slow... I guess I should curate the data object to only include the necessary information
        print('Curating data from network_metrics file...')
        # oh it looks like I'm sending the whole simulation data object - in addition to network metrics data - back to the main function. I should curate the data object to only include the necessary information.
        curated_data = { #yea this is going quite a bit faster. Dont neet all the individual data objects, just the network metrics and simData.
            'simConfig': data['simConfig'],
            'simData': data['simData'],
            'network_metrics': data['network_metrics'] 
        }  
        print('Loaded', basename, 'in', round(time.time() - start, 2), 'seconds.')
        return curated_data, summary_plot, basename
    except Exception as e:
        print('Error loading', network_metrics_file, ':', e)
        return None, None, basename

def plot_sensitivity_grid_plots(
    og_simulation_data_path, 
    sensitvity_analysis_output_dir,
    num_workers=None,
    burst_rates=None,
    original_burst_rate=None,
    format_option='long',
    levels=6,
    plot_grid=True,
    plot_heatmaps=True
    ):
    """
    Plots a grid of summary plots with color-modulated cells based on changes in burst rates.
    """
    
    # Set up paths and parameters
    #output_dir = sensitvity_analysis_output_dir
    input_dir = os.path.join(sensitvity_analysis_output_dir, 'simulations')
    output_dir = os.path.join(sensitvity_analysis_output_dir, 'summary_plots')
    sim_data_path = og_simulation_data_path
    
    # assertions
    assert os.path.exists(input_dir), f'Input directory {input_dir} does not exist.'
    assert os.path.exists(sim_data_path), f'Simulation data path {sim_data_path} does not exist.'
    
    # dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # get paths in grid format
    def get_clean_grid(
        input_dir,
        query = 'summary_plot.png'
        ):
        # find the original summary plot
        files = []
        for file in os.listdir(input_dir):
            if file.startswith('_') and query in file:
                og_sumamry_path = os.path.join(input_dir, file)
                files.append(os.path.join(input_dir, file))
                print('Found original summary plot:', og_sumamry_path)
                break
        assert len(files) == 1, f'Expected 1 original summary plot, found {len(files)}'
        
        # iterate through params, load associated plots, build a grid of paths
        grid = {} #store png paths here
        for param_idx, param in enumerate(params):
            
            #check if param value is a list, or tuple, of two values - if so dont skip, else skip
            param_val = params[param]
            if not isinstance(param_val, (list, tuple)):
                #print('skipping', param)
                continue
            #print() #print a line to separate outputs
            
            # Arrange data into a grid
            def get_perms_per_param(param):
                
                # init file list
                files = []
                
                #iterate through files in input_dir, get number of permutations from filename context
                num_permutations = 0
                param_elements = param.split('_')
                
                # iterate through files in input_dir, get number of permutations from filename context
                for file in os.listdir(input_dir):
                    
                    # check if slide was generated for this param - indicateing all plotting was done
                    # if not, skip
                    # if not 'comparison_summary_slide.png' in file:
                    if not query in file:
                        continue
                    
                    #print('is param in file?', param, file)
                    file_elements = file.split('_')
                    if all([param_element in file_elements for param_element in param_elements]):
                        num_permutations += 1
                        files.append(os.path.join(input_dir, file))
                        print('Found permutation for', param, 'in', file)
                        
                # debug - print number of permutations found
                # if num_permutations>0:
                #     print('Found', num_permutations, 'permutations for', param)
                
                # return number of permutations found
                return num_permutations, files
            num_permutations, summary_paths = get_perms_per_param(param)
            grid[param] = {}
            middle_idx = num_permutations // 2
            #insert og_summary_plot in the middle of summary_paths list
            if len(summary_paths) > 0:
                summary_paths.insert(middle_idx, og_sumamry_path)
                for idx, slide_path in enumerate(summary_paths):
                    #if idx < middle_idx or idx > middle_idx:
                    grid[param][idx] = slide_path
                    # elif idx == middle_idx:
                    #     grid[param][idx] = og_sumamry_path
                    #     idx = idx + 1
            print('num_permutations:', num_permutations)
            if num_permutations == 0: continue
            print() #print a line to separate outputs
            
            # quality check - make sure number of permutations is less than or equal to levels
            try:
                assert num_permutations <= levels, f'Expected {levels} permutations, found {num_permutations}'
            except Exception as e:
                print('Error:', e)                
            
        # remove empty rows
        clean_grid = {param: summary_paths for param, summary_paths in grid.items() if len(summary_paths) > 0}
        return clean_grid

    # Collect network_metrics.npy files and process
    def plot_summary_grid(
        output_dir,
        num_workers=None,
        #burst_rates=None,
        # original_burst_rate=None,
        # format_option = 'long' # aw 2025-01-11 17:02:34 - retiring this option
        format_option = 'matrix'
        ):
        
        # Plot summary grid
        print('Plotting summary grid')        
        
        if format_option == 'matrix':
            '''
            #arrange data into a grid of plots
            # y axis = params - any number of parameters that were varied in the simulation
            # x axis =  simulation permutations - usually 2-6 permutations of the simulation (ideally an even number) 
                        # + 1 column for the original simulation in the middle. (which is why an even number of permutations is nice)
            '''
            
            # get dict of paths for matrix
            clean_grid = get_clean_grid(output_dir)
            
            # Create a grid of plots
            n_rows = len(clean_grid)
            n_cols = levels+1
            #fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 7.5 * n_rows))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
            for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                for col_idx, summary_path in summary_paths.items():
                    try:
                        img = mpimg.imread(summary_path)
                        axs[row_idx, col_idx].imshow(img)
                        axs[row_idx, col_idx].axis('off')
                        axs[row_idx, col_idx].set_title(f'param: {param} (perm: {col_idx})', fontsize=14)
                    except Exception as e:
                        print('Error loading plot:', e)
                print(f'Plotted {param} in row {row_idx}')
                        
            # Save the plot
            print('Saving summary grid to', output_dir + '/_summary_grid.png')
            plt.tight_layout()
            plt.savefig(output_dir + '/_summary_grid.png', dpi=100)
            output_path = os.path.join(output_dir, '_summary_grid.png')
            print('done.')
            
            # Return original burst rate, burst rates, and output path
            # return original_burst_rate, burst_rates, output_path
            return output_path, clean_grid, original_burst_rate
        
        # reject unknown format options
        else:
            raise ValueError(f"Unknown format_option: {format_option}")    
    if plot_grid:
        summary_grid_path, clean_grid, original_burst_rate = plot_summary_grid(
        #sim_data_path,
        output_dir,
        num_workers=num_workers,
        #burst_rates=burst_rates,
        #original_burst_rate=original_burst_rate,
        )
        
    # aw 2025-01-14 08:09:53 - depreciating the above fuc, going to copy paste it and make changes. New one will do more than just burst rates
    def plot_heat_maps(
        output_dir,
        input_dir,
        num_workers=None,
        levels=6,
        ):
        
        # get dict of paths for matrix
        clean_grid = get_clean_grid(
            input_dir,
            query='network_metrics.npy'
            )
        
        # Collect network_metrics.npy files and process #NOTE: parallel processing is used here
        #if burst_rates is None or original_burst_rate is None:
        network_metrics_data = load_network_metrics(
            input_dir, 
            sim_data_path,
            num_workers=num_workers,
            )

        # # plot mean_Burst_Rate heatmap
        # def plot_mean_Burst_Rate(output_dir):
        #     # Plot a grid of parameters with burst rate changes
        #     print('Plotting summary grid with color gradient')   
        
        #     # original burst rate, find a key with '_' in basename
        #     for key in network_metrics_data.keys():
        #         if key.startswith('_'):
        #             original_key = key
        #             original_burst_rate = network_metrics_data[key]['data']['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Rate']
        #             break
                
        #     # get min and max burst rates
        #     min_burst_rate = 0 #initialize min_burst_rate
        #     max_burst_rate = 0 #initialize max_burst_rate
        #     for key in network_metrics_data.keys():
        #         data = network_metrics_data[key]['data']
        #         burst_rate = data['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Rate']
        #         if burst_rate < original_burst_rate and burst_rate < min_burst_rate:
        #             min_burst_rate = burst_rate
        #         if burst_rate > original_burst_rate and burst_rate > max_burst_rate:
        #             max_burst_rate = burst_rate
                    
                    
        #     # create a color gradient using min, max, and original burst rates. 
        #     # closer to original = more white
        #     # closer to min = more blue
        #     # closer to max = more red
        #     colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
        #     cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)

        #     # create a norm object to center the color gradient around the original burst rate
        #     norm = mcolors.CenteredNorm(
        #         vcenter=original_burst_rate, 
        #         #halfrange=max(abs(min_burst_rate - original_burst_rate), abs(max_burst_rate - original_burst_rate)))
        #         halfrange = min(abs(min_burst_rate - original_burst_rate), abs(max_burst_rate - original_burst_rate))
        #         )
            
        #     #prepare data dicts in each grid cell
        #     for param, summary_paths in clean_grid.items():
        #         clean_grid[param]['data'] = {}
                    
        #     # add network metrics data to clean grid
        #     for param, summary_paths in clean_grid.items():
        #         # if clean_grid[param]['data'] is None:
        #         #     clean_grid[param]['data'] = {}
        #         for key, data in network_metrics_data.items():
        #             if param in key:
        #                 clean_grid[param]['data'].update({key: data})
        #                 #break
        #     #print('Added network metrics data to clean grid')
            
        #     # first, heatmap burst rates
        #     n_rows = len(clean_grid)
        #     n_cols = levels+1
        #     fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1 * n_rows))
        #     for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                
        #         row_data = clean_grid[param]['data']
        #         #sort row_data by key
        #         sorted_row_data = dict(sorted(row_data.items()))
                
        #         # insert original summary plot in the middle of row_data
        #         middle_idx = len(sorted_row_data) // 2
        #         new_row_data = {}
        #         for idx, (key, value) in enumerate(sorted_row_data.items()):
        #             if idx == middle_idx:
        #                 new_row_data['original_data'] = network_metrics_data[original_key]
        #             new_row_data[key] = value
                
        #         clean_grid[param]['data'] = new_row_data
        #         row_data = clean_grid[param]['data']            
                
        #         #plot each cell in the row
        #         for col_idx, (key, data) in enumerate(row_data.items()):
        #             try:
        #                 #print()
                        
        #                 burst_rate = clean_grid[param]['data'][key]['data']['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Rate']
                        
        #                 # # Set color based on burst rate
        #                 # norm_burst_rate = (burst_rate - min_burst_rate) / (original_burst_rate)
        #                 # if burst_rate == original_burst_rate:
        #                 #     color = (1, 1, 1)  # white for original burst rate
        #                 # else:
        #                 #     colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        #                 #     cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
        #                 #     color = cmap(norm_burst_rate)
                        
        #                 # set color based on burst rate
        #                 # norm_burst_rate = (burst_rate) / (original_burst_rate)
        #                 # color = cmap(norm_burst_rate)
                        
        #                 color = cmap(norm(burst_rate))
                        
        #                 axs[row_idx, col_idx].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        #                 axs[row_idx, col_idx].text(0.5, 0.5, f'{burst_rate:.2f}', ha='center', va='center', fontsize=12)
        #                 axs[row_idx, col_idx].axis('off')
        #                 #axs[row_idx, col_idx].set_title(f'param: {param} (perm: {col_idx})', fontsize=18)
        #                 permuted_param = param
        #                 permuted_value = data['data']['simConfig'][param]
        #                 #axs[row_idx, col_idx].set_title(f'(perm: {col_idx})', fontsize=14)
        #                 #round permuted value to 3 decimal places
        #                 try: permuted_value = round(permuted_value, 3)
        #                 except: pass
        #                 axs[row_idx, col_idx].set_title(f'@{permuted_value}', fontsize=14)
        #             except Exception as e:
        #                 print('Error loading plot:', e)
        #         print(f'Plotted {param} in row {row_idx}')
            
        #     #tight layout
        #     plt.tight_layout()
            
        #     #reveal y axis labels on the left side of the figure
        #     plt.subplots_adjust(left=0.15)
            
        #     # Add space to the right of the figure for the color bar
        #     fig.subplots_adjust(right=.90)
            
        #     # Add space at the time for title
        #     fig.subplots_adjust(top=0.925)
            
        #     #
        #     for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                
        #         # Get the position of the leftmost subplot
        #         pos = axs[row_idx, 0].get_position()
        #         x = pos.x0 - 0.025

        #         # Add row title to the left of the row
        #         fig.text(x, pos.y0 + pos.height / 2, param, va='center', ha='right', fontsize=14, rotation=0)
            
        #     # Add a color bar legend
        #     #sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_burst_rate, vmax=max_burst_rate))
        #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        #     sm.set_array([])
        #     #cbar_ax = fig.add_axes([0.86, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        #     cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        #     cbar = fig.colorbar(sm, cax=cbar_ax)
        #     #cbar.set_label('Burst Rate')
            
        #     # Add title
        #     metric = 'mean_Burst_Rate' #HACK: hardcoding this for now
        #     fig.suptitle(f'heatmap: {metric}', fontsize=16)
            
        #     # Save the plot
        #     print('Saving heat map to', output_dir + '/_BR_heat_map.png')
        #     plt.savefig(output_dir + '/_BR_heat_map.png', dpi=100)
        #     output_path = os.path.join(output_dir, '_BR_heat_map.png')
        #     print('done.')
            
        #     # return output_path
        #     return output_path
        # BR_heatmap_path = plot_mean_Burst_Rate(output_dir)

        # # plot mean_Burst_amplitude heatmap
        # def plot_mean_Burst_Peak(output_dir):
        #     # Plot a grid of parameters with burst rate changes
        #     print('Plotting summary grid with color gradient')   
        
        #     # original burst rate, find a key with '_' in basename
        #     for key in network_metrics_data.keys():
        #         if key.startswith('_'):
        #             original_key = key
        #             original_burst_peak = network_metrics_data[key]['data']['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        #             break
                
        #     # get min and max burst rates
        #     min_burst_peak = 0
        #     max_burst_peak = 0
        #     for key in network_metrics_data.keys():
        #         data = network_metrics_data[key]['data']
        #         burst_peak = data['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        #         if burst_peak < original_burst_peak and burst_peak < min_burst_peak:
        #             min_burst_peak = burst_peak
        #         if burst_peak > original_burst_peak and burst_peak > max_burst_peak:
        #             max_burst_peak = burst_peak
                    
        #     # create a color gradient using min, max, and original burst rates.
        #     colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        #     cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
            
        #     # create a norm object to center the color gradient around the original burst rate
        #     norm = mcolors.CenteredNorm(
        #         vcenter=original_burst_peak, 
        #         halfrange = min(abs(min_burst_peak - original_burst_peak), abs(max_burst_peak - original_burst_peak))
        #         )
            
        #     #prepare data dicts in each grid cell
        #     for param, summary_paths in clean_grid.items():
        #         clean_grid[param]['data'] = {}
                
        #     # add network metrics data to clean grid
        #     for param, summary_paths in clean_grid.items():
        #         for key, data in network_metrics_data.items():
        #             if param in key:
        #                 clean_grid[param]['data'].update({key: data})
                        
        #     # heat map for mean_Burst_Peak
        #     n_rows = len(clean_grid)
        #     n_cols = levels+1
        #     fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1 * n_rows))
        #     for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                
        #         row_data = clean_grid[param]['data']
        #         #sort row_data by key
        #         sorted_row_data = dict(sorted(row_data.items()))
                
        #         # insert original summary plot in the middle of row_data
        #         middle_idx = len(sorted_row_data) // 2
        #         new_row_data = {}
        #         for idx, (key, value) in enumerate(sorted_row_data.items()):
        #             if idx == middle_idx:
        #                 new_row_data['original_data'] = network_metrics_data[original_key]
        #             new_row_data[key] = value
                
        #         clean_grid[param]['data'] = new_row_data
        #         row_data = clean_grid[param]['data']            
                
        #         #plot each cell in the row
        #         for col_idx, (key, data) in enumerate(row_data.items()):
        #             try:
        #                 burst_peak = clean_grid[param]['data'][key]['data']['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        #                 color = cmap(norm(burst_peak))
        #                 axs[row_idx, col_idx].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        #                 axs[row_idx, col_idx].text(0.5, 0.5, f'{burst_peak:.2f}', ha='center', va='center', fontsize=12)
        #                 axs[row_idx, col_idx].axis('off')
        #                 permuted_param = param
        #                 permuted_value = data['data']['simConfig'][param]
        #                 try: permuted_value = round(permuted_value, 3)
        #                 except: pass
        #                 axs[row_idx, col_idx].set_title(f'@{permuted_value}', fontsize=14)
        #             except Exception as e:
        #                 print('Error loading plot:', e)
        #         print(f'Plotted {param} in row {row_idx}')
                
        #     plt.tight_layout()
        #     plt.subplots_adjust(left=0.15)
        #     fig.subplots_adjust(right=.90)
        #     fig.subplots_adjust(top=0.925)
        #     for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
        #         pos = axs[row_idx, 0].get_position()
        #         x = pos.x0 - 0.025
        #         fig.text(x, pos.y0 + pos.height / 2, param, va='center', ha='right', fontsize=14, rotation=0)
        #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        #     sm.set_array([])
        #     cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
        #     cbar = fig.colorbar(sm, cax=cbar_ax)
        #     metric = 'mean_Burst_Peak'
        #     fig.suptitle(f'heatmap: {metric}', fontsize=16)
        #     print('Saving heat map to', output_dir + '/_BP_heat_map.png')
        #     plt.savefig(output_dir + '/_BP_heat_map.png', dpi=100)
        #     output_path = os.path.join(output_dir, '_BP_heat_map.png')
        #     print('done.')
            
        #     return output_path
        # BP_heatmap_path = plot_mean_Burst_Peak(output_dir)

        # # plot fano_factor
        # def plot_fano_factor(output_dir):
        #     # Plot a grid of parameters with fano factor changes
        #     print('Plotting summary grid with color gradient')
            
        #     # original fano factor, find a key with '_' in basename
        #     for key in network_metrics_data.keys():
        #         if key.startswith('_'):
        #             original_key = key
        #             original_fano_factor = network_metrics_data[key]['data']['network_metrics']['bursting_data']['bursting_summary_data']['fano_factor']
        #             break
                
        #     # get min and max fano factors
        #     min_fano_factor = 0
        #     max_fano_factor = 0
        #     for key in network_metrics_data.keys():
        #         data = network_metrics_data[key]['data']
        #         fano_factor = data['network_metrics']['bursting_data']['bursting_summary_data']['fano_factor']
        #         if fano_factor < original_fano_factor and fano_factor < min_fano_factor:
        #             min_fano_factor = fano_factor
        #         if fano_factor > original_fano_factor and fano_factor > max_fano_factor:
        #             max_fano_factor = fano_factor
                    
        #     # create a color gradient using min, max, and original fano factors.
        #     colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        #     cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
            
        #     # create a norm object to center the color gradient around the original fano factor
        #     norm = mcolors.CenteredNorm(
        #         vcenter=original_fano_factor, 
        #         halfrange = min(abs(min_fano_factor - original_fano_factor), abs(max_fano_factor - original_fano_factor))
        #         )
            
        #     #prepare data dicts in each grid cell
        #     for param, summary_paths in clean_grid.items():
        #         clean_grid[param]['data'] = {}
                
        #     # add network metrics data to clean grid
        #     for param, summary_paths in clean_grid.items():
        #         for key, data in network_metrics_data.items():
        #             if param in key:
        #                 clean_grid[param]['data'].update({key: data})
                        
        #     # heat map for fano factor
        #     n_rows = len(clean_grid)
        #     n_cols = levels+1
        #     fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1 * n_rows))
        #     for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                
        #         row_data = clean_grid[param]['data']
        #         #sort row_data by key
        #         sorted_row_data = dict(sorted(row_data.items()))
                
        #         # insert original summary plot in the middle of row_data
        #         middle_idx = len(sorted_row_data) // 2
        #         new_row_data = {}
        #         for idx, (key, value) in enumerate(sorted_row_data.items()):
        #             if idx == middle_idx:
        #                 new_row_data['original_data'] = network_metrics_data[original_key]
        #             new_row_data[key] = value
                
        #         clean_grid[param]['data'] = new_row_data
        #         row_data = clean_grid[param]['data']            
                
        #         #plot each cell in the row
        #         for col_idx, (key, data) in enumerate(row_data.items()):
        #             try:
        #                 fano_factor = clean_grid[param]['data'][key]['data']['network_metrics']['bursting_data']['bursting_summary_data']['fano_factor']
        #                 color = cmap(norm(fano_factor))
        #                 axs[row_idx, col_idx].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        #                 axs[row_idx, col_idx].text(0.5, 0.5, f'{fano_factor:.2f}', ha='center', va='center', fontsize=12)
        #                 axs[row_idx, col_idx].axis('off')
        #                 permuted_param = param
        #                 permuted_value = data['data']['simConfig'][param]
        #                 try: permuted_value = round(permuted_value, 3)
        #                 except: pass
        #                 axs[row_idx, col_idx].set_title(f'@{permuted_value}', fontsize=14)
        #             except Exception as e:
        #                 print('Error loading plot:', e)
        #         print(f'Plotted {param} in row {row_idx}')

        #     plt.tight_layout()
        #     plt.subplots_adjust(left=0.15)
        #     fig.subplots_adjust(right=.90)
        #     fig.subplots_adjust(top=0.925)
        #     for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
        #         pos = axs[row_idx, 0].get_position()
        #         x = pos.x0 - 0.025
        #         fig.text(x, pos.y0 + pos.height / 2, param, va='center', ha='right', fontsize=14, rotation=0)
                
        #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        #     sm.set_array([])
        #     cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
        #     cbar = fig.colorbar(sm, cax=cbar_ax)
        #     metric = 'fano_factor'
        #     fig.suptitle(f'heatmap: {metric}', fontsize=16)
        #     print('Saving heat map to', output_dir + '/_FF_heat_map.png')
        #     plt.savefig(output_dir + '/_FF_heat_map.png', dpi=100)
        #     output_path = os.path.join(output_dir, '_FF_heat_map.png')
        #     print('done.')
            
        #     return output_path
        # FF_heatmap_path = plot_fano_factor(output_dir)
        
        # aw 2025-01-21 16:33:28 developing generalized function to plot heat maps for any metric
        # import os
        # import matplotlib.pyplot as plt
        # import matplotlib.colors as mcolors

        def plot_metric_heatmap(output_dir, metric_path, metric_name, network_metrics_data, clean_grid, levels):
            """
            Generalized function to plot heatmaps for a specified network metric.
            
            Args:
                output_dir (str): Directory to save the heatmap.
                metric_path (list): List of keys to navigate the metric in the network_metrics_data dictionary.
                metric_name (str): Name of the metric to display in the title and filename.
                network_metrics_data (dict): Dictionary containing network metrics data.
                clean_grid (dict): Dictionary of parameters and their data paths.
                levels (int): Number of levels for each parameter.
            """
            print(f"Plotting summary grid for {metric_name} with color gradient")
            
            # Find the original metric value
            for key in network_metrics_data.keys():
                if key.startswith('_'):
                    original_key = key
                    original_metric = network_metrics_data[key]['data']
                    for path_part in metric_path:
                        original_metric = original_metric[path_part]
                    break
            
            # Determine min and max metric values
            metric_list = []  # Initialize list to store metric values
            min_metric = float('inf')
            max_metric = float('-inf')
            for key in network_metrics_data.keys():
                data = network_metrics_data[key]['data']
                metric_value = data
                for path_part in metric_path:
                    metric_value = metric_value[path_part]
                    #print(metric_value)
                metric_list.append(float(metric_value))
                min_metric = min(min_metric, metric_value)
                max_metric = max(max_metric, metric_value)
            
            # get min and max metric values within 2 std deviations to avoid outliers
            std_dev = np.std(metric_list)
            max_val = original_metric + 2 * std_dev
            min_val = original_metric - 2 * std_dev
            
            # now if min and max arre within 2 std deviations, use them, else use the std values
            min_metric = max(min_metric, min_val)
            max_metric = min(max_metric, max_val)
            
            # Define colormap and normalization
            colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
            cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
            
            # Handle the case where original_metric is NaN
            if not np.isnan(original_metric):
                #typical case
                norm = mcolors.CenteredNorm(vcenter=original_metric, halfrange=max(abs(min_metric - original_metric), abs(max_metric - original_metric)))
            else:
                # handle case where original_metric is NaN
                norm = mcolors.Normalize(vmin=min_metric, vmax=max_metric) # normalized without centering around original simulation
                
            # Prepare data dicts for clean_grid
            for param, summary_paths in clean_grid.items():
                clean_grid[param]['data'] = {}
            for param, summary_paths in clean_grid.items():
                for key, data in network_metrics_data.items():
                    if param in key:
                        clean_grid[param]['data'].update({key: data})
            
            # Generate heatmap
            n_rows = len(clean_grid)
            n_cols = levels + 1
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1 * n_rows))
            
            for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                row_data = clean_grid[param]['data']
                sorted_row_data = dict(sorted(row_data.items()))
                middle_idx = len(sorted_row_data) // 2
                new_row_data = {}
                for idx, (key, value) in enumerate(sorted_row_data.items()):
                    if idx == middle_idx:
                        new_row_data['original_data'] = network_metrics_data[original_key]
                    new_row_data[key] = value
                clean_grid[param]['data'] = new_row_data
                
                # Plot each cell in the row
                for col_idx, (key, data) in enumerate(clean_grid[param]['data'].items()):
                    try:
                        metric_value = data['data']
                        for path_part in metric_path:
                            metric_value = metric_value[path_part]
                        color = cmap(norm(metric_value))
                        axs[row_idx, col_idx].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
                        axs[row_idx, col_idx].text(0.5, 0.5, f'{metric_value:.2f}', ha='center', va='center', fontsize=12)
                        axs[row_idx, col_idx].axis('off')
                        permuted_param = param
                        permuted_value = data['data']['simConfig'][param]
                        try:
                            permuted_value = round(permuted_value, 3)
                        except:
                            pass
                        axs[row_idx, col_idx].set_title(f'@{permuted_value}', fontsize=14)
                    except Exception as e:
                        print(f"Error loading plot for key {key}: {e}")
                #print(f"Plotted {param} in row {row_idx}")
            
            plt.tight_layout()
            plt.subplots_adjust(left=0.15, right=0.90, top=0.925)
            for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                pos = axs[row_idx, 0].get_position()
                x = pos.x0 - 0.025
                fig.text(x, pos.y0 + pos.height / 2, param, va='center', ha='right', fontsize=14, rotation=0)
            
            # Add colorbar
            # NOTE: sm generated with norm based on original_metric = nan will result in stack overrflow when trying to generate the colorbar - to deal with this,
            # to deal with this, norm has a special case above for when original_metric is nan. norm will be set to a norm that is not centered on original simulaiton value.
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
            fig.colorbar(sm, cax=cbar_ax)
            
            # Add title and save
            fig.suptitle(f'Heatmap: {metric_name}', fontsize=16)
            output_path = os.path.join(output_dir, f'_heatmap_{metric_name}.png')
            plt.savefig(output_path, dpi=100)
            print(f'Saved heatmap to {output_path}')
            return output_path
        
        # # use this while debugging to select keys for metrics of interest
        # def walk_through_keys(data, indent=0, max_depth=3, skip_keys=None):
        #     """
        #     Recursively walk through the keys of a dictionary and print them with indentation for nested levels.
            
        #     Args:
        #         data (dict): The dictionary to traverse.
        #         indent (int): The current level of indentation.
        #     """            
        #     if indent > max_depth:
        #         return
            
        #     if not isinstance(data, dict):
        #         return  # Ensure the input is a dictionary

        #     for key, value in data.items():
        #         if skip_keys is not None:
        #             if any(skip_key in key for skip_key in skip_keys):
        #                 continue
                
        #         print("  " * indent + str(key))
        #         if isinstance(value, dict):  # If the value is a dictionary, recurse
        #             walk_through_keys(value, indent + 1, max_depth=max_depth, skip_keys=skip_keys)
                    
        #     # output should look something like:
        #     '''
        #     network_metrics
        #         source
        #         timeVector
        #         simulated_data
        #             soma_voltage
        #             E_Gids
        #             I_Gids
        #             MeanFireRate_E
        #             CoVFireRate_E
        #             MeanFireRate_I
        #             CoVFireRate_I
        #             MeanISI_E
        #             MeanISI_I
        #             CoV_ISI_E
        #             CoV_ISI_I
        #         spiking_data
        #             spike_times
        #             spiking_summary_data
        #             MeanFireRate
        #             CoVFireRate
        #             MeanISI
        #             CoV_ISI
        #         bursting_data
        #             bursting_summary_data
        #             MeanWithinBurstISI
        #             CoVWithinBurstISI
        #             MeanOutsideBurstISI
        #             CoVOutsideBurstISI
        #             MeanNetworkISI
        #             CoVNetworkISI
        #             NumUnits
        #             Number_Bursts
        #             mean_IBI
        #             cov_IBI
        #             mean_Burst_Rate
        #             mean_Burst_Peak
        #             cov_Burst_Peak
        #             fano_factor
        #             baseline
        #             ax
        #         mega_bursting_data
        #             bursting_summary_data
        #             MeanWithinBurstISI
        #             CoVWithinBurstISI
        #             MeanOutsideBurstISI
        #             CoVOutsideBurstISI
        #             MeanNetworkISI
        #             CoVNetworkISI
        #             NumUnits
        #             Number_Bursts
        #             mean_IBI
        #             cov_IBI
        #             mean_Burst_Rate
        #             mean_Burst_Peak
        #             cov_Burst_Peak
        #             fano_factor
        #             baseline
        #             ax
        #     '''
        # for key in network_metrics_data.keys():
        #     # Example usage
        #     walk_through_keys(network_metrics_data[key]['data'], max_depth=3, skip_keys=['_by_unit'])
        #     break
        
        # Metric paths of interest #TODO - I guess I could automate this by looking for any metric that resolves as a single value or something like that
        metric_paths = [
            'network_metrics.simulated_data.MeanFireRate_E',
            'network_metrics.simulated_data.MeanFireRate_I',
            'network_metrics.simulated_data.CoVFireRate_E',
            'network_metrics.simulated_data.CoVFireRate_I',
            'network_metrics.simulated_data.MeanISI_E',
            'network_metrics.simulated_data.MeanISI_I',
            'network_metrics.simulated_data.CoV_ISI_E',
            'network_metrics.simulated_data.CoV_ISI_I',
            
            'network_metrics.spiking_data.spiking_summary_data.MeanFireRate',
            'network_metrics.spiking_data.spiking_summary_data.CoVFireRate',
            'network_metrics.spiking_data.spiking_summary_data.MeanISI',
            'network_metrics.spiking_data.spiking_summary_data.CoV_ISI',
            
            'network_metrics.bursting_data.bursting_summary_data.MeanWithinBurstISI',
            'network_metrics.bursting_data.bursting_summary_data.CoVWithinBurstISI',
            'network_metrics.bursting_data.bursting_summary_data.MeanOutsideBurstISI',
            'network_metrics.bursting_data.bursting_summary_data.CoVOutsideBurstISI',
            'network_metrics.bursting_data.bursting_summary_data.MeanNetworkISI',
            'network_metrics.bursting_data.bursting_summary_data.CoVNetworkISI',
            'network_metrics.bursting_data.bursting_summary_data.mean_IBI',
            'network_metrics.bursting_data.bursting_summary_data.cov_IBI',
            'network_metrics.bursting_data.bursting_summary_data.mean_Burst_Rate',
            'network_metrics.bursting_data.bursting_summary_data.mean_Burst_Peak',
            'network_metrics.bursting_data.bursting_summary_data.fano_factor',
            'network_metrics.bursting_data.bursting_summary_data.baseline',
            
            'network_metrics.mega_bursting_data.bursting_summary_data.MeanWithinBurstISI',
            'network_metrics.mega_bursting_data.bursting_summary_data.CoVWithinBurstISI',
            'network_metrics.mega_bursting_data.bursting_summary_data.MeanOutsideBurstISI',
            'network_metrics.mega_bursting_data.bursting_summary_data.CoVOutsideBurstISI',
            'network_metrics.mega_bursting_data.bursting_summary_data.MeanNetworkISI',
            'network_metrics.mega_bursting_data.bursting_summary_data.CoVNetworkISI',
            'network_metrics.mega_bursting_data.bursting_summary_data.mean_IBI',
            'network_metrics.mega_bursting_data.bursting_summary_data.cov_IBI',
            'network_metrics.mega_bursting_data.bursting_summary_data.mean_Burst_Rate',
            'network_metrics.mega_bursting_data.bursting_summary_data.mean_Burst_Peak',
            'network_metrics.mega_bursting_data.bursting_summary_data.fano_factor',
            'network_metrics.mega_bursting_data.bursting_summary_data.baseline',            
        ] 
        
        # testing
        #metric_path = ['network_metrics', 'bursting_data', 'bursting_summary_data', 'mean_Burst_Rate']
        #output_dir = './output'
        #metric_name = 'mean_Burst_Rate'
        for metric_path in metric_paths:
            try:
                metric_path_parts = metric_path.split('.')
                if any('bursting' in part for part in metric_path_parts):
                    metric_name = f'{metric_path_parts[-3]}_{metric_path_parts[-1]}'
                else:
                    metric_name = f'{metric_path_parts[-2]}_{metric_path_parts[-1]}'
                plot_metric_heatmap(output_dir, metric_path_parts, metric_name, network_metrics_data, clean_grid, levels) #TODO: add reference data...
            except Exception as e:
                print(f"Error plotting heatmap for {metric_path}: {e}")
                continue


    if plot_heatmaps:
        plot_heat_maps(
            output_dir,
            input_dir, 
            num_workers=num_workers,
            levels=levels)
        
    # # combine the plots. put comp_grid on top, heat_map on bottom
    # from PIL import Image
    # comp_grid = Image.open(comp_grid_path)
    # heat_map = Image.open(heat_map_path)
    # combined = Image.new('RGB', (comp_grid.width, comp_grid.height + heat_map.height))
    # combined.paste(comp_grid, (0, 0))
    # combined.paste(heat_map, (0, comp_grid.height))
    # combined_path = os.path.join(output_dir, '_combined_summary_grid.png')
    # combined.save(combined_path)
    # print('Combined summary grid saved to', combined_path)
    # return combined_path