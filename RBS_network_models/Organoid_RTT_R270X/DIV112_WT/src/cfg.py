from netpyne import specs
from RBS_network_models.utils.cfg_helper import import_module_from_path
import os

# version control
#version = 0.0 # prior to 28Dec2024
version = 1.0 # major updates on 28Dec2024
version = 2.0 # # aw 2025-02-04 13:15:09

if version == 2.0:
    import random
    #import DIV21.src.fitness_targets as fitness_targets
    # import RBS_network_models.CDKL5.DIV21.src.fitness_targets as fitness_targets
    # num_excite = fitness_targets.fitnessFuncArgs['features']['num_excite']
    # num_inhib = fitness_targets.fitnessFuncArgs['features']['num_inhib']
    
    # iterate through feature data files (.py) in the features directory - get the latest one
    # import using import_module_from_path
    feature_data_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/features'
    feature_data_path = os.path.abspath(feature_data_path)
    feature_data_files = [f for f in os.listdir(feature_data_path) if f.endswith('.py')]
    feature_data_files = sorted(feature_data_files, key=lambda x: os.path.getmtime(os.path.join(feature_data_path, x)))
    feature_data_file = feature_data_files[-1]
    feature_data_file = os.path.join(feature_data_path, feature_data_file)
    feature_data_file = os.path.abspath(feature_data_file)
    fitness_targets = import_module_from_path(feature_data_file)
    
    # Initialize simulation configuration
    cfg = specs.SimConfig()
    #cfg.verbose = True # Show detailed messages
    
    #add data from fitness_targets to cfg -> useful in preparing netParams
    cfg.locations_known = True
    cfg.unit_locations = fitness_targets.fitnessFuncArgs['targets']['unit_locations']
    cfg.inhib_units = fitness_targets.fitnessFuncArgs['targets']['inhib_units']
    cfg.excit_units = fitness_targets.fitnessFuncArgs['targets']['excit_units']    
    num_excite = fitness_targets.fitnessFuncArgs['features']['num_excite']
    num_inhib = fitness_targets.fitnessFuncArgs['features']['num_inhib']    

    # Import evolutionary parameters
    def import_evol_params():
        try:
            from __main__ import params
        except:
            import warnings
            warnings.simplefilter('always')
            
            print('Could not import params from __main__. Attempting to import from path...')
            # warning, importing params in this way will import the evolutionary parameter space,
            # importantly, it won't be importing the params selected by evol config.
            # this is normal if you are running a single simulation, or otherwise testing the cfg script
            # but if you are running a batch simulation, you should be importing the params from __main__
            
            warning_message = ('\n'
                                'Importing params from src. '
                                'This is normal if you are running a single simulation, '
                                'or otherwise testing the cfg script. If you are running '
                                'a batch simulation, you should be '
                                'importing the params from __main__')
            warnings.warn(warning_message)
            
            # assert path is not None, 'Path to evolutionary parameter space not provided.'
            # assert cfg is not None, 'cfg object not provided.'
            
            # params = import_module_from_path(path)
            #params = params.params
            #from DIV21.src.evol_params import params
            from RBS_network_models.CDKL5.DIV21.src.evol_params import params
            
            # cycle through params, if any are ranges of values, randomly select one between the range
            print('Randomizing parameters within specified ranges...')
            for key, value in params.items():
                #check if list with 2 values
                if isinstance(value, list) and len(value) == 2:
                    #check if range
                    if value[0] < value[1]:
                        params[key] = random.uniform(value[0], value[1])
                    
            print('Adding params to cfg object...')
            for param in params:
                setattr(cfg, param, params[param]) 
            
            print('Evolutionary parameters imported successfully.')
    import_evol_params()
    
    cfg.duration_seconds = 1  # Duration of the simulation, in seconds
    #cfg.duration_seconds = 15  # Duration of the simulation, in seconds
    cfg.duration_seconds = 10  # Duration of the simulation, in seconds
    
    cfg.duration = cfg.duration_seconds * 1e3  # Duration of the simulation, in ms
    cfg.cache_efficient = True  # Use CVode cache_efficient option to optimize load on many cores
    cfg.dt = 0.025  # Internal integration timestep to use
    cfg.recordStep = 0.1  # Step size in ms to save data (e.g., V traces, LFP, etc)
    cfg.saveDataInclude = [
        'simData', 
        'simConfig', 
        'netParams', 
        'netCells', 
        'netPops'
        ]  # Data to save
    cfg.saveJson = False  # Save data in JSON format
    cfg.printPopAvgRates = [100, cfg.duration]  # Print population average rates
    cfg.savePickle = True  # Save params, network and sim output to pickle file

    # Record traces
    cfg.recordTraces['soma_voltage'] = {"sec": "soma", "loc": 0.5, "var": "v"}

    # Select cells for recording
    E_cells = random.sample(range(num_excite), min(2, num_excite))
    I_cells = random.sample(range(num_inhib), min(2, num_inhib))
    cfg.num_excite = num_excite
    cfg.num_inhib = num_inhib
    cfg.recordCells = [('E', E_cells), ('I', I_cells)]

    #testing new params
    #cfg.coreneuron = True
    #cfg.dump_coreneuron_model = True
    cfg.cache_efficient = True
    cfg.cvode_active = True
    cfg.use_fast_imem = True
    cfg.allowSelfConns = True
    cfg.oneSynPerNetcon = False

    #new new params
    cfg.validateNetParams = True
    #cfg.validateDataSaveOptions = True
    cfg.verbose = False
    #cfg.verbose = True

    # success message
    print('cfg.py script completed successfully.') 
elif version == 1.0:
    import random
    #import DIV21.src.fitness_targets as fitness_targets
    import RBS_network_models.CDKL5.DIV21.src.fitness_targets as fitness_targets
    num_excite = fitness_targets.fitnessFuncArgs['features']['num_excite']
    num_inhib = fitness_targets.fitnessFuncArgs['features']['num_inhib']
    
    # Initialize simulation configuration
    cfg = specs.SimConfig()
    cfg.verbose = True # Show detailed messages

    # Import evolutionary parameters
    def import_evol_params():
        try:
            from __main__ import params
        except:
            import warnings
            warnings.simplefilter('always')
            
            print('Could not import params from __main__. Attempting to import from path...')
            # warning, importing params in this way will import the evolutionary parameter space,
            # importantly, it won't be importing the params selected by evol config.
            # this is normal if you are running a single simulation, or otherwise testing the cfg script
            # but if you are running a batch simulation, you should be importing the params from __main__
            
            warning_message = ('\n'
                                'Importing params from src. '
                                'This is normal if you are running a single simulation, '
                                'or otherwise testing the cfg script. If you are running '
                                'a batch simulation, you should be '
                                'importing the params from __main__')
            warnings.warn(warning_message)
            
            # assert path is not None, 'Path to evolutionary parameter space not provided.'
            # assert cfg is not None, 'cfg object not provided.'
            
            # params = import_module_from_path(path)
            #params = params.params
            #from DIV21.src.evol_params import params
            from RBS_network_models.CDKL5.DIV21.src.evol_params import params
            
            # cycle through params, if any are ranges of values, randomly select one between the range
            print('Randomizing parameters within specified ranges...')
            for key, value in params.items():
                #check if list with 2 values
                if isinstance(value, list) and len(value) == 2:
                    #check if range
                    if value[0] < value[1]:
                        params[key] = random.uniform(value[0], value[1])
                    
            print('Adding params to cfg object...')
            for param in params:
                setattr(cfg, param, params[param]) 
            
            print('Evolutionary parameters imported successfully.')
    import_evol_params()
    
    cfg.duration_seconds = 1  # Duration of the simulation, in seconds
    cfg.duration = cfg.duration_seconds * 1e3  # Duration of the simulation, in ms
    cfg.cache_efficient = True  # Use CVode cache_efficient option to optimize load on many cores
    cfg.dt = 0.025  # Internal integration timestep to use
    cfg.recordStep = 0.1  # Step size in ms to save data (e.g., V traces, LFP, etc)
    cfg.saveDataInclude = [
        'simData', 
        'simConfig', 
        'netParams', 
        'netCells', 
        'netPops'
        ]  # Data to save
    cfg.saveJson = False  # Save data in JSON format
    cfg.printPopAvgRates = [100, cfg.duration]  # Print population average rates
    cfg.savePickle = True  # Save params, network and sim output to pickle file

    # Record traces
    cfg.recordTraces['soma_voltage'] = {"sec": "soma", "loc": 0.5, "var": "v"}

    # Select cells for recording
    E_cells = random.sample(range(num_excite), min(2, num_excite))
    I_cells = random.sample(range(num_inhib), min(2, num_inhib))
    cfg.num_excite = num_excite
    cfg.num_inhib = num_inhib
    cfg.recordCells = [('E', E_cells), ('I', I_cells)]

    #testing new params
    #cfg.coreneuron = True
    #cfg.dump_coreneuron_model = True
    cfg.cache_efficient = True
    cfg.cvode_active = True
    cfg.use_fast_imem = True
    cfg.allowSelfConns = True
    cfg.oneSynPerNetcon = False

    #new new params
    cfg.validateNetParams = True
    #cfg.validateDataSaveOptions = True
    cfg.verbose = False
    #cfg.verbose = True

    # success message
    print('cfg.py script completed successfully.')    
elif version == 0.0:
    from cfg_helper import *
    #from netpyne.batchtools import specs

    # Path to runtime kwargs
    cfg_runtime_kwargs_path = './cfg_kwargs.json'
    cfg_runtime_kwargs_path = os.path.abspath(cfg_runtime_kwargs_path)

    # Initialize simulation configuration
    cfg = specs.SimConfig()
    cfg.verbose = True # Show detailed messages

    # Import runtime kwargs
    assert os.path.exists(cfg_runtime_kwargs_path), f'{cfg_runtime_kwargs_path} not found'
    with open(cfg_runtime_kwargs_path, 'r') as f:
        cfg_kwargs = json.load(f)
    cfg.duration_seconds = cfg_kwargs['duration_seconds']
    cfg.target_script_path = cfg_kwargs['target_script_path']

    # Import evolutionary parameters
    cfg.params_script_path = cfg_kwargs['param_script_path']
    # cfg.params = handle_params_import(cfg.params_script_path, cfg)
    params = handle_params_import(cfg.params_script_path, cfg)
    #including the above line makes the json non-serializable, just leave it out for now.

    cfg.duration = cfg.duration_seconds * 1e3  # Duration of the simulation, in ms
    cfg.cache_efficient = True  # Use CVode cache_efficient option to optimize load on many cores
    cfg.dt = 0.025  # Internal integration timestep to use
    cfg.recordStep = 0.1  # Step size in ms to save data (e.g., V traces, LFP, etc)
    cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'netCells', 'netPops']  # Data to save
    cfg.saveJson = True  # Save data in JSON format
    cfg.printPopAvgRates = [100, cfg.duration]  # Print population average rates
    cfg.savePickle = True  # Save params, network and sim output to pickle file

    # Record traces
    cfg.recordTraces['soma_voltage'] = {"sec": "soma", "loc": 0.5, "var": "v"}
    print('Recording soma voltage traces.')

    # Select cells for recording
    num_excite, num_inhib = get_cell_numbers_from_fitness_target_script(target_script_path=cfg.target_script_path)
    #numExcitatory, numInhibitory = USER_num_excite, USER_num_inhib
    E_cells = random.sample(range(num_excite), min(2, num_excite))
    I_cells = random.sample(range(num_inhib), min(2, num_inhib))
    cfg.num_excite = num_excite
    cfg.num_inhib = num_inhib
    cfg.recordCells = [('E', E_cells), ('I', I_cells)]
    print(f'Cells selected for recording: {cfg.recordCells}')

    #testing new params
    #cfg.coreneuron = True
    #cfg.dump_coreneuron_model = True
    cfg.cache_efficient = True
    cfg.cvode_active = True
    cfg.use_fast_imem = True
    cfg.allowSelfConns = True
    cfg.oneSynPerNetcon = False

    #new new params
    cfg.validateNetParams = True
    #cfg.validateDataSaveOptions = True
    cfg.verbose = True

    # success message
    print('cfg.py script completed successfully.')    