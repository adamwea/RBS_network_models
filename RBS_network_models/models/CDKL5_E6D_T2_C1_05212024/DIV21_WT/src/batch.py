''' 

batch.py - Batch run script for Organoid_RTT_R270X_DIV112_WT

'''
# imports ==========================================================================================
import os
from netpyne import specs
from netpyne.batch import Batch
from RBS_network_models.utils.batch_helper import rangify_params, get_num_nodes, get_cores_per_node, get_tasks_per_node
from RBS_network_models.utils.cfg_helper import import_module_from_path
from RBS_network_models.fitnessFunc import fitnessFunc_v3 as fitnessFunc
from RBS_network_models.utils.helper import indent_decrease, indent_increase
import numpy as np
from pathlib import Path
from netpyne import sim
from copy import deepcopy

# functions ==========================================================================================
def batchEvol_v2(**kwargs):
    '''     
    Evolutionary algorithm optimization of a network using NetPyNE
    To run locally: mpiexec -np [num_cores] nrniv -mpi batchRun.py
    To run in interactive mode:
        salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=2 --tasks-per-node=32 --cpus-per-task=4 --image=adammwea/netsims_docker:v1
    '''
    # subfunctions ============================================================================================
    
    def init_params(params):
        params = kwargs.get('parameter_space', None)
        if params is None: raise ValueError("parameter_space must be provided in kwargs")
        params = rangify_params(params)
        return params
    
    def load_reference_data_paths(kwargs):
        reference_data_list = []
        reference_data_paths = kwargs.get('reference_data_paths', None)
        #reference_data_paths = [os.path.abspath(path) for path in reference_data_paths] # convert reference paths to abs paths if they are not already
        reference_data_paths = [Path(path).expanduser().resolve() for path in reference_data_paths]
        if reference_data_paths is None: raise ValueError("reference_data_paths must be provided in kwargs")
        for path in reference_data_paths:
            if not os.path.exists(path): raise ValueError(f"Reference data path does not exist: {path}")
            else: 
                # load numpy data w/ pickle
                print(f'loading reference data from {reference_data_paths}...')
                path = path.resolve().__str__()
                
                # instead lets append the path to the list
                reference_data_list.append(path)
                print('reference data loaded.')
        kwargs['reference_data_list'] = reference_data_list
        
        # load reference data into global
        reference_data_path = reference_data_list[0] # HACK: for now, this only works for one reference data set - i feel we may want to change this in the future
        kwargs['reference_data_path'] = reference_data_path
        
        return kwargs
    
    def init_fitnessFunc_args(**kwargs):
        # setting up fitnessFuncArgs
        print('setting up fitness function arguments...')
        conv_params = kwargs.get('conv_params', None)
        mega_params = kwargs.get('mega_params', None)
        if conv_params is None: raise ValueError("conv_params must be provided in kwargs")
        if mega_params is None: raise ValueError("mega_params must be provided in kwargs")
        fitnessFuncArgs = {
            #'reference_data': reference_data,
            'conv_params': conv_params,
            'mega_params': mega_params,
            'plot_sim': kwargs.get('plot_sim', False),
            'reference_data_path': kwargs.get('reference_data_path', None),
            'batching': True,
            
            # compute_network_metrics args
            'try_load': True, # try to load .npy metrics file if it exists
            'run_parallel': True,
            #'max_workers': 4, #NOTE: this should match the number of cores per node, would probably risk oversubscribing the node if set too high
            #'max_workers': 8, #NOTE: this should match the number of cores per node, would probably risk oversubscribing the node if set too high
            
            # nvm these run in series for now, just use 128 
            'max_workers': 128, 
            'burst_sequencing': True,       
        }
        return fitnessFuncArgs  
    
    def init_batch_attributes(kwargs):
        # simulation max iteration options -- max iterations before stopping generation
        time_sleep = 5 # seconds
        max_wait = 15 # minutes
        maxiter_wait = max_wait * 60 / time_sleep # convert to number of iterations
        #b.batchLabel = 'evol' #NOTE: if left unset, batchLabel will be set to datetime at runtime
        
        # pop size options
        pop_size = 512
        #pop_size = 256        
        #pop_size = 128
        
        # num elites options
        #num_elites = 50
        #num_elites = 75
        num_elites = 128
        
        kwargs.update({
            'time_sleep': time_sleep,
            #'max_wait': max_wait,
            'maxiter_wait': maxiter_wait,
            'pop_size': pop_size,
            'num_elites': num_elites,
        })
        return kwargs
    
    def get_seed_cfgs(params, **kwargs):
        seeds_iterable = []
        seeds = kwargs.get('seeds', None)
        evol_params = params.copy()
        for i, seed in enumerate(seeds):
            #load simcfg from seed
            seed_iterable = []
            sim.loadSimCfg(seed, setLoaded=True)
            simcfg = deepcopy(sim.cfg.todict())
            for key in evol_params:
                if key in simcfg:
                    seed_iterable.append(simcfg[key])
                else:
                    upper_bound = evol_params[key]['values'][1]
                    lower_bound = evol_params[key]['values'][0]
                    random_value = np.random.uniform(lower_bound, upper_bound)
                    seed_iterable.append(random_value)
            seeds_iterable.append(seed_iterable)
        if len(seeds_iterable) == 0: seeds_iterable = None
        return seeds_iterable    
        
    def init_runCfg(b, **kwargs):
        # set run configuration 
        run_Cfg_script_path = kwargs.get('runCfg_script_path', None)
        if run_Cfg_script_path is None: raise ValueError("runCfg_script_path must be provided in kwargs")       
        b.runCfg = {
            'type': 
                'mpi_direct', 
                #'hpc_slurm', #TODO: not sure if this is really an option
                #'mpi_bulletin', #TODO: not sure if this is really an option
            'script': run_Cfg_script_path,
            'mpiCommand': 
                #'',
                
                #'mpirun',
                
                #'shifter --image=adammwea/netsims_docker:v1'
                'srun -N 1',
                # # bind to socket
                # ' --cpu-bind=verbose,cores'
                # ' --hint=multithread' # enable multithreading on each core
                # ' --cores-per-task=4' # set number of cores per task
                #,
            'nrnCommand': 'nrniv',
            'nodes': 1,                             
                # NOTE: Importantly, these are the number of nodes to use for each simulation, I think. 
                # So if I want to put 4 simulations on each node, 2 per socket.
                # nodes should be set to 1, and tasks_per_node should be set to cores_per_node / 4                                        
            #'coresPerNode': 16,
            #'coresPerNode': 4, #i.e., 4 mpi tasks per sim, @256 cands per gen, @1 cpu per task = 1024 cores. 4 nodes, each with 256 logical cores, allows 1024 cores to be used.
            
            # aw 2025-04-22 13:17:37 4 is too slow. going to try more. allow srun commands to queue
            #'coresPerNode': 16, #i.e., 4 mpi tasks per sim, @256 cands per gen, @1 cpu per task = 1024 cores. 4 nodes, each with 256 logical cores, allows 1024 cores to be used.
            
            # # aw 2025-04-23 03:37:46 lets try maximizing for 1 sim / socket (i.e. 64 tasks per node)
            'coresPerNode': 64,
            
            'reservation': None,
            #'skip': False, #if rerunning, skip if output files already exist
            'skip': True, #if rerunning, skip if output files already exist
            }
        
        # for key in kwargs, replace b.runCfg[key] = kwargs[key] if matching key exists in b.runCfg
        for key in kwargs:
            if key in b.runCfg:
                b.runCfg[key] = kwargs[key]
                print(f'Overriding b.runCfg.{key} = {kwargs[key]}')
                
        # return batch object
        return b
    
    def initialize_batch(params, kwargs):
        indent_increase()
        
        # init file paths, modify as needed
        cfgFile_path = str(Path('~/workspace/repos/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_WT/src/cfg.py').expanduser().resolve())
        netParams_path = str(Path('~/workspace/repos/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_WT/src/netParams.py').expanduser().resolve())
        
        # init batch object
        print('initializing batch object...')
        b = Batch(cfgFile=cfgFile_path, netParamsFile=netParams_path, params=params,)
        b.method = 'evol' # set method to evolutionary algorithm

        #init fitness function args
        print('initializing fitness function arguments...')
        fitnessFuncArgs = init_fitnessFunc_args(**kwargs)      
        
        # apply kwargs to batch object
        print('setting batch object attributes...')
        kwargs=init_batch_attributes(kwargs)
        
        #convert seeds into interable list of params
        print('setting seeds configs...')
        seeds_iterable = get_seed_cfgs(params, **kwargs)      
        
        # evolutionary algorithm configuration
        print('setting evolutionary algorithm configuration...')
        b.evolCfg = {
            'evolAlgorithm': 'custom',
            'fitnessFunc': fitnessFunc,
            'fitnessFuncArgs': fitnessFuncArgs,
            'pop_size': kwargs.get('pop_size', 16),
            'num_elites': kwargs.get('num_elites', 4),
            'mutation_rate': 0.5,
            'crossover': 0.5,
            'maximize': False,
            'max_generations': 1000,
            'time_sleep': kwargs.get('time_sleep', 5),
            'maxiter_wait': kwargs.get('maxiter_wait', 10),
            'defaultFitness': 1000,
            'seeds': seeds_iterable, #requires params to put candidates in correct order,
            #'startGeneration': 8, #NOTE: dont used this. 
        }
        
        # init runcfg
        print('setting run configuration...')
        b = init_runCfg(b, **kwargs)
                
        # set save folder
        batchFolder = kwargs.get('batchFolder', None)
        if batchFolder is None: raise ValueError("batchFolder must be provided in kwargs")
        b.saveFolder = os.path.join(batchFolder, b.batchLabel)
        os.makedirs(b.saveFolder, exist_ok=True)
        print(f'b.saveFolder = {b.saveFolder}')
        
        # return batch object
        print('batch object initialized.')
        indent_decrease()
        return b
        
    # globals ==========================================================================================
    global reference_data
    
    # main ==========================================================================================
    indent_increase()
    
    #parameters space to explore
    print('rangifying parameters...')
    params = kwargs.get('parameter_space', None)
    params = init_params(params)
    
    # load reference data paths
    print('loading reference data paths...')
    kwargs = load_reference_data_paths(kwargs)
    
    ## create batch object
    print('initializing batch object...')
    b = initialize_batch(params, kwargs)

    ## run batch
    print('running batch...')
    b.run()
    
    # end indentation
    indent_decrease()

def batchEvol(feature_path, **kwargs):
    ''' 
    
    Evolutionary algorithm optimization of a network using NetPyNE
    To run locally: mpiexec -np [num_cores] nrniv -mpi batchRun.py
    To run in interactive mode:
        salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=2 --tasks-per-node=32 --cpus-per-task=4 --image=adammwea/netsims_docker:v1
    '''
    #parameters space to explore
    ## network
    # from .evol_params import params
    # from .fitness_targets import fitnessFuncArgs
    # from ....fitnessFunc import fitnessFunc
    
    ## format params so all values are lists of length 2, min and max values
    from RBS_network_models.CDKL5.DIV21.src.evol_params import params
    from RBS_network_models.fitnessFunc import fitnessFunc
    feature_module = import_module_from_path(feature_path)
    fitnessFuncArgs = feature_module.fitnessFuncArgs
    params = rangify_params(params)
    
    ## create batch object
    b = Batch(
        cfgFile='/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/cfg.py',
        netParamsFile='/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/netParams.py',
        #cfg=None,
        #netParams=None,
        params=params,
        #groupedParams=None,
        #initCfg=None,
        #seed=None,
    )
    
    ## set batch object attributes
    time_sleep = 5 # seconds
    #max_wait = 30 # minutes
    max_wait = 10 # minutes
    maxiter_wait = max_wait * 60 / time_sleep # convert to number of iterations
    #b.batchLabel = 'evol' #NOTE: if left unset, batchLabel will be set to datetime at runtime
    from RBS_network_models.CDKL5.DIV21.src.conv_params import conv_params
    from RBS_network_models.CDKL5.DIV21.src.conv_params import mega_params
    b.evolCfg = {
        'evolAlgorithm': 'custom',
        'fitnessFunc': fitnessFunc,
        'fitnessFuncArgs': {
            **fitnessFuncArgs,
            'conv_params': conv_params,
            'mega_params': mega_params, 
            'reference_data_path': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/network_metrics/Organoid_RTT_R270X_pA_pD_B1_d91_250107_M07297_Network_000028_network_metrics_well005.npy',
            'plot_sim': False,
            #'plot_sim': True,
            },
        #'pop_size': 8,
        #'pop_size': 128,
        '#pop_size': 256,
        #'pop_size': 196,
        'num_elites': 50, 
        #'num_elites': 1,
        'mutation_rate': 0.5,
        'crossover': 0.5,
        'maximize': False,
        'max_generations': 1000,
        'time_sleep': time_sleep,
        'maxiter_wait': maxiter_wait,
        'defaultFitness': 1000,
        #pass list of paths in seed_dir to seed the population
        #seed_dir = kwargs['seed_dir']
        #'seeds': get_seed_cfgs(kwargs['seed_dir'], params), #requires params to put candidates in correct order
    }
    #b.initCfg = {}
    b.method = 'evol'
    #b.mpiCommandDefault = 'mpiexec'
    #b.optimCfg = {}
    
    ## set run configuration
    # tasks_per_node = get_cores_per_node() // 4 #NOTE: I think this is the number of tasks to run on each node
    nodes_per_core = get_cores_per_node()
    #tasks_per_sim = get_tasks_per_node() // 4
    mpi_tasks_per_node = 64
    mpi_tasks_per_sim = mpi_tasks_per_node // 4
    
    b.runCfg = {
        'type': 
            'mpi_direct', 
            #'hpc_slurm', #TODO: not sure if this is really an option
            #'mpi_bulletin', #TODO: not sure if this is really an option
        #'script': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/CDKL5/DIV21/src/init.py',
        'script': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/init.py',
        'mpiCommand': '',
            
            # 'mpirun',
            
            # 'srun'
            # # bind to socket
            # ' --cpu-bind=verbose,cores'
            # ' --hint=multithread' # enable multithreading on each core
            # ' --cores-per-task=4' # set number of cores per task
            # ,
            
        'nrnCommand': 'nrniv',
        #'nodes': get_num_nodes(),
        'nodes': 1,                             # NOTE: Importantly, these are the number of nodes to use for each simulation, I think. 
                                                # So if I want to put 4 simulations on each node, 2 per socket.
                                                # nodes should be set to 1, and tasks_per_node should be set to cores_per_node / 4                                        
        #'coresPerNode': mpi_tasks_per_sim, #NOTE: I think this basically translates to mpi tasks per node
        #'coresPerNode': 1,
        'coresPerNode': 4,
        'reservation': None,
        'skip': False, #if rerunning, skip if output files already exist
        }
    
    # for key in kwargs, replace b.runCfg[key] = kwargs[key] if matching key exists in b.runCfg
    for key in kwargs:
        #print (f'kwargs: {key} = {kwargs[key]}')
        if key in b.runCfg:
        #if hasattr(b.runCfg, key):
            b.runCfg[key] = kwargs[key]
            #setattr(b.runCfg, key, kwargs[key])
            print(f'Overriding b.runCfg.{key} = {kwargs[key]}')
            #print(f'Overriding b.runCfg.{key} = {getattr(b.runCfg, key)}')
            
    b.saveFolder = f'/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/batch_runs/{b.batchLabel}'
    # b.seed = None #NOTE: I think this is for getting identical random numbers if rerunning the same batch
    
    # To debug the batch script without running the full optimization, you can uncomment the following line:
    # import sys
    # sys.exit()
    
    ## run batch
    b.run()

def batchOptuna(**kwargs):
    print('not implemented yet')
    pass