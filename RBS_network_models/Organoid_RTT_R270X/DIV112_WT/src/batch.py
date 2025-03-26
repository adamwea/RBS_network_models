''' 

batch.py - Batch run script for Organoid_RTT_R270X_DIV112_WT

'''
# imports ==========================================================================================
import os
from netpyne import specs
from netpyne.batch import Batch
from RBS_network_models.utils.batch_helper import rangify_params, get_seed_cfgs, get_num_nodes, get_cores_per_node, get_tasks_per_node
from RBS_network_models.utils.cfg_helper import import_module_from_path
from RBS_network_models.fitnessFunc import fitnessFunc_v2 as fitnessFunc
from RBS_network_models.utils.helper import indent_decrease, indent_increase
import numpy as np

# functions ==========================================================================================
def batchEvol_v2(**kwargs):
    ''' 
    
    Evolutionary algorithm optimization of a network using NetPyNE
    To run locally: mpiexec -np [num_cores] nrniv -mpi batchRun.py
    To run in interactive mode:
        salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=2 --tasks-per-node=32 --cpus-per-task=4 --image=adammwea/netsims_docker:v1
    '''
    # subfunctions ============================================================================================
    def initialize_batch(**kwargs):
        indent_increase()
        b = Batch(
            cfgFile='/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/cfg.py',
            netParamsFile='/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/netParams.py',
            params=params,
            ## other options
                #cfg=None,
                #netParams=None,
                #groupedParams=None,
                #initCfg=None,
                #seed=None,
        )
        
        # # set up reference data
        # print('setting up reference data...')
        # reference_data_list = kwargs.get('reference_data_list', None)
        # if reference_data_list is None: raise ValueError("reference_data_list must be provided in kwargs")
        # reference_data = reference_data_list[0] # HACK: for now, this only works for one reference data set - i feel we may want to change this in the future
        
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
        }        
        
        # apply kwargs to batch object
        print('setting batch object attributes...')
            
        # simulation max iteration options -- max iterations before stopping generation
        time_sleep = 5 # seconds
        #max_wait = 30 # minutes
        max_wait = 10 # minutes
        maxiter_wait = max_wait * 60 / time_sleep # convert to number of iterations
        #b.batchLabel = 'evol' #NOTE: if left unset, batchLabel will be set to datetime at runtime
        
        # pop size options
        #pop_size = 256
        #pop_size = 196
        #pop_size = 128
        pop_size = 8 # use low pop size for testing
        
        # num elites options
        #num_elites = 75
        #num_elites = 50
        num_elites = 1
        
        # evolutionary algorithm configuration
        print('setting evolutionary algorithm configuration...')
        b.evolCfg = {
            'evolAlgorithm': 'custom',
            'fitnessFunc': fitnessFunc,
            'fitnessFuncArgs': fitnessFuncArgs,
            'pop_size': pop_size,
            'num_elites': num_elites, 
            'mutation_rate': 0.5,
            'crossover': 0.5,
            'maximize': False,
            'max_generations': 1000,
            'time_sleep': time_sleep,
            'maxiter_wait': maxiter_wait,
            'defaultFitness': 1000,
        }

        # set method to evolutionary algorithm
        b.method = 'evol'
        
        # set run configuration 
        run_Cfg_script_path = kwargs.get('runCfg_script_path', None)
        if run_Cfg_script_path is None: raise ValueError("runCfg_script_path must be provided in kwargs")       
        b.runCfg = {
            'type': 
                'mpi_direct', 
                #'hpc_slurm', #TODO: not sure if this is really an option
                #'mpi_bulletin', #TODO: not sure if this is really an option
            'script': run_Cfg_script_path,
            'mpiCommand': '',
                
                # 'mpirun',
                
                # 'srun'
                # # bind to socket
                # ' --cpu-bind=verbose,cores'
                # ' --hint=multithread' # enable multithreading on each core
                # ' --cores-per-task=4' # set number of cores per task
                # ,
            'nrnCommand': 'nrniv',
            'nodes': 1,                             
                # NOTE: Importantly, these are the number of nodes to use for each simulation, I think. 
                # So if I want to put 4 simulations on each node, 2 per socket.
                # nodes should be set to 1, and tasks_per_node should be set to cores_per_node / 4                                        
            'coresPerNode': 4,
            'reservation': None,
            'skip': False, #if rerunning, skip if output files already exist
            }
        
        # for key in kwargs, replace b.runCfg[key] = kwargs[key] if matching key exists in b.runCfg
        for key in kwargs:
            if key in b.runCfg:
                b.runCfg[key] = kwargs[key]
                print(f'Overriding b.runCfg.{key} = {kwargs[key]}')
                
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
    
    
        # global variables
    
    # globals ==========================================================================================
    global reference_data
    
    # main ==========================================================================================
    indent_increase()
    
    #parameters space to explore
    print('rangifying parameters...')
    params = kwargs.get('parameter_space', None)
    if params is None: raise ValueError("parameter_space must be provided in kwargs")
    params = rangify_params(params)
    
    # load reference data paths
    print('loading reference data paths...')
    reference_data_list = []
    reference_data_paths = kwargs.get('reference_data_paths', None)
    if reference_data_paths is None: raise ValueError("reference_data_paths must be provided in kwargs")
    for path in reference_data_paths:
        if not os.path.exists(path): raise ValueError(f"Reference data path does not exist: {path}")
        else: 
            # load numpy data w/ pickle
            print(f'loading reference data from {reference_data_paths}...')
            data = np.load(path, allow_pickle=True).item()
            reference_data_list.append(data)
            print('reference data loaded.')
    kwargs['reference_data_list'] = reference_data_list
    
    # load reference data into global
    reference_data = reference_data_list[0] # HACK: for now, this only works for one reference data set - i feel we may want to change this in the future
    
    ## create batch object
    print('initializing batch object...')
    b = initialize_batch(**kwargs)

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