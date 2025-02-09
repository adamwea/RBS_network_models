''' 

batch.py - Batch run script for CDKL5_DIV21

'''
import os
from netpyne import specs
from netpyne.batch import Batch
# from ....utils.batch_helper import rangify_params, get_seed_cfgs, get_num_nodes, get_cores_per_node, get_tasks_per_node
from RBS_network_models.utils.batch_helper import rangify_params, get_seed_cfgs, get_num_nodes, get_cores_per_node, get_tasks_per_node
from RBS_network_models.utils.cfg_helper import import_module_from_path

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
        cfgFile='/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/CDKL5/DIV21/src/cfg.py',
        netParamsFile='/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/CDKL5/DIV21/src/netParams.py',
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
    #max_wait = 1 # minutes
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
            'reference_data_path': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/network_metrics/CDKL5-E6D_T2_C1_05212024_240611_M08029_Network_000091_network_metrics_well001.npy',
            'plot_sim': False,
            #'plot_sim': True,
            },
        #'pop_size': 8,
        #'pop_size': 128,
        'pop_size': 256,
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
        'script': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/CDKL5/DIV21/src/init.py',
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
        #'coresPerNode': 4,
        'coresPerNode': 8,
        'reservation': None,
        #'skip': False, #if rerunning, skip if output files already exist
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