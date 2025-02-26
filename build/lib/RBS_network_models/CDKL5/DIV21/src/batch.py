''' 
batch.py - Batch run script for CDKL5_DIV21
'''
from netpyne.batch import Batch
from RBS_network_models.utils.batch_helper import rangify_params, get_seed_cfgs, get_num_nodes, get_cores_per_node, get_tasks_per_node
from RBS_network_models.utils.cfg_helper import import_module_from_path
# ===============================================================================================================================
def batchEvol(feature_path, modify_batch_label=None, seed_dir = None, **kwargs):
    from RBS_network_models.CDKL5.DIV21.src.evol_params import params
    from RBS_network_models.fitnessFunc import fitnessFunc
    from RBS_network_models.CDKL5.DIV21.src.conv_params import conv_params, mega_params
    ''' 
    Evolutionary algorithm optimization of a network using NetPyNE
    To run locally: mpiexec -np [num_cores] nrniv -mpi batchRun.py
    To run in interactive mode:
        salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=2 --tasks-per-node=32 --cpus-per-task=4 --image=adammwea/netsims_docker:v1
    '''
    # initialize
    # import module from path
    feature_module = import_module_from_path(feature_path)
    fitnessFuncArgs = feature_module.fitnessFuncArgs
    
    ## format params so all values are lists of length 2, min and max values
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
    b.method = 'evol'
    
    #modulate or set b.batchLabel
    #b.batchLabel = 'evol' #NOTE: if left unset, batchLabel will be set to datetime at runtime
    if modify_batch_label is not None and isinstance(modify_batch_label, str):
        #mod_label = "testing"
        mod_label = modify_batch_label
        b.batchLabel = b.batchLabel + '_' + mod_label
    
    ## set batch object attributes
    time_sleep = 5 # seconds
    max_wait = 10 # minutes
    maxiter_wait = max_wait * 60 / time_sleep # convert to number of iterations    
    
    ## set evolution configuration
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
        'num_elites': 75, 
        #'num_elites': 1,
        'mutation_rate': 0.5,
        'crossover': 0.5,
        'maximize': False,
        'max_generations': 1000,
        'time_sleep': time_sleep,
        'maxiter_wait': maxiter_wait,
        'defaultFitness': 1000,
        #'seeds': seed_list,
    }
    
    ## generate list of seed configurations
    seed_list = get_seed_cfgs(seed_dir, params, verbose=False, num_workers=16) #NOTE: use verbose to print cfgs loaded instead of progress bar.
    num_seeds_loaded = len(seed_list)
    print(f'num_seeds_loaded: {num_seeds_loaded}')
    
    if num_seeds_loaded > b.evolCfg['pop_size']:
        print(f'WARNING: pop_size ({b.evolCfg["pop_size"]}) is less than the number of seeds loaded ({num_seeds_loaded}). Reducing the number of seeds loaded to match pop_size.')
        seed_list = seed_list[:b.evolCfg['pop_size']]
        num_seeds_loaded = len(seed_list)
        print(f'seeds culled to be equal to pop_size: {num_seeds_loaded}')
    
    if num_seeds_loaded > b.evolCfg['num_elites']:
        print(f'WARNING: num_elites ({b.evolCfg["num_elites"]}) is greater than the number of seeds loaded ({num_seeds_loaded}). Reducing num_elites to match the number of seeds loaded.')
        #shorten seed_list to 75% of num_elites max
        b.evolCfg['num_elites'] = int(b.evolCfg['num_elites'] * 0.75)
        num_seeds_loaded = len(seed_list)
        print(f'seeds culled to be equal to be 75% of num_elites: {num_seeds_loaded} - allowing for novel cfgs to propagate')
    
    b.evolCfg['seeds'] = seed_list
    
    ## set batch run configuration
    b.runCfg = {
        'type': 
            'mpi_direct', 
            #'hpc_slurm', #TODO: not sure if this is really an option
            #'mpi_bulletin', #TODO: not sure if this is really an option
        'script': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/CDKL5/DIV21/src/init.py',
        'mpiCommand': '',            
        'nrnCommand': 'nrniv',
        'nodes': 1,             # NOTE: Importantly, these are the number of nodes to use for each simulation, I think. 
                                # So if I want to put 4 simulations on each node, 2 per socket.
                                # nodes should be set to 1, and tasks_per_node should be set to cores_per_node / 4                                        
        'coresPerNode': 8,
        'reservation': None,
        #'skip': True, #if rerunning, skip if output files already exist
        'skip': False, #if rerunning, skip if output files already exist
        }
    
    # for key in kwargs, replace b.runCfg[key] = kwargs[key] if matching key exists in b.runCfg
    def override_cfg(b, kwargs):        
        for key in kwargs:
            #print (f'kwargs: {key} = {kwargs[key]}')
            if key in b.runCfg:
            #if hasattr(b.runCfg, key):
                b.runCfg[key] = kwargs[key]
                #setattr(b.runCfg, key, kwargs[key])
                print(f'Overriding b.runCfg.{key} = {kwargs[key]}')
                #print(f'Overriding b.runCfg.{key} = {getattr(b.runCfg, key)}')
            if key in b.evolCfg:
                b.evolCfg[key] = kwargs[key]
                print(f'Overriding b.evolCfg.{key} = {kwargs[key]}')
                #print(f'Overriding b.evolCfg.{key} = {getattr(b.evolCfg, key)}')
        return b
    b = override_cfg(b, kwargs)
            
    b.saveFolder = f'/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/batch_runs/{b.batchLabel}'
    # b.seed = None #NOTE: I think this is for getting identical random numbers if rerunning the same batch
    
    ## run batch
    # To debug the batch script without running the full optimization, you can uncomment the following line:
    # import sys
    # sys.exit()
    b.run()

def batchOptuna(**kwargs):
    print('not implemented yet')
    pass