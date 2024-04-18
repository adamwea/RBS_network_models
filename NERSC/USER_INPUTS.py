import sys
from batch_helper_functions import get_walltime_per_sim

## Run Name
try: USER_run_label = sys.argv[1] ### Change this to a unique name for the batch run
except: USER_run_label = 'debug_run' ### Change this to a unique name for the batch run

##Simulation Duration
try: USER_seconds = int(sys.argv[2]) ### Change this to the number of seconds for the simulation
except: USER_seconds = 5

## Available Methods
USER_method = 'evol'
#USER_method = 'grid'

## Fitness Params
#Set paramaeters for convolving raster data into network activity plot
USER_raster_convolve_params = {
    'binSize': .03*1000, 
    'gaussianSigma': .12*1000, 
    'thresholdBurst': 1.0
    }
USER_plot_fitness_bool = False

## Evol Params
USER_frac_elites = 0.1 # must be 0 < USER_frac_elites < 1. This is the fraction of elites in the population.
USER_pop_size = 2
USER_max_generations = 1
USER_time_sleep = 10 #seconds between checking for completed simulations
USER_maxiter_wait_minutes = 20 #Maximum minutes to wait before new simulation starts before killing generation

## Parallelization
options = ['local', 'NERSC_evol']
# 0 - local
# 1 - NERSC
option = options[1]
if option == 'local':
    USER_nodes = 1 #This should be set to the number of nodes available
    USER_runCfg_type = 'mpi_bulletin'
    USER_cores_per_node_per_sim = 4 #This should be set to the number of cores desired for each simulation
    USER_cores_per_node = USER_cores_per_node_per_sim    
    USER_walltime = None
    USER_email = None
    USER_custom = None
elif option == 'NERSC_evol':
    Perlmutter_cores_per_node = 256
    USER_nodes = 1 #This should be set to the number of nodes available    
    USER_runCfg_type = 'hpc_slurm'
    USER_cores_per_node_per_sim = int((Perlmutter_cores_per_node*USER_nodes)/USER_pop_size) #This should be set to the number of cores desired for each simulation
    assert USER_cores_per_node_per_sim <= (Perlmutter_cores_per_node*USER_nodes)/USER_pop_size, 'USER_cores_per_node_per_sim must be less than or equal to Perlmutter_cores_per_node'    
    USER_cores_per_node = USER_cores_per_node_per_sim
    USER_walltime_per_gen = '01:00:00' # set this value to the maxiumum walltime allowed to charge
    USER_walltime_per_sim = get_walltime_per_sim(USER_walltime_per_gen, USER_pop_size, USER_nodes)
    USER_walltime = USER_walltime_per_sim    
    USER_email = 'amwe@ucdavis.edu'
    USER_custom = ''    
else: 
    print('Invalid Parallelization Option')
    sys.exit()

## Overwrite and Continue
USER_overwrite_run = False
USER_continue_run = False

