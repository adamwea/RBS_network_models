import sys
import os
import datetime
import re
from mpi4py import MPI
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
rank = mpi_rank
if not rank: rank = 0
rank = int(rank)

'''Functions'''
def get_USER_duration(args = sys.argv):
    args = args
    # Check if the flag -d is present
    #if __name__ == "__main__":
    if "-d" in sys.argv:
        # Find the index of the flag -d
        index = sys.argv.index("-d")        
        # Check if there is an integer value after the flag -d
        if index + 1 < len(sys.argv):
            try:
                # Get the integer value after the flag -d
                USER_seconds = int(sys.argv[index + 1])
                #print(f'USER_seconds: {USER_seconds}')
            except ValueError:
                print("Invalid value after -d flag. Please provide an integer value.")
                raise Exception
        else:
            print("No value provided after -d flag.")
            raise Exception
    else:
        print("Flag -d not found.")
        raise Exception
    return USER_seconds

'''Batch Dir Inputs'''
if '-rp' in sys.argv:
    #print('Found -rp flag')
    index = sys.argv.index('-rp')
    USER_run_path = sys.argv[index + 1]
try: assert os.path.exists(USER_run_path), f'USER_run_path does not exist: {USER_run_path}'
except: USER_run_path = None
if rank == 0: 
    print(f'USER_run_path: {USER_run_path}')
    print(f'USER_run_path will be automatically generated in batchRun.py.')

'''Simulation Inputs'''
script_path = os.path.dirname(os.path.realpath(__file__))
try: USER_seconds = get_USER_duration()
except: USER_seconds = 1
if rank == 0: print(f'USER_seconds: {USER_seconds}')
USER_method = 'evol' #'evol', 'grid', 'asd'
USER_init_script = f'{script_path}/init.py'
USER_cfgFile = f'{script_path}/cfg.py'

'''Network Inputs'''
USER_netParamsFile = f'{script_path}/netParams.py'
assert os.path.exists(USER_init_script), f'initFile does not exist: {USER_init_script}'
assert os.path.exists(USER_cfgFile), f'cfgFile does not exist: {USER_cfgFile}'
assert os.path.exists(USER_netParamsFile), f'netParamsFile does not exist: {USER_netParamsFile}'

'''Overwrite Inputs'''
USER_skip = False #Skip running the simulation if data already exist
USER_overwrite = False #Overwrite existing batch_run with same name
USER_continue = False #Continue from last completed simulation
if USER_continue: USER_skip = True #continue will re-run existing simulations if skip is False

'''Evol Params'''
USER_pop_size = 128
USER_pop_size = 4
script_path = os.path.dirname(os.path.realpath(__file__))
USER_HOF = f'{script_path}/HOF/hof.csv' #seed gen 0 with solutions in HOF.csv
#print(f'USER_HOF: {USER_HOF}') 
USER_HOF = os.path.abspath(USER_HOF)
USER_frac_elites = 0.1 # must be 0 < USER_frac_elites < 1. This is the fraction of elites in the population.
USER_num_elites = int(USER_frac_elites * USER_pop_size) if USER_frac_elites > 0 else 1
USER_max_generations = 3000
USER_time_sleep = 10 #seconds between checking for completed simulations
maxiter_wait_minutes = 4*60 #Maximum minutes to wait before starting new Generation
USER_maxiter_wait = maxiter_wait_minutes*60/USER_time_sleep
USER_mutation_rate = 0.7
USER_crossover = 0.5

'''Plotting Inputs'''
##Plotting Params
USER_plot_fitness_bool = False
USER_plot_NetworkActivity = False
USER_plotting_path = 'NERSC/plots/'
#USER_ploting_path = None #prevent plotting even if USER_plot_fitness_bool = True
USER_figsize = (10, 10)
# Network Activity Plotting Params
USER_plotting_params = {
    'fresh_plots': True,
    'saveFig': USER_plotting_path,
    'figsize': USER_figsize,
    'NetworkActivity': {
        'figsize': USER_figsize,
        #limits (will override modifiers)
        'ylim': None,
        'xlim': None,
        #range mods
        'yhigh100': 1.05, #high modifier limit for y axis
        'ylow100': 0.95, #low modifier limit for y axis
        'saveFig': USER_plotting_path,
        'title_font': 11
        }
    }

'''Fitness Inputs'''
## Fitness Params
#Set paramaeters for convolving raster data into network activity plot
USER_raster_convolve_params = {
    #'binSize': .03*1000,
    'binSize': .03*250,
    #'gaussianSigma': .12*1000, 
    'gaussianSigma': .12*250,
    'thresholdBurst': 1.0
    }
USER_raster_crop = None

''' Parallelization Inputs'''
options = ['local-mpidirect', 'nersc-mpidirect',]
option = options[1]
if option == 'local-mpidirect':
    USER_runCfg_type = 'mpi_bulletin'
    USER_mpiCommand = 'mpirun -bootstrap fork'
elif option == 'nersc-mpidirect':
    USER_runCfg_type = 'mpi_bulletin'
    USER_mpiCommand = 'mpirun' 
    USER_nodes = 4
    USER_cores_per_indv = 128/USER_pop_size
    USER_cores_per_node = USER_cores_per_indv
else: 
    print('Invalid Parallelization Option')
    sys.exit()

## Overwrite and Continue
USER_overwrite_run = False
USER_continue_run = False

