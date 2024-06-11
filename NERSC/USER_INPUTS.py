import os
import pandas as pd

'''init'''
script_path = os.path.dirname(os.path.realpath(__file__)) #get script path

'''parse arguments'''
import argparse
parser = argparse.ArgumentParser(description='Batch Run')
#parser.add_argument('run_label', type=str, help='Run Label')
parser.add_argument('-rp', '--run_path', type=str, help='Run Path')
parser.add_argument('-d', '--duration', type=int, help='Duration')
parser.add_argument('-l', '--label', type=str, help='Label')
parser.add_argument('--pop_size', type=int, help='Population Size')
parser.add_argument('--mpi_type', type=str, help='MPI type')
parser.add_argument('-t', '--tasks', type=int, help='Tasks')
parser.add_argument('-rr', '--rerun', action='store_true', help='Rerun Mode')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose Mode')
parser.add_argument('-seed', '--seed_evol', action='store_true', help='Seed Evolution')
parser.add_argument('-hof', '--hof', action='store_true', help='Hall of Fame')
parser.add_argument('-gens', '--max_generations', type=int, help='Max Generations')
args, unknown = parser.parse_known_args()

'''HANDLE ARGS'''
## DURATION
if args.duration: USER_seconds = args.duration
else: USER_seconds = 1
## POP SIZE
if args.pop_size: USER_pop_size = args.pop_size
else: USER_pop_size = 4
## MAX GENERATIONS
if args.max_generations: USER_max_generations = args.max_generations
else: USER_max_generations = 3000
## SEED EVOL (HOF)
if args.hof: 
    USER_HOF = f'{script_path}/HOF/hof.csv' #seed gen 0 with solutions in HOF.csv
    USER_HOF = os.path.abspath(USER_HOF)
## RERUN MODE
if args.rerun: 
    rerun_mode = True
    USER_HOF = f'{script_path}/rerun/rerun.csv' #seed gen 0 with solutions in HOF.csv
    USER_HOF = os.path.abspath(USER_HOF)
    USER_pop_size = len(pd.read_csv(USER_HOF).values.flatten()) #override pop_size with HOF size
    USER_max_generations=1 #override max_generations with 1 for rerun mode
else: 
    USER_seed_evol = True
    rerun_mode = False    
## MPI COMMANDS
#print(f'USER_runCfg_type: {args.mpi_type}')
if args.mpi_type: USER_runCfg_type = args.mpi_type
else:     
    USER_runCfg_type = 'mpi_direct'
    #USER_runCfg_type = 'mpi_bulletin'
try: assert USER_runCfg_type in ['mpi_direct', 'mpi_bulletin'], 'Invalid MPI type. Must be "mpi_direct" or "mpi_bulletin"'
except: USER_runCfg_type = 'mpi_direct'; print(f'Invalid MPI type. Must be "mpi_direct" or "mpi_bulletin". Defaulting to {USER_runCfg_type}')
## RUN LABEL
if args.label: USER_run_label = args.label    
else: 
    USER_runCfg_type = 'mpi_bulletin'
    USER_run_label = 'vscode'
    USER_pop_size = 4 #if running locally, or in vscode, keep pop_size small
    if rerun_mode: USER_pop_size = len(pd.read_csv(USER_HOF).values.flatten()) #override pop_size with HOF size
    #run_path, run_name, _ = init_new_batch(USER_run_label, run_path_only = False)
if 'mpi_direct' in USER_runCfg_type:    
    USER_shifterCommmand = 'shifter --image=adammwea/netpyneshifter:v5'
    #USER_nrnCommand = f'--cpu_bind=cores {USER_shifterCommmand} nrniv'
    USER_nrnCommand = f'--sockets-per-node 1 --cpu_bind=cores {USER_shifterCommmand} nrniv'
    
    '''HACkz (send srun commands to command.txt via srun_extractor.py)'''
    #include sleep to allow watcher to catch up
    HACKz = f'\
        \nsleep 10\
        \necho $(pwd)\
        \npython srun_extractor.py'       
    USER_mpiCommand = 'srun -N 1'
    HACKz_mpiCommand = f'{HACKz} {USER_mpiCommand}'
    USER_mpiCommand = HACKz_mpiCommand
    #print(f'USER_mpiCommand: {USER_mpiCommand}')
elif 'mpi_bulletin' in USER_runCfg_type:
    USER_mpiCommand = 'mpirun'
## TASKS
if args.tasks: USER_mpis_per_batch = args.tasks
else: USER_mpis_per_batch = 16
assert int(USER_mpis_per_batch) == USER_mpis_per_batch, 'USER_mpis_per_batch must be an integer'
assert USER_mpis_per_batch > 0, 'USER_mpis_per_batch must be greater than 0'
USER_cores_per_node = USER_mpis_per_batch #send mpi tasks to NetPyNE as num cores even if this isnt technically true    
## RUN PATH
if args.run_path: USER_run_path = args.run_path
else: USER_run_path = None
    # run_path, run_name, _ = init_new_batch(USER_run_label, run_path_only = True)
    # USER_run_path = run_path

'''Extrapolate from args'''
## Elites
USER_frac_elites = 0.2 # must be 0 < USER_frac_elites < 1. This is the fraction of elites in the population.
USER_num_elites = int(USER_frac_elites * USER_pop_size) if USER_frac_elites > 0 else 1

'''Simulation options'''    
USER_method = 'evol' #'evol', 'grid', 'asd'
USER_init_script = f'{script_path}/init.py'
USER_cfgFile = f'{script_path}/cfg.py'

'''Network Inputs'''
USER_netParamsFile = f'{script_path}/netParams.py'
assert os.path.exists(USER_init_script), f'initFile does not exist: {USER_init_script}'
assert os.path.exists(USER_cfgFile), f'cfgFile does not exist: {USER_cfgFile}'
assert os.path.exists(USER_netParamsFile), f'netParamsFile does not exist: {USER_netParamsFile}'

'''Batch Options'''
USER_nodes = 1  #Using the payload method, this doesnt really matter. 
                #It's important that each command is run on a single node, so it's hardcoded in.
USER_HOF = f'{script_path}/HOF/hof.csv' #seed gen 0 with solutions in HOF.csv
USER_HOF = os.path.abspath(USER_HOF)
USER_plot_fitness = True
maxiter_wait_minutes = 10 #Maximum minutes to wait before starting new Generation
USER_time_sleep = 10 #seconds between checking for completed simulations
USER_maxiter_wait = maxiter_wait_minutes*60/USER_time_sleep
USER_mutation_rate = 0.7
USER_crossover = 0.5

'''Overwrite Options'''
USER_skip = True #Skip running the simulation if data already exist
USER_overwrite = False #Overwrite existing batch_run with same name
USER_continue = True #Continue from last completed simulation
if USER_continue: USER_skip = True #continue will re-run existing simulations if skip is False   

'''Plotting Inputs'''
USER_plot_fitness = False
USER_plotting_path = 'NERSC/plots/'
#USER_ploting_path = None #prevent plotting even if USER_plot_fitness_bool = True
USER_svg_mode = False
USER_figsize = (10, 10)
USER_plotting_params = {
    'fresh_plots': True,
    'saveFig': USER_plotting_path,
    'figsize': USER_figsize,
    'NetworkActivity': {
        'figsize': USER_figsize,
        # limits (will override modifiers)
        #'ylim': None,
        'ylim': 3000,
        'xlim': None,
         #range mods
        'yhigh100': 1.05, #high modifier limit for y axis
        'ylow100': 0.95, #low modifier limit for y axis
        'saveFig': USER_plotting_path,
        'title_font': 11
        }
    }

'''Fitness Inputs'''
# KCNT1 params
USER_raster_convolve_params = {
    'binSize': .1,
    'gaussianSigma': .15,
    'thresholdBurst': 2.0,
    'min_peak_distance': 1, #sec
    }
USER_raster_crop = None

