import os
import datetime
import shutil
import argparse
import pandas as pd
import simulate._config_files.setup_environment as setup_environment
import os
import shutil

# Constants
DEFAULT_DURATION = 1
DEFAULT_POP_SIZE = 10
DEFAULT_MAX_GENERATIONS = 1
DEFAULT_MPI_TYPE = 'mpi_bulletin'
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
DEFAULT_USER_SEED_EVOL = False
DEFAULT_METHOD = 'evol'
# DEFAULT_CFG_FILE = 'simulate_config_files/cfg.py'
# DEFAULT_NETPARAMSFILE = 'simulate_config_files/netParams.py'
# DEFAULT_INIT_SCRIPT = 'simulate_config_files/init.py'
DEFAULT_CFG_FILE = 'simulate/_config_files/cfg.py'
DEFAULT_NETPARAMSFILE = 'simulate/_config_files/netParams.py'
DEFAULT_INIT_SCRIPT = 'simulate/_config_files/init.py'
DEFAULT_NRNCOMMAND = 'nrniv'
DEFAULT_NODES = 1
DEFAULT_CORES_PER_NODE = 1
DEFAULT_SKIP = False
DEFAULT_NUM_ElITES = 1
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_CROSSOVER_RATE = 0.5
DEFAULT_TIME_SLEEP = 10 #seconds
DEFAULT_MAXITER_WAIT = 50 #iterations
DEFAULT_PLOT_FITNESS = False #plot fitness while simulation is running - less efficient to do so
DEFAULT_CONTINUE = False
DEFAULT_OVERWRITE = False
DEFAULT_NUM_EXCITE = 75
DEFAULT_NUM_INHIB = 25
DEFAULT_WORKSPACE_PATH = setup_environment.get_git_root()
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_WORKSPACE_PATH, 'xRBS_network_simulation_outputs')
DEFAULT_FITNESS_TARGET_SCRIPT = os.path.join(DEFAULT_WORKSPACE_PATH, 'simulate/_fitness_targets/fitnessFuncArgs_CDKL5_WT.py')

# Global Variables (for export)
USER_seconds = None
USER_pop_size = None
USER_max_generations = None
USER_HOF = None
USER_runCfg_type = None
USER_run_label = None
USER_mpiCommand = None
USER_mpis_per_batch = None
USER_run_path = None
USER_continue = None
USER_overwrite = None
USER_output_path = None
USER_seed_evol = None
USER_method = None
USER_cfg_file = None
USER_netParamsFile = None
USER_init_script = None
USER_nrnCommand = None
USER_nodes = None
USER_cores_per_node = None
USER_skip = None
USER_num_elites = None
USER_mutation_rate = None
USER_crossover = None
USER_time_sleep = None
USER_maxiter_wait = None
USER_plot_fitness = None
USER_continue = None
USER_overwrite = None
USER_num_excite = None
USER_num_inhib = None
USER_output_dir = None
USER_fitness_target_script = None

# Function to get or set default argument values
def get_arg_value(arg, default):
    return arg if arg is not None else default

# # Function to handle rerun mode
# def handle_rerun_mode(script_path):
#     hof_path = os.path.abspath(f'{script_path}/rerun/rerun.csv')
#     pop_size = len(pd.read_csv(hof_path).values.flatten())
#     return hof_path, pop_size, 1

def check_for_existing_runs(output_path, current_date):
    '''Check for existing runs in the output path'''
    try:
        existing_runs = [run for run in os.listdir(output_path) if run.startswith(current_date)]
    except:
        existing_runs = []
        
    # Find the highest run number for the day
    if existing_runs:
        highest_run_number = max(int(run.split('_Run')[1].split('_')[0]) for run in existing_runs)
    else:
        highest_run_number = 0
    return existing_runs, highest_run_number

def build_run_path(run_path, label, output_path, overwrite_run, continue_run):
    '''Build the output path for the run'''
    
    '''Initialize variables'''
    current_date = datetime.datetime.now().strftime('%y%m%d')     # Get current date in YYMMDD format
    if output_path is None: output_path = DEFAULT_OUTPUT_DIR     # Set the output path
    print(f'Output Path: {output_path}')

    existing_runs, highest_run_number = check_for_existing_runs(output_path, current_date)     # Get list of existing runs for the day

    # Increment the run number for the new run
    new_run_number = highest_run_number + 1
    prev_run_number = new_run_number - 1

    # Update run_name with the new format
    run_name = f'{current_date}_Run{new_run_number}_{label}'
    prev_run_name = f'{current_date}_Run{prev_run_number}_{label}'

    # Get unique run path
    run_path = os.path.join(output_path, run_name)
    prev_run_path = os.path.join(output_path, prev_run_name)

    #print(f'New run path: {run_path}')
    #print(f'Previous run path: {prev_run_path}')

    # Check if run exists and if it should be overwritten or continued
    if overwrite_run or continue_run:
        if prev_run_name in existing_runs:
            if overwrite_run and os.path.exists(prev_run_path):
                print(f'Overwriting previous run at: {prev_run_path}')
                print(f'Deleting previous run at: {prev_run_path}')
                shutil.rmtree(prev_run_path)
                #assert that prev_run_path has been deleted
                assert not os.path.exists(prev_run_path), f'Previous run at {prev_run_path} was not deleted'
                run_path = prev_run_path
            elif continue_run and os.path.exists(prev_run_path):
                print(f'Continuing previous run at: {prev_run_path}')
                run_path = prev_run_path

    # Create the new run directory if it does not exist
    if not os.path.exists(run_path):
        print(f'Creating new run directory at: {run_path}')
        os.makedirs(run_path)

    return run_path

# Function to handle argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Batch Run')
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
    parser.add_argument('-op', '--simulation_output_path', type=str, help='Simulation Output Path')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite Run') 
    parser.add_argument('-c', '--continue_run', action='store_true', help='Continue Run')
    parser.add_argument('-m', '--method', type=str, help='Method')
    parser.add_argument('-cfg', '--cfg_file', type=str, help='Configuration File')
    parser.add_argument('-npf', '--netParamsFile', type=str, help='NetParams File')
    parser.add_argument('-is', '--init_script', type=str, help='Run Configuration Initialization Script')
    parser.add_argument('-nrn', '--nrnCommand', type=str, help='NEURON Command')
    parser.add_argument('-n', '--nodes', type=int, help='Nodes')
    parser.add_argument('-cpn', '--cores_per_node', type=int, help='Cores per Node')
    parser.add_argument('-s', '--skip', action='store_true', help='Skip')
    parser.add_argument('-el', '--num_elites', type=int, help='Number of Elites')
    parser.add_argument('-mr', '--mutation_rate', type=float, help='Mutation Rate')
    parser.add_argument('-cr', '--crossover', type=float, help='Crossover Rate')
    parser.add_argument('-ts', '--time_sleep', type=int, help='Time Sleep')
    parser.add_argument('-mw', '--maxiter_wait', type=int, help='Max Iterations Wait')
    parser.add_argument('-pf', '--plot_fitness', action='store_true', help='Plot Fitness')
    parser.add_argument('-ne', '--num_excite', type=int, help='Number of Excitatory Cells')
    parser.add_argument('-ni', '--num_inhib', type=int, help='Number of Inhibitory Cells')
    parser.add_argument('-ft', '--fitness_target_script', type=str, help='Fitness Target Script')
    return parser.parse_args()

# Main configuration setup
def configure_global_user_vars(**kwargs):
    
    '''Parse the arguments from the command line'''
    args = parse_arguments()
    
    '''Merge the arguments with the kwargs'''
    for key, value in kwargs.items():
        if value is not None:
            setattr(args, key, value)
            
    '''Build output and run paths'''
    run_path = build_run_path(args.run_path, args.label, args.output_path, args.overwrite, args.continue_run)

    # Initialize USER variables
    USER_vars = {
        'USER_seed_evol': get_arg_value(args.seed_evol, DEFAULT_USER_SEED_EVOL),
        'USER_method': get_arg_value(args.method, DEFAULT_METHOD),
        'USER_cfg_file': get_arg_value(args.cfg_file, DEFAULT_CFG_FILE),
        'USER_netParamsFile': get_arg_value(args.netParamsFile, DEFAULT_NETPARAMSFILE),
        'USER_init_script': get_arg_value(args.init_script, DEFAULT_INIT_SCRIPT),
        'USER_nrnCommand': get_arg_value(args.nrnCommand, DEFAULT_NRNCOMMAND),
        'USER_nodes': get_arg_value(args.nodes, DEFAULT_NODES),
        'USER_cores_per_node': get_arg_value(args.cores_per_node, DEFAULT_CORES_PER_NODE),
        'USER_skip': get_arg_value(args.skip, DEFAULT_SKIP),
        'USER_num_elites': get_arg_value(args.num_elites, DEFAULT_NUM_ElITES),
        'USER_mutation_rate': get_arg_value(args.mutation_rate, DEFAULT_MUTATION_RATE),
        'USER_crossover': get_arg_value(args.crossover, DEFAULT_CROSSOVER_RATE),
        'USER_time_sleep': get_arg_value(args.time_sleep, DEFAULT_TIME_SLEEP),
        'USER_maxiter_wait': get_arg_value(args.maxiter_wait, DEFAULT_MAXITER_WAIT),
        'USER_plot_fitness': get_arg_value(args.plot_fitness, DEFAULT_PLOT_FITNESS),
        'USER_num_excite': get_arg_value(args.num_excite, DEFAULT_NUM_EXCITE),
        'USER_num_inhib': get_arg_value(args.num_inhib, DEFAULT_NUM_INHIB),
        'USER_seconds': get_arg_value(args.duration, DEFAULT_DURATION),
        'USER_runCfg_type': get_arg_value(args.mpi_type, DEFAULT_MPI_TYPE),
        'USER_run_label': get_arg_value(args.label, 'vscode'),
        'USER_mpis_per_batch': get_arg_value(args.tasks, 16),
        'USER_run_path': run_path,
        #'USER_output_path': get_arg_value(args.simulation_output_path, 'simulation_output'),
        'USER_output_path': args.output_path,
        'USER_continue': get_arg_value(args.continue_run, False),
        'USER_overwrite': get_arg_value(args.overwrite, False),
        'USER_fitness_target_script': get_arg_value(args.fitness_target_script, DEFAULT_FITNESS_TARGET_SCRIPT)
    }

    # if args.rerun:
    #     USER_vars['USER_HOF'], USER_vars['USER_pop_size'], USER_vars['USER_max_generations'] = handle_rerun_mode(SCRIPT_PATH)
    # else:
    #USER_vars['USER_HOF'] = os.path.abspath(f'{SCRIPT_PATH}/HOF/hof.csv') #TODO: Reimplement this a bit later
    USER_vars['USER_pop_size'] = get_arg_value(args.pop_size, DEFAULT_POP_SIZE)
    USER_vars['USER_max_generations'] = get_arg_value(args.max_generations, DEFAULT_MAX_GENERATIONS)

    if USER_vars['USER_runCfg_type'] not in ['mpi_direct', 'mpi_bulletin']:
        print(f'Invalid MPI type. Defaulting to {DEFAULT_MPI_TYPE}')
        USER_vars['USER_runCfg_type'] = DEFAULT_MPI_TYPE

    if 'mpi_direct' in USER_vars['USER_runCfg_type']:
        USER_vars['USER_mpiCommand'] = configure_mpi_direct(SCRIPT_PATH)
        USER_vars['USER_nrnCommand'] = ''
    else:
        USER_vars['USER_mpiCommand'] = 'mpirun'

    assert not (USER_vars['USER_continue'] and USER_vars['USER_overwrite']), 'overwrite_run and continue_run cannot both be True'

    return USER_vars


# Separate configuration for MPI commands
def configure_mpi_direct(script_path):
    shifter_command = 'shifter --image=adammwea/netpyneshifter:v5'
    mpi_command = f'srun -N 1 --sockets-per-node 1 --cpu_bind=cores {shifter_command} nrniv'
    return f'''\
sleep 10
python ./sbatch_scripts/srun_extractor.py {mpi_command}'''

# Function to initialize batch run
def init_batch_pathing(USER_run_label=None, run_path_only=False, **kwargs):
    '''
    Initialize batch run with default label "vscode" if not provided.
    '''
    # Set default run label to "vscode" if none is provided
    if USER_run_label is None:
        USER_run_label = "vscode"

    # Get current date in YYMMDD format
    current_date = datetime.datetime.now().strftime('%y%m%d')

    # Set the output path
    if USER_output_path is None:
        output_path = 'simulation_output'
    else:
        output_path = USER_output_path

    # Get list of existing runs for the day
    try:
        existing_runs = [run for run in os.listdir(output_path) if run.startswith(current_date)]
    except:
        existing_runs = []

    # Find the highest run number for the day
    if existing_runs:
        highest_run_number = max(int(run.split('_Run')[1].split('_')[0]) for run in existing_runs)
    else:
        highest_run_number = 0

    # Increment the run number for the new run
    new_run_number = highest_run_number + 1
    prev_run_number = new_run_number - 1

    # Update run_name with the new format
    run_name = f'{current_date}_Run{new_run_number}_{USER_run_label}'
    prev_run_name = f'{current_date}_Run{prev_run_number}_{USER_run_label}'

    # Get unique run path
    run_path = f'{output_path}/{run_name}'
    prev_run_path = f'{output_path}/{prev_run_name}'

    '''Check if run exists and if it should be overwritten or continued'''
    if USER_overwrite or USER_continue:
        if prev_run_name in existing_runs:
            if USER_overwrite and os.path.exists(prev_run_path):
                #if not run_path_only:
                    #shutil.rmtree(prev_run_path)
                shutil.rmtree(prev_run_path)
                run_path = prev_run_path   
            elif USER_continue and os.path.exists(prev_run_path):
                run_path = prev_run_path

    # Create the new run directory if it does not exist
    if not os.path.exists(run_path):
        os.makedirs(run_path)
        
    '''Get absolute paths'''
    #get absolute paths for all paths
    USER_HOF = os.path.abspath(USER_HOF)
    USER_cfg_file = os.path.abspath(USER_cfg_file)
    USER_netParamsFile = os.path.abspath(USER_netParamsFile)
    USER_init_script = os.path.abspath(USER_init_script)
    USER_output_path = os.path.abspath(USER_output_path)
    USER_run_path = os.path.abspath(USER_run_path)

def send_user_vars_to_bash(USER_vars):
    '''Send the USER_vars to the bash environment so they can be accessed by slurm sbatch scripts'''
    for key, value in USER_vars.items():
        os.environ[key] = str(value)
        # Export the variable to the bash environment
        print(f'export {key}="{value}"')
        os.system(f'export {key}="{value}"')
        
def save_config_to_file(USER_vars):
    run_path = USER_vars['USER_run_path']
    temp_user_args_path = f'temp_user_args.py'
    user_args_path = f'{run_path}/temp_user_args.py'
    
    if os.path.exists(temp_user_args_path):
        os.remove(temp_user_args_path)
    
    with open(temp_user_args_path, "w") as f:
        for key, value in USER_vars.items():
            if isinstance(value, str):
                if '\n' in value:
                    f.write(f"{key} = '''{value}'''\n")
                else:
                    f.write(f"{key} = '{value}'\n")
            elif isinstance(value, bool):
                f.write(f"{key} = {value}\n")
            elif isinstance(value, int):
                f.write(f"{key} = {value}\n")
            elif isinstance(value, float):
                f.write(f"{key} = {value}\n")
            else:
                f.write(f"{key} = {value}\n")
    
    # Copy the temp_user_args.py file to the run_path
    shutil.copy(temp_user_args_path, user_args_path)
            
        # f.write(f"USER_seconds = {USER_seconds}\n")
        # f.write(f"USER_pop_size = {USER_pop_size}\n")
        # f.write(f"USER_max_generations = {USER_max_generations}\n")
        # f.write(f"USER_HOF = '{USER_HOF}'\n")
        # f.write(f"USER_runCfg_type = '{USER_runCfg_type}'\n")
        # f.write(f"USER_run_label = '{USER_run_label}'\n")        
        # f.write(f"USER_mpis_per_batch = {USER_mpis_per_batch}\n")
        # f.write(f"USER_run_path = '{USER_run_path}'\n")
        # f.write(f"USER_continue = {USER_continue}\n")
        # f.write(f"USER_overwrite = {USER_overwrite}\n")
        # f.write(f"USER_output_path = '{USER_output_path}'\n")
        # f.write(f"USER_seed_evol = {USER_seed_evol}\n")
        # f.write(f"USER_method = '{USER_method}'\n")
        # f.write(f"USER_cfg_file = '{USER_cfg_file}'\n")
        # f.write(f"USER_netParamsFile = '{USER_netParamsFile}'\n")
        # f.write(f"USER_init_script = '{USER_init_script}'\n")
        # f.write(f"USER_nodes = {USER_nodes}\n")
        # f.write(f"USER_cores_per_node = {USER_cores_per_node}\n")
        # f.write(f"USER_skip = {USER_skip}\n")
        # f.write(f"USER_num_elites = {USER_num_elites}\n")
        # f.write(f"USER_mutation_rate = {USER_mutation_rate}\n")
        # f.write(f"USER_crossover = {USER_crossover}\n")
        # f.write(f"USER_time_sleep = {USER_time_sleep}\n")
        # f.write(f"USER_maxiter_wait = {USER_maxiter_wait}\n")
        # f.write(f"USER_plot_fitness = {USER_plot_fitness}\n")
        # f.write(f"USER_mpiCommand = '{USER_mpiCommand}'\n")
        # f.write(f"USER_nrnCommand = '{USER_nrnCommand}'\n")
        # f.write(f"USER_continue = {USER_continue}\n")
        # f.write(f"USER_overwrite = {USER_overwrite}\n")
        # f.write(f"USER_num_excite = {USER_num_excite}\n")
        # f.write(f"USER_num_inhib = {USER_num_inhib}\n")


        # f.write("USER_mpiCommand = '''")
        # f.write(f"{USER_mpiCommand}\n")
        # f.write("'''\n")

# Main function
def main(**kwargs):
    '''Main function to configure the global variables and save the configuration to a file.'''
    USER_vars = configure_global_user_vars(**kwargs)     # Parse the arguments
    #send_user_vars_to_bash(USER_vars)     # Send the USER_vars to the bash environment so they can be accessed by slurm sbatch scripts   
    save_config_to_file(USER_vars)     # Save the config variables to temp_user_args.py


    # Save run_path to a temporary file for future reference
    # run_output_dir = kwargs.get('run_output_dir', None)
    # script_path = os.path.dirname(os.path.realpath(__file__))
    # temp_path = f'{script_path}/temp'
    # if not os.path.exists(temp_path):
    #     os.makedirs(temp_path)
    # with open(f'{temp_path}/run_path.txt', 'w') as file:
    #     file.write(run_path)

if __name__ == '__main__':
    main()