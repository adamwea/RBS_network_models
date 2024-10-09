import os
import datetime
import shutil
import argparse
import pandas as pd

# Constants
DEFAULT_DURATION = 1
DEFAULT_POP_SIZE = 10
DEFAULT_MAX_GENERATIONS = 1
DEFAULT_MPI_TYPE = 'mpi_bulletin'
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
DEFAULT_USER_SEED_EVOL = False
DEFAULT_METHOD = 'evol'
DEFAULT_CFG_FILE = 'simulate_config_files/cfg.py'
DEFAULT_NETPARAMSFILE = 'simulate_config_files/netParams.py'
DEFAULT_INIT_SCRIPT = 'simulate_config_files/init.py'
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
    return parser.parse_args()

# Function to get or set default argument values
def get_arg_value(arg, default):
    return arg if arg is not None else default

# Function to handle rerun mode
def handle_rerun_mode(script_path):
    hof_path = os.path.abspath(f'{script_path}/rerun/rerun.csv')
    pop_size = len(pd.read_csv(hof_path).values.flatten())
    return hof_path, pop_size, 1

# Main configuration setup
def configure_run(args):
    global USER_seconds, USER_pop_size, USER_max_generations, USER_HOF, USER_runCfg_type
    global USER_run_label, USER_mpiCommand, USER_mpis_per_batch, USER_run_path
    global USER_continue, USER_overwrite, USER_output_path
    global USER_seed_evol, USER_method, USER_cfg_file, USER_netParamsFile, USER_init_script
    global USER_nrnCommand, USER_nodes, USER_cores_per_node, USER_skip
    global USER_num_elites, USER_mutation_rate, USER_crossover, USER_time_sleep, USER_maxiter_wait
    global USER_plot_fitness

    USER_seed_evol = get_arg_value(args.seed_evol, DEFAULT_USER_SEED_EVOL)
    USER_method = get_arg_value(args.method, DEFAULT_METHOD)
    USER_cfg_file = get_arg_value(args.cfg_file, DEFAULT_CFG_FILE)
    USER_netParamsFile = get_arg_value(args.netParamsFile, DEFAULT_NETPARAMSFILE)
    USER_init_script = get_arg_value(args.init_script, DEFAULT_INIT_SCRIPT)
    USER_nrnCommand = get_arg_value(args.nrnCommand, DEFAULT_NRNCOMMAND)
    USER_nodes = get_arg_value(args.nodes, DEFAULT_NODES)
    USER_cores_per_node = get_arg_value(args.cores_per_node, DEFAULT_CORES_PER_NODE)
    USER_skip = get_arg_value(args.skip, DEFAULT_SKIP)
    USER_num_elites = get_arg_value(args.num_elites, DEFAULT_NUM_ElITES)
    USER_mutation_rate = get_arg_value(args.mutation_rate, DEFAULT_MUTATION_RATE)
    USER_crossover = get_arg_value(args.crossover, DEFAULT_CROSSOVER_RATE)
    USER_time_sleep = get_arg_value(args.time_sleep, DEFAULT_TIME_SLEEP)
    USER_maxiter_wait = get_arg_value(args.maxiter_wait, DEFAULT_MAXITER_WAIT)
    USER_plot_fitness = get_arg_value(args.plot_fitness, DEFAULT_PLOT_FITNESS)

    if args.rerun:
        USER_HOF, USER_pop_size, USER_max_generations = handle_rerun_mode(SCRIPT_PATH)
    else:
        USER_HOF = os.path.abspath(f'{SCRIPT_PATH}/HOF/hof.csv')
        USER_pop_size = get_arg_value(args.pop_size, DEFAULT_POP_SIZE)
        USER_max_generations = get_arg_value(args.max_generations, DEFAULT_MAX_GENERATIONS)

    USER_seconds = get_arg_value(args.duration, DEFAULT_DURATION)
    USER_runCfg_type = get_arg_value(args.mpi_type, DEFAULT_MPI_TYPE)

    if USER_runCfg_type not in ['mpi_direct', 'mpi_bulletin']:
        print(f'Invalid MPI type. Defaulting to {DEFAULT_MPI_TYPE}')
        USER_runCfg_type = DEFAULT_MPI_TYPE
    
    USER_run_label = get_arg_value(args.label, 'vscode')

    if 'mpi_direct' in USER_runCfg_type:
        USER_mpiCommand = configure_mpi_direct(SCRIPT_PATH)
    else:
        USER_mpiCommand = 'mpirun'

    USER_mpis_per_batch = get_arg_value(args.tasks, 16)
    USER_run_path = args.run_path or None
    USER_output_path = get_arg_value(args.simulation_output_path, 'simulation_output')

    #continue or not to continue, that is the question
    USER_continue = get_arg_value(args.continue_run, False)
    USER_overwrite = get_arg_value(args.overwrite, False)
    assert not (USER_continue and USER_overwrite), 'overwrite_run and continue_run cannot both be True'


# Separate configuration for MPI commands
def configure_mpi_direct(script_path):
    shifter_command = 'shifter --image=adammwea/netpyneshifter:v5'
    mpi_command = f'srun -N 1 --sockets-per-node 1 --cpu_bind=cores {shifter_command} nrniv'
    return f'''\nsleep 10\npython srun_extractor.py {mpi_command}'''

# Function to initialize batch run
def init_batch_pathing(USER_run_label=None, run_path_only=False):
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

    return run_path, run_name, USER_run_label

# Function to save configuration to a Python file
def save_config_to_file():
    if os.path.exists("temp_user_args.py"):
        os.remove("temp_user_args.py")
    
    with open("temp_user_args.py", "w") as f:
        f.write(f"USER_seconds = {USER_seconds}\n")
        f.write(f"USER_pop_size = {USER_pop_size}\n")
        f.write(f"USER_max_generations = {USER_max_generations}\n")
        f.write(f"USER_HOF = '{USER_HOF}'\n")
        f.write(f"USER_runCfg_type = '{USER_runCfg_type}'\n")
        f.write(f"USER_run_label = '{USER_run_label}'\n")        
        f.write(f"USER_mpis_per_batch = {USER_mpis_per_batch}\n")
        f.write(f"USER_run_path = '{USER_run_path}'\n")
        f.write(f"USER_continue = {USER_continue}\n")
        f.write(f"USER_overwrite = {USER_overwrite}\n")
        f.write(f"USER_output_path = '{USER_output_path}'\n")
        f.write(f"USER_seed_evol = {USER_seed_evol}\n")
        f.write(f"USER_method = '{USER_method}'\n")
        f.write(f"USER_cfg_file = '{USER_cfg_file}'\n")
        f.write(f"USER_netParamsFile = '{USER_netParamsFile}'\n")
        f.write(f"USER_init_script = '{USER_init_script}'\n")
        f.write(f"USER_nodes = {USER_nodes}\n")
        f.write(f"USER_cores_per_node = {USER_cores_per_node}\n")
        f.write(f"USER_skip = {USER_skip}\n")
        f.write(f"USER_num_elites = {USER_num_elites}\n")
        f.write(f"USER_mutation_rate = {USER_mutation_rate}\n")
        f.write(f"USER_crossover = {USER_crossover}\n")
        f.write(f"USER_time_sleep = {USER_time_sleep}\n")
        f.write(f"USER_maxiter_wait = {USER_maxiter_wait}\n")
        f.write(f"USER_plot_fitness = {USER_plot_fitness}\n")
        f.write(f"USER_mpiCommand = '{USER_mpiCommand}'\n")
        f.write(f"USER_nrnCommand = '{USER_nrnCommand}'\n")
        f.write(f"USER_continue = {USER_continue}\n")
        f.write(f"USER_overwrite = {USER_overwrite}\n")


        # f.write("USER_mpiCommand = '''")
        # f.write(f"{USER_mpiCommand}\n")
        # f.write("'''\n")

# Main function
def main(**kwargs):
    global USER_run_path, USER_output_path, USER_HOF, USER_cfg_file, USER_netParamsFile, USER_init_script, USER_continue, USER_overwrite

    # Parse the arguments
    args = parse_arguments()
    for key, value in kwargs.items():
        if value is not None:
            setattr(args, key, value)
    configure_run(args)
    
    # Initialize batch run and save the results
    run_path, run_name, USER_run_label = init_batch_pathing(args.label)
    USER_run_path = run_path

    #get absolute paths for all paths
    USER_HOF = os.path.abspath(USER_HOF)
    USER_cfg_file = os.path.abspath(USER_cfg_file)
    USER_netParamsFile = os.path.abspath(USER_netParamsFile)
    USER_init_script = os.path.abspath(USER_init_script)
    USER_output_path = os.path.abspath(USER_output_path)
    USER_run_path = os.path.abspath(USER_run_path)
    
    # Save the config variables to temp_user_args.py
    save_config_to_file()

    # Save run_path to a temporary file for future reference
    script_path = os.path.dirname(os.path.realpath(__file__))
    temp_path = f'{script_path}/temp'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    with open(f'{temp_path}/run_path.txt', 'w') as file:
        file.write(run_path)

if __name__ == '__main__':
    main()