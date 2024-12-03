''' 

batchRun_evol_srun_direct.py - Batch run script for evolutionary optimization using srun direct MPI command

using this script for early development and testing of the batchRun function.

'''

def init_user_kwargs():        
    '''Initialize user kwargs
    note: if new kwargs are added, they must be added to the parse_kwargs.main() function in the parse_kwargs.py script
    '''
    
    # User arguments
    output_folder_name = 'yThroughput/zRBS_network_simulation_outputs'
    #run_label = f'improved_netparams'  # subfolder for this run will be created in output_folder_path
    run_label = f'CDKL5_seeded_evol'  # subfolder for this run will be created in output_folder_path
    batch_type = 'evol'
    max_wait_time_minutes = 60 #minutes - maximum time to wait for a generation or stalled simulation to finish new candidates
    time_sleep = 1 #seconds - time to sleep between checking for new candidates
    
    # Fitness target scripts
    #fitness_target_script = '/pscratch/sd/a/adammwea/RBS_network_simulations/tunning_scripts/CDKL5-E6D_T2_C1_DIV21/derived_fitness_args/fitness_args_20241123-155335.py'
    fitness_target_script = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_1_derive_features_from_experimental_data/derived_fitness_args/fitness_args_20241202_145331.py'
    
    # seed scripts
    seed_script = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_3_plot_all_runs_and_select_seeds/_2_seed_review_241126_Run2_improved_netparams.py'
    
    # read seed_script to get seed paths
    seed_paths = None
    import importlib.util
    spec = importlib.util.spec_from_file_location("seed_module", seed_script)
    seed_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed_module)    
    # load dict, seed_paths from seed_script
    seed_paths = seed_module.seed_paths
    
    # param_space_script
    param_space_script = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/parameter_spaces/_241202_adjusted_evol_params.py'
    
    # read param_space_script to get param_space
    param_space = None
    spec = importlib.util.spec_from_file_location("param_space_module", param_space_script)
    param_space_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(param_space_module)
    # load dict, param_space from param_space_script
    param_space = param_space_module.params
    
    # user kwargs
    kwargs = {        
        # netpyne kwargs
        'duration': 25, # Duration of the simulation, in seconds
        #'pop_size': 128,
        'pop_size': 64,
        'num_elites': 30,
        'max_generations': 100,
        'time_sleep': time_sleep, #seconds per iteration
        'maxiter_wait': max_wait_time_minutes*60/time_sleep, #max wait time in iterations
        'batch_type': batch_type,
        # ... other netpyne kwargs can be added here and they should work as expected
        
        # custom kwargs
        'continue_run': True, #specified in main - idk which i prefer
        'overwrite': False,
        'label': run_label,
        #'output_path': output_folder_path,
        'output_folder_name': output_folder_name, #configure_command will use this to build the output folder path
        'fitness_target_script': fitness_target_script,
        'outside_of_repo': True, #if True, the output folder will be created outside of the git repo
        'recalculate_fitness': False, #if True, when continuing a run, the fitness will be recalculated for all candidates
        'seed_evol': True, #if True, the initial population will be seeded with the best candidates curated from previous generations
        'seed_paths': seed_paths,
        'param_space': param_space,
        # ... other custom kwargs can be added here but then they must be added to the parse_kwargs.main() function in the parse_kwargs.py script
    }
    
    return kwargs    

def mock_command_example(**kwargs):

        type=kwargs.get('type', 'mpi_direct')
    
        # Copy pasted (and slightly modifed) from batch/utils.py
        if type == 'mpi_bulletin':
            # ----------------------------------------------------------------------
            # MPI master-slaves
            # ----------------------------------------------------------------------
            pc.submit(runJob, nrnCommand, script, cfgSavePath, netParamsSavePath, jobPath, pc.id())
            print('-' * 80)
        else:
            # ----------------------------------------------------------------------
            # MPI job command
            # ----------------------------------------------------------------------

            if mpiCommand == '':
                command = '%s %s simConfig=%s netParams=%s ' % (nrnCommand, script, cfgSavePath, netParamsSavePath)
            else:
                command = '%s -n %d %s -python -mpi %s simConfig=%s netParams=%s ' % (
                    mpiCommand,
                    numproc,
                    nrnCommand,
                    script,
                    cfgSavePath,
                    netParamsSavePath,
                )

            # ----------------------------------------------------------------------
            # run on local machine with <nodes*coresPerNode> cores
            # ----------------------------------------------------------------------
            if type == 'mpi_direct':
                #executer = '/bin/bash'
                executer = executor
                jobString = jobStringMPIDirect(custom, folder, command)
            # ----------------------------------------------------------------------
            # Create script to run on HPC through slurm
            # ----------------------------------------------------------------------
            elif type == 'hpc_slurm':
                executer = 'sbatch'
                jobString = jobStringHPCSlurm(
                    jobName,
                    allocation,
                    walltime,
                    nodes,
                    coresPerNode,
                    jobPath,
                    email,
                    reservation,
                    custom,
                    folder,
                    command,
                )

def configure_command(preset_configs, **kwargs):
    ''' Build mpi command'''
    from datetime import datetime
    import os
    
    # Check that run_path is not None
    run_path = kwargs.get('run_path', None)
    assert run_path is not None, 'run_path must be specified'
    
    #get preset config
    interactive_node = preset_configs.get('interactive_node', False)
    sbatch = preset_configs.get('sbatch', False)
    login_node = preset_configs.get('login node', False)
    local = preset_configs.get('local', False)
    
    # select from preset configs
    if interactive_node:    
        # User arguments
        #shifter_command = 'shifter --image=adammwea/netsims_docker:v1'
        shifter_command = 'shifter --image=adammwea/netpyneshifter:v5'
        init_script = 'modules/simulation_config/init.py'
        init_script = os.path.abspath(init_script)
        

        # Detect SLURM environment or use defaults
        cores_per_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))  # Default to 1 core per node if not in SLURM
        number_of_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))  # Default to 1 node if not in SLURM
        ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", cores_per_node))  # Default tasks per node = cores
        total_tasks = number_of_nodes * ntasks_per_node
        #ntasks_per_node = 256
        tasks_per_node = int(ntasks_per_node/2) #use half the tasks per node to get a single simulation on a single socket
        
        #something I'm trying - not sure if it will work
        pop_size = kwargs.get('pop_size', 100)
        cores_per_srun = cores_per_node//pop_size
        tasks_per_srun = cores_per_srun #run the whole generation in parallel
        if tasks_per_node < 1: tasks_per_node = 1
        
        command_kwargs = {
            'type': 'mpi_direct',
            'script': init_script,
            #'mpiCommand': 'srun',
            #'mpiCommand': f'srun',
            'mpiCommand': f'srun -N 1', 
            'nrnCommand': f'--cpus-per-task=1 --hint=nomultithread --cpu-bind=cores {shifter_command} nrniv', #get single simulation on a single socket
            #'coresPerNode': cores_per_node,
            'coresPerNode': tasks_per_srun,
            'reservation': None,
            'skip': True, #skip set to true avoids repeating the same simulations ... need to verify if it skips whole generations or just individual simulations
            #'nodes': number_of_nodes,
            'nodes': 1,
        }
        
        # Extract command kwargs
        mpiCommand = command_kwargs['mpiCommand']
        #numproc = command_kwargs['coresPerNode']*command_kwargs['nodes']
        numproc = command_kwargs['coresPerNode']
        nrnCommand = command_kwargs['nrnCommand']
        script = command_kwargs['script']
        
        # Build output folder and fitness target script paths
        mock_gen_dir = 'gen_i'
        mock_cfgsavepath = os.path.join(run_path, mock_gen_dir, f'{mock_gen_dir}_cand_j_cfg.json')
        mock_netparamssavepath = os.path.join(run_path, mock_gen_dir, f'{mock_gen_dir}_cand_j_netParams.json')
        cfgSavePath = mock_cfgsavepath
        netParamsSavePath = mock_netparamssavepath
        
        #this directly mimics the code in netpyne/batch/utils.py
        # ----------------------------------------------------------------------
        # MPI job command
        # ----------------------------------------------------------------------
        if mpiCommand == '':
            command = '%s %s simConfig=%s netParams=%s ' % (nrnCommand, script, cfgSavePath, netParamsSavePath)
        else:
            command = '%s -n %d %s -python -mpi %s simConfig=%s netParams=%s ' % (
                mpiCommand,
                numproc,
                nrnCommand,
                script,
                cfgSavePath,
                netParamsSavePath,
            )
    elif sbatch:
        implemented=False
        assert implemented, 'sbatch not implemented yet'
    elif login_node:
        # User arguments
        #shifter_command = 'shifter --image=adammwea/netsims_docker:v1'
        shifter_command = 'shifter --image=adammwea/netpyneshifter:v5'
        init_script = 'modules/simulation_config/init.py'
        init_script = os.path.abspath(init_script)
        

        # Detect SLURM environment or use defaults
        cores_per_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))  # Default to 1 core per node if not in SLURM
        number_of_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))  # Default to 1 node if not in SLURM
        ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", cores_per_node))  # Default tasks per node = cores
        total_tasks = number_of_nodes * ntasks_per_node
        ntasks_per_node = 256
        tasks_per_node = int(ntasks_per_node/2) #use half the tasks per node to get a single simulation on a single socket
        
        command_kwargs = {
            'type': 'mpi_direct',
            'script': init_script,
            #'mpiCommand': 'srun',
            #'mpiCommand': f'srun',
            'mpiCommand': f'', 
            'nrnCommand': f'{shifter_command} nrniv', #get single simulation on a single socket
            #'coresPerNode': cores_per_node,
            #'coresPerNode': tasks_per_node,
            #'reservation': None,
            #'skip': USER_skip,
            #'nodes': number_of_nodes,
            'nodes': 1,
        }
        
        # Extract command kwargs
        mpiCommand = command_kwargs['mpiCommand']
        #numproc = command_kwargs['coresPerNode']*command_kwargs['nodes']
        nrnCommand = command_kwargs['nrnCommand']
        script = command_kwargs['script']
        
        # Build output folder and fitness target script paths
        mock_gen_dir = 'gen_i'
        mock_cfgsavepath = os.path.join(run_path, mock_gen_dir, f'{mock_gen_dir}_cand_j_cfg.json')
        mock_netparamssavepath = os.path.join(run_path, mock_gen_dir, f'{mock_gen_dir}_cand_j_netParams.json')
        cfgSavePath = mock_cfgsavepath
        netParamsSavePath = mock_netparamssavepath
        
        #this directly mimics the code in netpyne/batch/utils.py
        # ----------------------------------------------------------------------
        # MPI job command
        # ----------------------------------------------------------------------
        if mpiCommand == '':
            command = '%s %s simConfig=%s netParams=%s ' % (nrnCommand, script, cfgSavePath, netParamsSavePath)
        else:
            command = '%s -n %d %s -python -mpi %s simConfig=%s netParams=%s ' % (
                mpiCommand,
                numproc,
                nrnCommand,
                script,
                cfgSavePath,
                netParamsSavePath,
            )
    elif local:
        implemented=False
        assert implemented, 'local not implemented yet'
    else: 
        raise ValueError('No preset config selected')    
    
    #consolidate command kwargs
    kwargs.update(command_kwargs)
        
    # print/return command
    mock_command = command
    #print(f'Example Command:\n{mock_command}')    
    return mock_command, kwargs

'''Main Code'''
from batch_run_helper import setup_environment_wrapper, add_output_path_to_kwargs, add_run_path_to_kwargs, init_output_paths_as_needed
from batch_run_helper import init_batch_cfg, batchRun
from parse_kwargs import configure_global_user_vars, save_config_to_file
from batch_run_helper import pre_run_checks
import sys
if __name__ == '__main__':    
    
    '''Initialize'''    
    # Setup environment
    print('Setting up environment...')
    sys_path = setup_environment_wrapper(verbose = False) #output sys path for double checking that the environment is set up correctly
    kwargs = init_user_kwargs()
    kwargs = add_output_path_to_kwargs(**kwargs)
    kwargs = add_run_path_to_kwargs(**kwargs) #overwrite_run and continue_run are set to False by default
    
    #get mpi command kwargs and print example command to check before running
    print('Configuring command...')
    #presets available in configure_command, only one should be set to True
    preset_configs={
        'interactive_node': True, #salloc --nodes=4 --ntasks-per-node=256 -C cpu -q interactive -t 04:00:00 --image=adammwea/netpyneshifter:v5
        
        #'sbatch': False,       #sbatch --nodes=n --ntasks-per-node=m -C cpu -q interactive -t 04:00:00 --image=adammwea/netpyneshifter:v5; 
                                # where n and m are determined by SLURM environment variables; 
                                # m should be half of the number of tasks per node
        
        #'login node': True,
        #'local': False,
    }
    assert sum([preset_configs[key] for key in preset_configs]) == 1, 'Exactly one preset config must be set to True'
    example_command, kwargs = configure_command(preset_configs, **kwargs)
    kwargs.update({'example_command': example_command})
    
    #init all args
    USER_vars = configure_global_user_vars(**kwargs) #parse kwargs and import global user variables
    save_config_to_file(USER_vars, **kwargs) #must precede import of temp_user_args
    from temp_user_args import * # Import user arguments from temp file created by parse_kwargs.main(**kwargs)
    #TODO: probably need to move this ^ to the very top of the script so that bash flags supercede the kwargs
    
    # Initialize batch config
    batch_cfg = init_batch_cfg(USER_vars, **kwargs)
    
    '''last checks before running'''
    # Initialize batch config
    print('Initiating batch_config...')        
    kwargs.update({'batch_cfg': batch_cfg})
    
    # Perform pre-run checks
    print('Performing pre-run checks...')
    pre_run_checks(USER_vars, **kwargs)
       
    # Wait for user input before running
    try:
        user_input = input('Press Enter to run the command, or "c" to cancel: ')
        if user_input.lower() == 'c':
            print('Command execution cancelled.')
            sys.exit()
    except KeyboardInterrupt:
        print('\nCommand execution cancelled by user.')
        sys.exit()
        
    '''Run'''        
    # Commit to running the command, begin creating output paths
    init_output_paths_as_needed(**kwargs)
    cfg =batch_cfg
    batchRun(batch_config = cfg, **kwargs)    
    print(f'Batch run completed')