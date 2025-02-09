
''' 

batchRun_evol_srun_direct.py - Batch run script for evolutionary optimization using srun direct MPI command

using this script for early development and testing of the batchRun function.

'''
from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import *
add_repo_root_to_sys_path()
from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.batch_helper import *
#import pdb
#pdb.set_trace()

#==============================================================================

def init_user_kwargs(): # initialize user kwargs here, then pass them to the main function
    '''Initialize user kwargs
    note: if new kwargs are added, they must be added to the parse_kwargs.main() function in the parse_kwargs.py script
    '''
    def get_output_folder():
        """Returns the output folder and run label."""
        output_folder_name = 'workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21'
        #run_label = 'CDKL5_seeded_evol_2' probably not much data here but I didnt want to delete it
        #run_label = 'CDKL5_seeded_evol_3'
        #run_label = 'CDKL5_seeded_evol_4'
        #run_label = 'CDKL5_seeded_evol_5'
        #run_label = 'CDKL5_seeded_evol_6'
        #run_label = 'CDKL5_seeded_evol_7_fixing_baseline'
        #run_label = 'CDKL5_seeded_evol_8_reloaded'
        run_label = 'CDKL5_new_netparams_test'
        #run_label = 'test'
        #run_label='test2'
        return output_folder_name, run_label

    def get_simulation_params():
        """Returns simulation parameters."""
        return {
            'max_wait_time_minutes': 30,  # Maximum time to wait for new candidates
            'time_sleep': 1  # Time to sleep between checks in seconds
        }

    def get_fitness_and_seed_scripts():
        """Returns paths for fitness and seed scripts."""
        fitness_target_script = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/_config/experimental_data_features/fitness_args_20241205_022033.py'
        seed_script = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/_config/seed_selection_scripts/241126_Run2_improved_netparams.py'
        assert os.path.exists(seed_script), f'seed_script not found: {seed_script}'
        return fitness_target_script, seed_script

    def load_seed_paths(seed_script):
        """Loads seed paths from the seed script."""
        seed_module = import_module_from_path(seed_script)
        return seed_module.seed_paths

    def get_param_space_script():
        """Returns the path to the parameter space script."""
        param_space_script = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/_config/evol_parameter_spaces/241202_adjusted_evol_params.py'
        assert os.path.exists(param_space_script), f'param_space_script not found: {param_space_script}'
        return param_space_script

    def load_param_space(param_space_script):
        """Loads parameter space from the parameter space script."""
        param_space_module = import_module_from_path(param_space_script)
        return param_space_module.params

    """Builds and returns the user kwargs dictionary."""
    # Load primary configurations
    output_folder_name, run_label = get_output_folder()
    sim_params = get_simulation_params()
    fitness_target_script, seed_script = get_fitness_and_seed_scripts()
    seed_paths = load_seed_paths(seed_script)
    param_space_script = get_param_space_script()
    param_space = load_param_space(param_space_script)

    # Construct kwargs
    kwargs = {
       
        # Netpyne kwargs
        'duration': 30,  # Duration of the simulation, in seconds
        'pop_size': 192,
        'num_elites': 75,
        
        # test params
        # 'duration': 1,
        # 'pop_size': 4,
        # 'num_elites': 1,
        
        'max_generations': 100,
        'time_sleep': sim_params['time_sleep'],
        'maxiter_wait': sim_params['max_wait_time_minutes'] * 60 / sim_params['time_sleep'],
        'batch_type': 'evol',
        'cfg_file': '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/batch_scripts/cfg.py',
        'netParamsFile': '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/batch_scripts/netParams.py',
        'init_script': '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/batch_scripts/init.py',
        
        # Custom kwargs
        'continue_run': True,
        'overwrite': False,
        'label': run_label,
        'output_path': os.path.abspath(output_folder_name),
        'output_folder_name': output_folder_name,
        'fitness_target_script': fitness_target_script,
        'outside_of_repo': True,
        'recalculate_fitness': False,
        'seed_evol': True,
        'seed_paths': seed_paths,
        'param_space': param_space,
    }
    return kwargs    
    
    # def init_user_kwargs(): # initialize user kwargs here, then pass them to the main function
    #     '''Initialize user kwargs
    #     note: if new kwargs are added, they must be added to the parse_kwargs.main() function in the parse_kwargs.py script
    #     '''
        
    #     # User arguments
    #     output_folder_name = 'workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21' # relative to home directory
    #     #run_label = f'improved_netparams'  # subfolder for this run will be created in output_folder_path
    #     run_label = f'CDKL5_seeded_evol_2'  # subfolder for this run will be created in output_folder_path
    #     batch_type = 'evol'
    #     max_wait_time_minutes = 60 #minutes - maximum time to wait for a generation or stalled simulation to finish new candidates
    #     time_sleep = 1 #seconds - time to sleep between checking for new candidates
        
    #     # Fitness target scripts
    #     #fitness_target_script = '/pscratch/sd/a/adammwea/RBS_network_simulations/tunning_scripts/CDKL5-E6D_T2_C1_DIV21/derived_fitness_args/fitness_args_20241123-155335.py'
    #     #fitness_target_script = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_1_derive_features_from_experimental_data/derived_fitness_args/fitness_args_20241202_145331.py'
    #     #fitness_target_script = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_1_derive_features_from_experimental_data/derived_fitness_args/fitness_args_20241203_160213.py'
    #         #reran for update convolution params yeilding different target values
    #         # also fixed issue with baseline calculation
        
    #     # seed scripts
    #     #seed_script = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_3_plot_all_runs_and_select_seeds/_2_seed_review_241126_Run2_improved_netparams.py'
        
    #     #convolution params were changed to be more sensitive to the target values - so fitness targets were recalculated
    #     fitness_target_script = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/_config/experimental_data_features/fitness_args_20241205_022033.py'
    #     seed_script = "/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/_config/seed_selection_scripts/241126_Run2_improved_netparams.py"
    #     assert os.path.exists(seed_script), f'seed_script not found: {seed_script}'
        
    #     # read seed_script to get seed paths
    #     seed_paths = None
    #     seed_module = import_module_from_path(seed_script)
    #     seed_paths = seed_module.seed_paths
        
    #     # param_space_script
    #     #param_space_script = '/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/parameter_spaces/_241202_adjusted_evol_params.py'
    #     param_space_script = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/_config/evol_parameter_spaces/241202_adjusted_evol_params.py'
    #     assert os.path.exists(param_space_script), f'param_space_script not found: {param_space_script}'
        
    #     # read param_space_script to get param_space
    #     param_space = None
    #     param_space_module = import_module_from_path(param_space_script)
    #     param_space = param_space_module.params
        
    #     # user kwargs
    #     kwargs = {        
    #         # netpyne kwargs
    #         'duration': 1, # Duration of the simulation, in seconds
    #         #'pop_size': 128,
    #         'pop_size': 30,
    #         'num_elites': 15,
    #         'max_generations': 100,
    #         'time_sleep': time_sleep, #seconds per iteration
    #         'maxiter_wait': max_wait_time_minutes*60/time_sleep, #max wait time in iterations
    #         'batch_type': batch_type,
    #         'cfg_file': '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/batch_scripts/cfg.py',
    #         'netParamsFile': '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/batch_scripts/netParams.py',
    #         'init_script': '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/batch_scripts/init.py',
    #         # ... other netpyne kwargs can be added here and they should work as expected
            
    #         # custom kwargs
    #         'continue_run': True, #specified in main - idk which i prefer
    #         'overwrite': False,
    #         'label': run_label,
    #         'output_path': os.path.abspath(output_folder_name),
    #         'output_folder_name': output_folder_name, #configure_command will use this to build the output folder path
    #         'fitness_target_script': fitness_target_script,
    #         'outside_of_repo': True, #if True, the output folder will be created outside of the git repo
    #         'recalculate_fitness': False, #if True, when continuing a run, the fitness will be recalculated for all candidates
    #         'seed_evol': True, #if True, the initial population will be seeded with the best candidates curated from previous generations
    #         'seed_paths': seed_paths,
    #         'param_space': param_space,
    #         # ... other custom kwargs can be added here but then they must be added to the parse_kwargs.main() function in the parse_kwargs.py script
    #     }
        
    #     return kwargs 

def configure_command_and_preview(**kwargs): # configure command and preview it before running
    #presets available in configure_command, only one should be set to True
    preset_configs={
        'interactive_node': True, #salloc --nodes=4 --ntasks-per-node=256 -C cpu -q interactive -t 04:00:00 --image=adammwea/netpyneshifter:v5
        
        #'sbatch': False,       #sbatch --nodes=n --ntasks-per-node=m -C cpu -q interactive -t 04:00:00 --image=adammwea/netpyneshifter:v5; 
                                # where n and m are determined by SLURM environment variables; 
                                # m should be half of the number of tasks per node
        
        #'login_node': True,
        #'local': False,
    }
    kwargs.update({'preset_configs': preset_configs})
    assert sum([preset_configs[key] for key in preset_configs]) == 1, 'Exactly one preset config must be set to True'
    example_command, kwargs = configure_command(**kwargs)
    kwargs.update({'example_command': example_command})
    
    print(f'Example command: {example_command}')
    return kwargs

def main(**kwargs):
    '''Initialize''' 
    print('Initializing...')   
    kwargs = init_user_kwargs()
    #kwargs = add_output_path_to_kwargs(**kwargs)
    continue_latest_run = True
    kwargs.update({'continue_latest_run': continue_latest_run})
    kwargs = add_run_path_to_kwargs(**kwargs) #overwrite_run and continue_run are set to False by default
    
    print('Configuring command...')
    kwargs = configure_command_and_preview(**kwargs)
    
    print('Configuring arguments...')
    USER_vars = configure_global_user_vars(**kwargs) #parse kwargs and import global user variables
    
    print('MANUALLY ADDING SEEDS TO MAKE UP FOR FITNESS CALC ERROR')
    def manually_get_seeds(**kwargs):
        target_paths = [
            '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run5_CDKL5_seeded_evol_5',
            '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run6_CDKL5_seeded_evol_6',
        ]
        
        # os walk through target paths to get seed files ending in _cfg.json
        seed_paths = []
        for target_path in target_paths:
            for root, dirs, files in os.walk(target_path):
                for file in files:
                    if file.endswith('_cfg.json'):
                        fitness_json = re.sub('_cfg.json', '_fitness.json', file)
                        if os.path.exists(os.path.join(root, fitness_json)):
                            #fitness = import_module_from_path(os.path.join(root, fitness_json))
                            #load fitness w/ json
                            with open(os.path.join(root, fitness_json), 'r') as f:
                                fitness = json.load(f)
                            fitness = fitness['average_fitness']
                            if fitness < 1000:
                                seed_paths.append(os.path.join(root, file))
        for seed_path in seed_paths:
            seed_temp_dict = {
                'path': seed_path,
                'seed': True
            }
            kwargs['seed_paths'].append(seed_temp_dict)
        
        
        return kwargs
    kwargs = manually_get_seeds(**kwargs)
    #combine seed_paths with kwargs['seed_paths']
    
    
    print('Preparing batch_config...')
    batch_cfg = init_batch_cfg(USER_vars, **kwargs)     
    kwargs.update({'batch_cfg': batch_cfg})  
    
    # Perform pre-run checks
    print('Performing pre-run checks...')
    pre_run_checks(USER_vars, **kwargs)
        
    '''save all chosen kwargs to a file'''
    # saving arguments to file
    print('Saving arguments to temp_user_args.py...')
    #global_kwargs_path = save_global_kwargs_to_pickle(**kwargs)
    #USER_vars['global_kwargs_path'] = global_kwargs_path
    save_config_to_file(USER_vars, **kwargs) #must precede import of temp_user_args. access temp_user_args as needed in downstream scripts        
    save_batch_cfg_to_file(**kwargs)
    
    global global_kwargs
    global_kwargs = kwargs.copy()
        
    '''Run'''
    print('Running...')        
    # Commit to running the command, begin creating output paths
    init_output_paths_as_needed(**kwargs)
    cfg =batch_cfg
    batchRun(batch_config = cfg, **kwargs)    
    print(f'Batch run completed')

'''Main Code'''
if __name__ == '__main__':    
    main()

# ==============================================================================    
# Perlmutter Allocaiton Notes:
# ==============================================================================
#allocation:
# '''
# salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=4 --ntasks=128 --cpus-per-task=64 --image adammwea/netsims_docker:v1
# '''
'''
salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=4 --image adammwea/netsims_docker:v1
'''
#command (template):
'''
srun -N 1 -n 64 --cpus-per-task=2 --hint=multithread --cpu-bind=cores \
     shifter --image=adammwea/netsims_docker:v1 \
     nrniv -python -mpi -threads \
     /pscratch/sd/a/adammwea/modules/simulation_config/init.py \
     simConfig=/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run1_CDKL5_seeded_evol_2/gen_i/gen_i_cand_j_cfg.json \
     netParams=/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run1_CDKL5_seeded_evol_2/gen_i/gen_i_cand_j_netParams.json

'''

#TODO: lookinto nvprof? nvprof --profile-child-processes \
#TODO: gather data on efficiency here and begin tracking to improve.

