#from _3_rerun_simulation_of_interest import *
global PROGRESS_SLIDES_PATH, SIMULATION_RUN_PATH, REFERENCE_DATA_NPY, CONVOLUTION_PARAMS, DEBUG_MODE
from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import *
from fitness_helper import *
from fitting.calculate_fitness_vCurrent import fitnessFunc
from time import time
from analysis_helper import *
from netpyne import sim
add_repo_root_to_sys_path()
from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep._4_rerun_simulation_of_interest import reprocess_simulation
from concurrent.futures import ProcessPoolExecutor

# Paths and constants
SIMULATION_RUN_PATH = (
    #'/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_3/gen_3_cand_9_data.json')
    #'/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_17_data.json'
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_2/gen_2_cand_1_data.json'
    )
REFERENCE_DATA_NPY = (
    '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/fitting/experimental_data_features/network_metrics/CDKL5-E6D_T2_C1_05212024_240611_M06844_Network_000076_network_metrics_well000.npy')
CONVOLUTION_PARAMS = (
    "/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/fitting/convolution_params/241202_convolution_params.py")
cfg_script_path = 'batching/241211_cfg.py'
param_script_path = 'fitting/evol_parameter_spaces/241202_adjusted_evol_params.py'
target_script_path = 'fitting/experimental_data_features/fitness_args_20241205_022033.py'

sim_data_path = SIMULATION_RUN_PATH
sim_cfg_path = SIMULATION_RUN_PATH.replace('_data', '_cfg')

# Duration of the simulation in seconds
duration_seconds = 30

# Import parameter space
current_dir = os.path.dirname(os.path.abspath(__file__))
param_script_path = os.path.join(current_dir, param_script_path)
param_space = import_module_from_path(param_script_path).params
sim.loadSimCfg(sim_cfg_path)
print(f'simLabel: {sim.cfg.simLabel}')
sim_params = sim.cfg.__dict__.copy()

def rerun_simulation(sim_data_path, reference_data_path, conv_params_path, target_script_path, duration_seconds=1, save_data=True, overwrite_cfgs=False):
    privileged_print(f'Running original simulation {sim_data_path}')
    reprocess_simulation(
        sim_data_path, reference_data_path, conv_params_path,
        target_script_path=target_script_path, duration_seconds=duration_seconds,
        save_data=False, overwrite_cfgs=overwrite_cfgs
    )

# Top-level functions for parallel execution
def reduced_permutation(param, sim_param, sim_cfg_path, sim_data_path, reference_data_path, conv_params_path, target_script_path, duration_seconds, save_data, overwrite_cfgs, sim_params):
    permuted_cfg_path = sim_cfg_path.replace('_cfg', f'_reduced_{param}_cfg')
    permuted_cfg_copy = {'simConfig': sim.cfg.__dict__.copy()}
    permuted_cfg_copy['simConfig'][f'{param}'] = sim_params[sim_param] * 0.2
    permuted_cfg_copy['simConfig']['simLabel'] = sim_params['simLabel'] + f'_reduced_{param}'

    with open(permuted_cfg_path, 'w') as f:
        json.dump(permuted_cfg_copy, f, indent=4)

    privileged_print(f'Running permutation {sim_params["simLabel"]}_reduced_{param}')
    reprocess_simulation(
        sim_data_path, reference_data_path, conv_params_path,
        target_script_path=target_script_path, duration_seconds=duration_seconds,
        permuted_cfg_path=permuted_cfg_path, save_data=save_data, overwrite_cfgs=overwrite_cfgs
    )
def increased_permutation(param, sim_param, sim_cfg_path, sim_data_path, reference_data_path, conv_params_path, target_script_path, duration_seconds, save_data, overwrite_cfgs, sim_params):
    permuted_cfg_path = sim_cfg_path.replace('_cfg', f'_increased_{param}_cfg')
    permuted_cfg_copy = {'simConfig': sim.cfg.__dict__.copy()}
    permuted_cfg_copy['simConfig'][f'{param}'] = sim_params[sim_param] * 1.2
    permuted_cfg_copy['simConfig']['simLabel'] = sim_params['simLabel'] + f'_increased_{param}'

    with open(permuted_cfg_path, 'w') as f:
        json.dump(permuted_cfg_copy, f, indent=4)

    privileged_print(f'Running permutation {sim_params["simLabel"]}_increased_{param}')
    reprocess_simulation(
        sim_data_path, reference_data_path, conv_params_path,
        target_script_path=target_script_path, duration_seconds=duration_seconds,
        permuted_cfg_path=permuted_cfg_path, save_data=save_data, overwrite_cfgs=overwrite_cfgs
    )
def run_sensitivity_analysis(
    sim_data_path,
    reference_data_path,
    conv_params_path,
    target_script_path,
    duration_seconds=1,
    save_data=True,
    overwrite_cfgs=False,
    parallel=False):
    
    # Collect tasks
    tasks = []
    #sim_cfg_path = os.path.join(sim_data_path, 'sim_cfg.json')  # Example path
    sim_cfg_path = sim_data_path.replace('_data', '_cfg')

    #first task, run the original simulation
    tasks.append((rerun_simulation, sim_data_path, reference_data_path, conv_params_path, target_script_path, duration_seconds, save_data, overwrite_cfgs))
        
    for param in param_space:
        for sim_param in sim_params:
            if param in sim_param and isinstance(param_space[param], list):
                tasks.append((reduced_permutation, param, sim_param, sim_cfg_path, sim_data_path, reference_data_path, conv_params_path, target_script_path, duration_seconds, save_data, overwrite_cfgs, sim_params))
                tasks.append((increased_permutation, param, sim_param, sim_cfg_path, sim_data_path, reference_data_path, conv_params_path, target_script_path, duration_seconds, save_data, overwrite_cfgs, sim_params))

    if parallel:
        # Determine the number of workers
        num_workers = min(len(tasks), os.cpu_count()//4)
        print(f"Running in parallel with {num_workers} workers...")

        # Run tasks in a process pool
        #num_workers = 1 # For debugging
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(task[0], *task[1:]) for task in tasks]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Task failed with error: {e}")
    else:
        print("Running sequentially...")
        for task in tasks:
            try:
                task[0](*task[1:])
            except Exception as e:
                print(f"Task failed with error: {e}")

# Example invocation
run_sensitivity_analysis(
    sim_data_path=sim_data_path,
    reference_data_path=REFERENCE_DATA_NPY,
    conv_params_path=CONVOLUTION_PARAMS,
    target_script_path=target_script_path,
    duration_seconds=duration_seconds,
    save_data=True,
    overwrite_cfgs=True,
    parallel=True  # Set to False for sequential execution
)
