from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep._4_rerun_simulation_of_interest import *

# ===================================================================================================
SIMULATION_RUN_PATH = (    
   '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_3/gen_3_cand_9_data.json'
   #'/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_3/gen_3_cand_173_data.json'
    # '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_11_data.json'
    )
REFERENCE_DATA_NPY = (
    '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/fitting/experimental_data_features/network_metrics/CDKL5-E6D_T2_C1_05212024_240611_M06844_Network_000076_network_metrics_well000.npy'
    )
CONVOLUTION_PARAMS = (
    #"workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/convolution_params/241202_convolution_params.py"
    "/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/fitting/convolution_params/241202_convolution_params.py"
    )
cfg_script_path = 'batching/241211_cfg.py'
param_script_path = 'fitting/evol_parameter_spaces/241202_adjusted_evol_params.py'
target_script_path = 'fitting/experimental_data_features/fitness_args_20241205_022033.py'
sim_data_path = SIMULATION_RUN_PATH
sim_cfg_path = SIMULATION_RUN_PATH.replace('_data', '_cfg')
netParams_data_path = SIMULATION_RUN_PATH.replace('_data', '_netParams')

# ===================================================================================================

# import evol parameter space from path
# permute parameter space
from netpyne import sim
script_path = os.path.dirname(os.path.abspath(__file__))
param_script_path = os.path.join(script_path, param_script_path)
param_space = import_module_from_path(param_script_path).params
sim_data_path = SIMULATION_RUN_PATH
#output = sim.load(sim_data_path, output=True)
sim.loadSimCfg(sim_cfg_path)
sim_params = sim.cfg.__dict__.copy()

# modulate and run one param at a time
sim_cfg_path = sim_cfg_path
reference_data_path = REFERENCE_DATA_NPY
conv_params_path = CONVOLUTION_PARAMS
target_script_path = target_script_path
def run_sensitivity_analysis(
    sim_data_path,
    reference_data_path,
    conv_params_path,
    target_script_path,
    duration_seconds = 1,
    permuted_cfg_path=None,
    save_data=True,
    overwrite_cfgs=False):
    
    for param in param_space:
        for sim_param in sim_params:
            if param in sim_param:            
                # if param is a list
                if isinstance(param_space[param], list): #exclude constant params
                    #param_space[param] = param_space[param][0]
                    def reduced_permutation():
                        #reduced
                        #sim.cfg[f'{param}'] = sim_params[sim_param] *0.2 # 20% reduction
                        #save cfg as json at the same path as the sim_data_path
                        permuted_cfg_path = sim_cfg_path.replace('_cfg', f'_reduced_{param}_cfg')
                        permuted_cfg_copy = {
                            'simConfig': sim.cfg.__dict__.copy(),
                        }
                        permuted_cfg_copy['simConfig'][f'{param}'] = sim_params[sim_param] *0.2 # 20% reduction
                        #sim.saveJSON(permuted_cfg_path)
                        #current_sim_label = sim_param
                        permuted_cfg_copy['simConfig'][f'simLabel'] = sim_params['simLabel'] + f'_reduced_{param}' #apparently only need to update simLabel to change filename
                        
                        # permuted_cfg_copy['simConfig'][f'filename'] = permuted_cfg_copy['simConfig']['filename'] + f'_reduced_{param}'
                        with open(permuted_cfg_path, 'w') as f:
                            json.dump(permuted_cfg_copy, f, indent=4)
                            
                        # validate updated params
                        #print(f'original {sim_param}: {sim_params[sim_param]}')
                        print(f'updated {sim_param}: {permuted_cfg_copy["simConfig"][sim_param]}')
                        #print(f'original simLabel: {sim_params["simLabel"]}')
                        print(f'updated simLabel: {permuted_cfg_copy["simConfig"]["simLabel"]}')
                        
                        # sim.loadSimCfg(permuted_cfg_path)
                        # sim.setSimCfg(sim.cfg)
                        
                        reprocess_simulation(
                            sim_data_path, 
                            reference_data_path,
                            conv_params_path,
                            target_script_path = target_script_path,
                            duration_seconds = duration_seconds, 
                            permuted_cfg_path=permuted_cfg_path, 
                            save_data=save_data, 
                            overwrite_cfgs=overwrite_cfgs) #only overwrite if permute cfg already exists
                    try: reduced_permutation()
                    except: 
                        print(f'reduced permutation failed')
                        pass
                    
                    def increased_permutation():
                        #increased
                        permuted_cfg_path = sim_cfg_path.replace('_cfg', f'_increased_{param}_cfg')
                        permuted_cfg_copy = {
                            'simConfig': sim.cfg.__dict__.copy(),
                        }
                        permuted_cfg_copy['simConfig'][f'{param}'] = sim_params[sim_param] *1.2
                        permuted_cfg_copy['simConfig'][f'simLabel'] = sim_params['simLabel'] + f'_increased_{param}' #apparently only need to update simLabel to change filename
                        
                        with open(permuted_cfg_path, 'w') as f:
                            json.dump(permuted_cfg_copy, f, indent=4)
                            
                        # validate updated params
                        #print(f'original {sim_param}: {sim_params[sim_param]}')
                        print(f'updated {sim_param}: {permuted_cfg_copy["simConfig"][sim_param]}')
                        #print(f'original simLabel: {sim_params["simLabel"]}')
                        print(f'updated simLabel: {permuted_cfg_copy["simConfig"]["simLabel"]}')
                        
                        # sim.loadSimCfg(permuted_cfg_path)
                        # sim.setSimCfg(sim.cfg)
                        
                        reprocess_simulation(
                            sim_data_path, 
                            reference_data_path,
                            conv_params_path,
                            target_script_path = target_script_path,
                            duration_seconds = duration_seconds, 
                            permuted_cfg_path=permuted_cfg_path, 
                            save_data=save_data, 
                            overwrite_cfgs=overwrite_cfgs)
                    try: increased_permutation()
                    except: 
                        print(f'increased permutation failed')
                        pass                   
run_sensitivity_analysis(
    sim_data_path,
    reference_data_path,
    conv_params_path,
    target_script_path,
    duration_seconds = 30,
    permuted_cfg_path=None,
    save_data=True,
    overwrite_cfgs=True) #TODO: need to rename/fix this parameter

                