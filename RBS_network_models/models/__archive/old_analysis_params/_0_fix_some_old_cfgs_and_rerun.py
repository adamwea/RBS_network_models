global PROGRESS_SLIDES_PATH, SIMULATION_RUN_PATH, REFERENCE_DATA_NPY, CONVOLUTION_PARAMS, DEBUG_MODE
#from .. import sim_helper as sh
# ===================================================================================================
sim_data_path = ('/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/'
                 'optimization_projects/CDKL5_DIV21/'
                 'scripts/_1_sims/testing/test_run_3_data.pkl')
evol_params_path = ('/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/'
                    'optimization_projects/CDKL5_DIV21/'
                    'scripts/_3_analysis/evol_parameter_spaces/adjusted_evol_params_241202.py')
sensitivity_analysis_output_path =('/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/'
                                   'optimization_projects/CDKL5_DIV21/'
                                   'scripts/_1_sims/testing_sensitivity_analysis')
# Paths and constants
SIMULATION_RUN_PATH = ('/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_2/gen_2_cand_1_data.pkl')
REFERENCE_DATA_NPY = ('/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/fitting/experimental_data_features/network_metrics/CDKL5-E6D_T2_C1_05212024_240611_M06844_Network_000076_network_metrics_well000.npy')
CONVOLUTION_PARAMS = ("/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/fitting/convolution_params/241202_convolution_params.py")
cfg_script_path = 'batching/241211_cfg.py'
param_script_path = 'fitting/evol_parameter_spaces/241202_adjusted_evol_params.py'
target_script_path = 'fitting/experimental_data_features/fitness_args_20241205_022033.py'
sim_data_path = SIMULATION_RUN_PATH
# ===================================================================================================
# run sensitivity analysis
#run_sensitivity_analysis(sim_data_path, evol_params_path, sensitivity_analysis_output_path, option='sequential')
run_sensitivity_analysis(
    sim_data_path, 
    evol_params_path, 
    sensitivity_analysis_output_path,
    duration_seconds=1, 
    #option='parallel'                          #default is 'serial', so just uncomment out if parallel is desired
    #sim_cfg_path=SIMULATION_RUN_PATH,
    #reference_data_path=REFERENCE_DATA_NPY,
    #conv_params_path=CONVOLUTION_PARAMS,
    #target_script_path=target_script_path,
    )
print('Sensitivity analysis successfully ran!')