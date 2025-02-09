#global PROGRESS_SLIDES_PATH, SIMULATION_RUN_PATH, REFERENCE_DATA_NPY, CONVOLUTION_PARAMS, DEBUG_MODE
#from CDKL5_DIV21.utils.sim_helper import *
#from DIV21.utils.sensitivity_analysis import plot_sensitivity_grid_plots as plot_sensitivity_grid_plots
from RBS_network_models.developing.utils.sensitivity_analysis import plot_sensitivity_grid_plots
import os
# ===================================================================================================
sim_data_path = (
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/'
    'tests/outputs/test_run_a_simulation/test_run_5_data.pkl')
output_dir =(
    #original testing
    # '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/'
    # 'CDKL5_DIV21/tests/outputs/test_sensitivity_analysis'
    
    #level testing - 2025-01-10 10:14:16
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/'
    'tests/outputs/test_sensitivity_analysis'
    )
reference_data_path = (
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/CDKL5/'
    'DIV21/src/experimental_reference_data/CDKL5-E6D_T2_C1_05212024_240611_M06844_Network_000076_network_metrics_well000.npy')
# ===================================================================================================
#assert paths are correct
assert os.path.exists(sim_data_path), f"sim_data_path does not exist: {sim_data_path}"
assert os.path.exists(reference_data_path), f"reference_data_path does not exist: {reference_data_path}"
assert os.path.exists(output_dir), f"output_dir does not exist: {output_dir}"
# ===================================================================================================

# NOTE: run _4_test_sensitivity_analysis.py before running this script

sensitvity_analysis_output_dir = output_dir
plot_sensitivity_grid_plots(
    sim_data_path,
    sensitvity_analysis_output_dir
    )