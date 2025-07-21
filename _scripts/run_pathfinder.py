# modular run script for sensitivity analysis

# imports =====================================================================

#from RBS_network_models.utils.sensitivity_analysis import run_sensitivity_analysis
from RBS_network_models.utils.heatmaps import plot_pathfinder
#from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.conv_params import conv_params, mega_params
#from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.evol_params import params

# arguments ============================================================

# src paths
#ORIGIN = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_FxHET/manual_tunning/27jun2025_CDKL5_DIV21_WT/nb2/tau2_inh_1.0_pIE_0.01/tau2_inh_down_pIE_down_data.pkl' # Original simulation data path, starting point for the sensitivity analysis
INPUT_DIR = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_FxHET/250715/sensitivity_analysis/run0000' # output directory for simulated data, corresponding to the project
# INIT_PATH = '/global/homes/a/adammwea/dev/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/init.py'
# CONV_PATH = '/global/homes/a/adammwea/dev/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/conv_params.py'
EVOL_PATH = '/global/homes/a/adammwea/dev/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/evol_params.py'
OUTPUT_DIR = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_FxHET/250715/sensitivity_analysis/run0000/pathfinders' # output directory for the sensitivity analysis results, corresponding to the project
CACHE_DIR = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_FxHET/250715/sensitivity_analysis/run0000/.cache' # cache directory for the sensitivity analysis results, corresponding to the project
PERM_DIR = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_FxHET/250715/sensitivity_analysis/run0000/permutations' # directory for the permutation files, corresponding to the project

kwargs = {
    
    # init
    'input_dir': INPUT_DIR,  # output directory for the simulation data, if not specified, it will be derived from the simulation data path
    'output_dir': OUTPUT_DIR,  # output directory for the sensitivity analysis results, if not specified, it will be derived from the simulation data path
    'cache_dir': CACHE_DIR,  # cache directory for the sensitivity analysis results, if not specified, it will be derived from the simulation data path
    'permutation_dir': PERM_DIR,  # directory for the permutation files, if not specified, it will be derived from the simulation data path
    'num_workers': 16, # number of workers for parallel processing, if not specified, it will be derived from the system's CPU count
    #'num_workers': 256,  # regular node
    
    # permute cfg and netParams
    'params': EVOL_PATH,  # evolution parameters for the simulation data, if not specified, it will be derived from the simulation data path
    'levels': 10,  # number of sensitivity analysis levels, subsequently heatmap levels.
    
    # simulation runtime options
    #'overwrite_cache': True,  # if True, overwrite the cache directory, if False, use the existing cache directory
    #'parallel': True,  # if True, run the simulations in parallel using multiprocessing or MPI, if False, run the simulations sequentially
    'cache': False,  # if True, cache data_grid to avoid recompiling the data grid, if False, recompute the data grid every time
    # plotting options
    'print_values': True,  # if True, print the values of the parameters and metrics in the heatmap, if False, do not print the values
    
    # debug
    # 'debug_limited_heatmap': True,  # if True, limit the heatmap to a subset of the data for debugging purposes
    # 'limited_load_num': 5,  # number of files to load for debugging purposes, if not specified, it will be set to 10
    
    # queries for pathfinder
    'queries': {
        'spk_metrics.num_e_firing': {
            'greater_than': 0,  # only consider values that change more than this percentage over the original value
        },
        'spk_metrics.E_spikes.mean': {
            'greater_than': 0,  # only consider values that change more than this percentage over the original value
        },
        'hyperburst_metrics.baseline' : {
            'abs_less_than': 5,  # only consider values that change less than this absolute value over the original value
        },
        'hyperburst_metrics.burst_metrics.burst_amp.mean' : {
            'greater_than': 0,  # only consider values that change more than this percentage over the original value
        },
        # 'hyperburst_metrics.burst_metrics.burst_duration.mean' : {
        #     'less_than': 0,  # only consider values that change more than this percentage over the original value
        # },
        'hyperburst_metrics.burst_metrics.burst_rate' : {
            'less_than': 0,  # only consider values that change more than this percentage over the original value
        },
    },
}
    
if __name__ == "__main__":

    # run the sensitivity analysis
    #run_sensitivity_analysis(**kwargs)  # run the sensitivity analysis with the specified parameters
    #plot_heatmaps(**kwargs)  # plot the heatmaps for the sensitivity analysis results
    plot_pathfinder(**kwargs)  # plot the pathfinder for the sensitivity analysis results

    # just a place for debug to stop
    print()