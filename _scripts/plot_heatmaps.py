# modular run script for sensitivity analysis

# imports =====================================================================

#from RBS_network_models.utils.sensitivity_analysis import run_sensitivity_analysis
from RBS_network_models.utils.heatmaps import plot_heatmaps
#from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.conv_params import conv_params, mega_params
#from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.evol_params import params

# arguments ============================================================

# src paths
#ORIGIN = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_FxHET/manual_tunning/27jun2025_CDKL5_DIV21_WT/nb2/tau2_inh_1.0_pIE_0.01/tau2_inh_down_pIE_down_data.pkl' # Original simulation data path, starting point for the sensitivity analysis
INPUT_DIR = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_FxHET/250715/sensitivity_analysis/run0000' # output directory for simulated data, corresponding to the project
# INIT_PATH = '/global/homes/a/adammwea/dev/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/init.py'
# CONV_PATH = '/global/homes/a/adammwea/dev/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/conv_params.py'
EVOL_PATH = '/global/homes/a/adammwea/dev/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/evol_params.py'
OUTPUT_DIR = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_FxHET/250715/sensitivity_analysis/run0000/heatmaps' # output directory for the sensitivity analysis results, corresponding to the project
CACHE_DIR = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_FxHET/250715/sensitivity_analysis/run0000/.cache' # cache directory for the sensitivity analysis results, corresponding to the project

kwargs = {
    
    # init
    'input_dir': INPUT_DIR,  # output directory for the simulation data, if not specified, it will be derived from the simulation data path
    'output_dir': OUTPUT_DIR,  # output directory for the sensitivity analysis results, if not specified, it will be derived from the simulation data path
    'cache_dir': CACHE_DIR,  # cache directory for the sensitivity analysis results, if not specified, it will be derived from the simulation data path
    #'num_workers': 16, # number of workers for parallel processing, if not specified, it will be derived from the system's CPU count
    'num_workers': 256,  # regular node
    
    # permute cfg and netParams
    'params': EVOL_PATH,  # evolution parameters for the simulation data, if not specified, it will be derived from the simulation data path
    'levels': 10,  # number of sensitivity analysis levels, subsequently heatmap levels.
    
    # simulation runtime options
    'overwrite_cache': True,  # if True, overwrite the cache directory, if False, use the existing cache directory
    'parallel': True,  # if True, run the simulations in parallel using multiprocessing or MPI, if False, run the simulations sequentially
    'cache': True,  # if True, cache data_grid to avoid recompiling the data grid, if False, recompute the data grid every time
    # plotting options
    'print_values': True,  # if True, print the values of the parameters and metrics in the heatmap, if False, do not print the values
    
    
    # debug
    # 'debug_limited_heatmap': True,  # if True, limit the heatmap to a subset of the data for debugging purposes
    # 'limited_load_num': 5,  # number of files to load for debugging purposes, if not specified, it will be set to 10
}

if __name__ == "__main__":

    # run the sensitivity analysis
    #run_sensitivity_analysis(**kwargs)  # run the sensitivity analysis with the specified parameters
    plot_heatmaps(**kwargs)  # plot the heatmaps for the sensitivity analysis results
    
    
    # just a place for debug to stop
    print()
    #print(f'Sensitivity analysis completed. Results saved to {SA_OUTPUT_DIR}')



import sys
sys.exit() # dont execute the rest of the script, this is just a template for the sensitivity analysis script

# helper functions ============================================================
def load_metrics_file(fpath):
    try:
        metrics = np.load(fpath, allow_pickle=True).item()
        sim_data_path = metrics.get('sim_data_path', None)
        if sim_data_path is None:
            sim_data_dir = os.path.dirname(os.path.dirname(fpath))
            # find a file in sim_data_dir that ends with '_data.pkl'
            sim_data_files = glob.glob(os.path.join(sim_data_dir, '*_data.pkl'))
            if sim_data_files:
                sim_data_path = sim_data_files[0]
            else:
                raise FileNotFoundError(f'No simulation data file found in {sim_data_dir} matching *_data.pkl')
            #sim_data_path = fpath.replace('_metrics.npy', '_data.pkl')
            metrics['sim_data_path'] = sim_data_path
        print(f'Loaded {fpath}')
        return metrics
    except Exception as e:
        print(f'Error loading {fpath}: {e}')
        return None

# arguments ==========

# Imports =====================================================================
#from RBS_network_models.sensitivity_analysis import run_sensitivity_analysis_v2
from RBS_network_models.sensitivity_analysis import run_simulation_permutations#, compute_permutation_network_metrics, plot_permutation_network_metrics
from RBS_network_models.sensitivity_analysis import compute_permutation_network_metrics
# from RBS_network_models.sensitivity_analysis import plot_permutation_network_metrics
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.conv_params import conv_params, mega_params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.evol_params import params
import glob
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_network_metrics_v2
from multiprocessing import Pool
import numpy as np
from RBS_network_models.sensitivity_analysis import plot_heat_maps

# reference code below
# analysis options
RUN_ANALYSIS = True # if True, run analysis on the simulation data
PLOT_ANALYSIS = True # if True, plot the analysis results
PLOT_GRID = True # if True, plot the grid of the analysis results
PLOT_HEATMAPS = True # if True, plot the heatmap of the analysis results
LEVELS = 10 # number of sensitivity analysis levels, subsequently heatmap levels
CONV_PARAMS = conv_params # conversion parameters for the simulation data
MEGA_PARAMS = mega_params # mega parameters for the simulation data
EVOL_PARAMS = params # evolution parameters for the simulation data

#allow max workers available on the system
import os
import multiprocessing
MAX_WORKERS = multiprocessing.cpu_count() # maximum number of workers for parallel processing

# runtime
#MAX_WORKERS = 24 # maximum number of workers for parallel processing

#RUN_PARALLEL = False # if True, run the sensitivity analysis in parallel
RUN_PARALLEL = True # if True, run the sensitivity analysis in parallel
DURATION_SECONDS = 140 # duration of the simulation in seconds
TRY_LOAD_SIM_DATA = True # if True, try to load the simulation data from the specified path
TRY_LOAD_NETWORK_DATA = True # if True, try to load the network data from the specified path, avoid recomputing metrics
TRY_LOAD_NETWORK_SUMMARY = True # if True, try to load the network plots, avoid re plotting
#DERIVE_OUTPUT_DIR = True # if True, derive the output directory from the simulation data path, even if output_dir is specified
DERIVE_OUTPUT_DIR = False # if True, derive the output directory from the simulation data path, even if output_dir is specified

# debug options
DEBUG_MODE = False # if True, run the simulation in debug mode - general and havent used this in a while, not sure if it works
DEBUG_LIMITED_PLOTTING = False # if True, limit the plotting to a subset of the data for debugging purposes
DEBUG_LIMITED_HEATMAP = False # if True, limit the heatmap to a subset of the data for debugging purposes

kwargs = {
    'sim_data_path': SIM_DATA_PATH,
    'output_dir': OUTPUT_DIR,  # output directory for the simulation data, if not specified, it will be derived from the simulation data path
    'derive_output_dir': DERIVE_OUTPUT_DIR,  # if True, derive the output directory from the simulation data path
    'reference_data_path': REFERENCE_DATA,
    'run_analysis': RUN_ANALYSIS,
    'plot_analysis': PLOT_ANALYSIS,
    'plot_grid': PLOT_GRID,
    'plot_heatmaps': PLOT_HEATMAPS,
    'levels': LEVELS,
    'conv_params': CONV_PARAMS,
    'mega_params': MEGA_PARAMS,
    'evol_params': EVOL_PARAMS,
    'max_workers': MAX_WORKERS,
    'run_parallel': RUN_PARALLEL,
    'duration_seconds': DURATION_SECONDS,
    'try_load_sim_data': TRY_LOAD_SIM_DATA,
    'try_load_network_data': TRY_LOAD_NETWORK_DATA,
    'try_load_network_summary': TRY_LOAD_NETWORK_SUMMARY,
    'debug_mode': DEBUG_MODE,
    'debug_limited_heatmap': DEBUG_LIMITED_HEATMAP,
    
    # runtime options - all set to false by default so comment out if not needed
    'run_sims': True,  # if True, run the simulations that constitute the sensitivity analysis
    'compute_metrics': True,  # if True, compute the network metrics for the simulation data
    'plot_metrics': True,  # if True, plot the network metrics for the simulation data
    'plot_heatmaps': True,  # if True, plot the heatmaps for
}

if __name__ == "__main__":
    #unpack runtime options
    run_sims = kwargs.get('run_sims', False)  # if True, run the simulations that constitute the sensitivity analysis
    compute_metrics = kwargs.get('compute_metrics', False)  # if True, compute the
    plot_metrics = kwargs.get('plot_metrics', False)  # if True, plot the network metrics for the simulation data
    plot_heatmaps = kwargs.get('plot_heatmaps', False)  # if True, plot the heatmaps for the simulation data
    
    # run simulations that constitute the sensitivity analysis
    if run_sims:
        print(f'Running simulations for sensitivity analysis with {LEVELS} levels...')
        # run the simulations that constitute the sensitivity analysis
        permuted_sim_paths = run_simulation_permutations(kwargs)
        kwargs['permuted_sim_paths'] = permuted_sim_paths  # add the permuted data paths to the kwargs for further processing
    else:
        print('Skipping simulation runs for sensitivity analysis.')
    
    # compute the network metrics for the simulation data
    if compute_metrics:
        print('Computing network metrics for the simulation data...')
        # compute the network metrics for the simulation data
        permuted_metric_paths, permuted_metric_data = compute_permutation_network_metrics(kwargs)
        kwargs['permuted_metric_paths'] = permuted_metric_paths  # add the permuted metric paths to the kwargs for further processing
    else:
        print('Skipping network metrics computation for sensitivity analysis.')
    
    # plot the network metrics for the simulation data
    if plot_metrics:
        print('Plotting network metrics for the simulation data...')    
        # # HACK
        # search output dir for .npy files recursively
        permuted_metric_paths = glob.glob(os.path.join(OUTPUT_DIR, '**', '*.npy'), recursive=True)
        max_workers = MAX_WORKERS # maximum number of workers for parallel processing
        selected_files = permuted_metric_paths # select the files to be processed, in this case all .npy files in the output directory
        if DEBUG_LIMITED_PLOTTING:
            # limit the number of files to be processed for debugging purposes
            selected_files = selected_files[:10]
        if max_workers>1:
        #if args.num_workers > 1:
            #with Pool(processes=args.num_workers) as pool:
            with Pool(processes=max_workers) as pool:
                npy_list = pool.map(load_metrics_file, selected_files)
        else:
            npy_list = list(map(load_metrics_file, selected_files))
            
        # remove None values from the list
        npy_list = [npy for npy in npy_list if npy is not None]
        
        plot_kwargs = {'y_lim': tuple([0, 16])} # set y-axis limits for the plot, adjust as needed
        parallel = True
        num_workers = MAX_WORKERS  # use the maximum number of workers available
        plot_network_metrics_v2(npy_list, plot_kwargs, parallel=parallel, num_workers=num_workers)  # plot the network metrics for the simulation data
        print(f'Plotted {len(npy_list)} network metrics files from {OUTPUT_DIR}')
        # # HACK
    else:
        print('Skipping network metrics plotting for sensitivity analysis.')
    
    # plot the sensitivity analysis results as heatmap
    if plot_heatmaps:
        print('Plotting heatmaps for the simulation data...')
        heatmap_output_dir = os.path.join(OUTPUT_DIR, 'heatmaps')  # define the output directory for the heatmaps
        hkwargs = {
            'output_dir': heatmap_output_dir,  # output directory for the heatmaps
            'input_dir': OUTPUT_DIR,  # input directory containing the simulation data
            'num_workers': MAX_WORKERS,  # number of workers for parallel processing
            'levels': LEVELS,  # number of sensitivity analysis levels, subsequently heatmap levels
            'params': EVOL_PARAMS,  # evolution parameters for the simulation data
            'debug_limited_heatmap': DEBUG_LIMITED_HEATMAP,  # if True, limit the heatmap to a subset of the data for debugging purposes
        }
        plot_heat_maps(**hkwargs)
    else:
        print('Skipping heatmap plotting for sensitivity analysis.')
    
    # plot the network metrics for the simulation data
    #plot_permutation_network_metrics(kwargs) # plot the network metrics for the simulation data