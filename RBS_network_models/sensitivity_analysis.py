# Notes ===================================================================================================
'''
Notes predating # aw 2025-02-27 14:36:09
    #NOTE: it has occured to me that modifying certain params this way just isnt very practical or useful
    # for example, modifying std of distribution for a param, would require getting all the values of
    # the param, and somehow remapping them to the new std for each cell. 
    # I dont think this would be very useful, and would be pretty complicated to implement.
    # by contrast, if the mean of the distribution was modified, it would be much simpler to just 
    # shift all the values by the same proportion.
    
    #'probLengthConst', 
        # # NOTE: this is included into a string that passed and evaluated in hoc,
        # 'probability': 'exp(-dist_3D / {})*{}'.format(cfg.probLengthConst, spec['prob']),
        # it's easier to modify probability to get a sense of how it affects the network
        
        # NOTE: nvm I figured it out. I can just modify the string directly.
'''
# imports ===================================================================================================
import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from concurrent.futures import ProcessPoolExecutor, as_completed
from netpyne import sim, specs
from RBS_network_models.CDKL5.DIV21.src.evol_params import params
from RBS_network_models.sim_analysis import process_simulation_v2
import traceback
from .utils.helper import indent_increase, indent_decrease
from concurrent.futures import ProcessPoolExecutor, as_completed
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import compute_network_metrics
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_network_summary_v2
from copy import deepcopy
from PyPDF2 import PdfReader, PdfWriter
import fitz  # PyMuPDF
import re
import os
import glob
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
# functions ===================================================================================================
'''newer functions'''

def metrics_loader_v3(network_metrics_file, use_memmap=True):
    """
    Helper function to process network metrics files and extract relevant data.
    
    Args:
        network_metrics_file (str): Path to the network metrics .npy file.
        use_memmap (bool): If True, loads the numpy file using memory mapping to optimize large file loading.
    
    Returns:
        dict: Dictionary containing loaded data and configuration file.
    """
    try:
        start = time.time()
        
        # Load network data with optional memory mapping
        network_data = np.load(network_metrics_file, mmap_mode='r' if use_memmap else None, allow_pickle=True).item()
        
        # Locate configuration file
        perm_dir_parent = os.path.dirname(network_metrics_file)
        perm_dir_gp = os.path.dirname(perm_dir_parent)
        
        # Attempt to find JSON config file
        cfg_file = glob.glob(f'{perm_dir_gp}/*_cfg.json')
        try:
            if not cfg_file:
                raise FileNotFoundError(f'No cfg file found for {network_metrics_file}')
            with open(cfg_file[0], 'r') as f:
                cfg_file = json.load(f)
        except:
            # Fall back to finding and loading a pickle file
            sim_file = glob.glob(f'{perm_dir_gp}/*_data.pkl')
            if not sim_file:
                raise FileNotFoundError(f'No alternative config found for {network_metrics_file}')
            from netpyne import sim  # Ensure netpyne is imported properly
            sim.loadSimCfg(sim_file[0])
            cfg_file = sim.cfg
        
        return {
            'data': network_data,
            'cfg': cfg_file
        }
    except Exception as e:
        print(f'Error loading {network_metrics_file}: {e}')
        return {'error': str(e)}

def load_network_metrics_v3(input_dir, num_workers=None, use_threads=False, use_memmap=False):
    """
    Loads network metrics from .npy files in the given directory using either threading or multiprocessing.
    Stores the loaded numpy arrays in a dictionary with filenames as keys.
    Optionally uses memory mapping to optimize large file loading.
    
    Args:
        input_dir (str): Directory to search for network metrics files.
        num_workers (int, optional): Number of workers (threads or processes) to use. Defaults to available CPUs.
        use_threads (bool, optional): If True, uses ThreadPoolExecutor; otherwise, uses ProcessPoolExecutor.
        use_memmap (bool, optional): If True, loads files using memory mapping (`mmap_mode='r'`) for efficiency.
    
    Returns:
        dict: Dictionary with filenames as keys and loaded numpy arrays as values.
    """
    
    # Locate network metrics files
    #network_metrics_files = glob.glob(os.path.join(input_dir, '**', '**', 'network_data.npy'), recursive=True)
    network_metrics_files = glob.glob(input_dir + '/**/**/network_data.npy')
    if not network_metrics_files:
        raise ValueError(f"No network metrics files found in {input_dir}")
    
    # Set number of workers
    available_cpus = os.cpu_count()
    num_workers = min(num_workers or available_cpus, len(network_metrics_files), available_cpus)
    
    # Adjust thread count per worker if using multiprocessing
    if not use_threads:
        threads_per_worker = max(1, available_cpus // num_workers)
        os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    
    # Select executor type
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    total_files = len(network_metrics_files)
    completed_files = 0
    network_metrics_data = {}
    
    
    #override
    #shorten network_metrics_files for testing
    network_metrics_files = sorted(network_metrics_files)
    #network_metrics_files = network_metrics_files[:40]
    num_workers = total_files if total_files < available_cpus else available_cpus
    #override
    results = []
    print(f"Using {num_workers} {'threads' if use_threads else 'processes'} to load {len(network_metrics_files)} network metrics files.")
    with Executor(max_workers=num_workers) as executor:
        futures = {executor.submit(metrics_loader_v3, file, use_memmap): file for file in network_metrics_files}
        for future in as_completed(futures):
            result = future.result()
            #if 'error' not in result:
                #network_metrics_data[os.path.basename(futures[future])] = result
            results.append(result)
            completed_files += 1
            print(f"Completed {completed_files} out of {total_files}")
    
    return results

def load_network_metrics_v2(input_dir, num_workers=None, use_threads=False):
    """
    Loads network metrics from .npy files in the given directory using either threading or multiprocessing.
    
    Args:
        input_dir (str): Directory to search for network metrics files.
        num_workers (int, optional): Number of workers (threads or processes) to use. Defaults to available CPUs.
        use_threads (bool, optional): If True, uses ThreadPoolExecutor; otherwise, uses ProcessPoolExecutor.
    
    Returns:
        list: List of results from processing the network metrics files.
    """
    
    # Locate network metrics files
    network_metrics_files = glob.glob(input_dir + '/**/**/network_data.npy')
    if not network_metrics_files:
        raise ValueError(f"No network metrics files found in {input_dir}")
    
    # Set number of workers
    available_cpus = os.cpu_count()
    num_workers = min(num_workers or available_cpus, len(network_metrics_files), available_cpus)
    print(f"Using {num_workers} {'threads' if use_threads else 'processes'} to load {len(network_metrics_files)} network metrics files.")
    
    # Adjust thread count per worker if using multiprocessing
    if not use_threads:
        threads_per_worker = max(1, available_cpus // num_workers)
        os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    
    # Select executor type
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    total_files = len(network_metrics_files)
    completed_files = 0
    
    results = []
    with Executor(max_workers=num_workers) as executor:
        futures = {executor.submit(metrics_loader_v2, file): file for file in network_metrics_files}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed_files += 1
            print(f"Completed {completed_files} out of {total_files}")
    
    return results

def plot_heat_maps(output_dir, input_dir, num_workers=None, levels=6, **hkwargs):
    
    # subfunctions ===================================================================================================

    def get_clean_grid(input_dir, query=None):
        """
        Constructs a grid of summary plot paths for a given query.
        
        This function identifies the original simulation directory, extracts parameter-specific 
        summary plots, and organizes them into a structured grid with the original summary 
        plot inserted in the middle of each parameter's variations.
        
        Args:
            input_dir (str): The directory to search for simulation results.
            query (str): The specific summary plot to retrieve.
        
        Returns:
            dict: A dictionary where keys are parameter names and values are dictionaries 
                mapping index positions to summary plot paths.
        
        Raises:
            ValueError: If `query` is not specified.
            AssertionError: If the number of detected original simulation directories is not exactly one.
        """
        if query is None:
            raise ValueError("query must be specified")
        
        # Identify the original simulation directory (must contain '.sa_origin' as a subdirectory)
        found = [root for root, _, _ in os.walk(input_dir) if '.sa_origin' in root]
        
        # Ensure there is exactly one original summary directory
        assert len(found) == 1, f"Expected 1 original summary plot directory, found {len(found)}"
        origin_marker_path = found[0]
        origin_summary_path = os.path.dirname(origin_marker_path)
        for root, _, files in os.walk(origin_summary_path): # find network_data.npy in origin_summary_path - may need to recurse
            for file in files:
                if query in file:
                    original_summary_path = os.path.join(root, file)
                    break
        origin_network_metrics_path = os.path.join(origin_summary_path, query)
        
        # Initialize the grid for storing summary plot paths
        grid = {}
        
        params_to_exclude = [
            'E_diam_mean', 'I_diam_mean', 'E_L_mean', 'I_L_mean', 'E_Ra_mean', 'I_Ra_mean',
        ]
        
        for param_name, param_value in params.items():
            # Skip parameters that are not lists or tuples of length 2
            if not isinstance(param_value, (list, tuple)):
                continue
            
            # Skip parameters that are not within the specified range
            if param_name in params_to_exclude:
                continue
            
            # Retrieve permutations for the current parameter
            num_permutations, summary_paths = get_perms_per_param(param_name)
            
            if num_permutations == 0:
                continue  # Skip parameters with no variations
            
            # Insert the original summary plot at the middle index of the variations
            middle_idx = num_permutations // 2
            summary_paths.insert(middle_idx, origin_network_metrics_path)
            
            # Store summary paths in the grid
            grid[param_name] = {idx: path for idx, path in enumerate(summary_paths)}
            
            # Quality check: Ensure the number of permutations does not exceed expected levels
            try:
                assert num_permutations <= levels, f"Expected at most {levels} permutations, found {num_permutations}"
            except AssertionError as e:
                print("Error:", e)
        
        # Remove empty entries from the grid
        clean_grid = {param: paths for param, paths in grid.items() if paths}
        
        return clean_grid

    def get_perms_per_param(param):
        ''' get number of permutations for a given param '''
        # init file list
        #files = []
        found = []
        
        #iterate through files in input_dir, get number of permutations from filename context
        num_permutations = 0
        param_elements = param.split('_')
        
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                parent_dir = os.path.basename(os.path.dirname(file_path))
                perm_label = parent_dir
                if param in perm_label:
                    #if all([element in perm_label for element in param_elements]):
                    if '.npy' in file:
                        num_permutations += 1
                        #files.append(file_path)
                        found.append(file_path)
                        print('Found permutation for', param, 'in', file_path)
        
        # return number of permutations found
        return num_permutations, found
            
    def extract_original_metric(data_list, metric_path):
        """Extracts the original metric value from network metrics data."""
        for i, data in enumerate(data_list):
            #base = os.path.basename(data['sim_data_path'])
            sim_data_path = data['sim_data_path']
            sim_data_dir = os.path.dirname(sim_data_path)
            # if folder .sa_origin is found, then it's the original sim
            list_of_dirs_in_dir = os.listdir(sim_data_dir)
            if '.sa_origin' in list_of_dirs_in_dir:
                #if base.startswith('_'):
                original_metric = data.copy()
                for path_part in metric_path:
                    if 'network_metrics' in path_part:
                        continue
                    original_metric = original_metric.get(path_part, np.nan)
                return i, original_metric
            
            
            
            # if base.startswith('_'):
            #     original_metric = data.copy()
            #     for path_part in metric_path:
            #         if 'network_metrics' in path_part:
            #             continue
            #         original_metric = original_metric.get(path_part, np.nan)
            #     return i, original_metric
        return None, None

    def compute_metric_bounds(metric_values, original_metric):
        """Computes min and max metric values within 2 standard deviations."""
        if original_metric is None or np.isnan(original_metric):
            return np.nan, np.nan
        
        std_dev = np.std(metric_values)
        min_val, max_val = original_metric - 2 * std_dev, original_metric + 2 * std_dev
        return max(min(metric_values), min_val), min(max(metric_values), max_val)

    def prepare_clean_grid(clean_grid, data_list, param, levels):
        """Prepares clean_grid by populating it with data and arranging levels."""
        clean_grid[param]['data'] = {}
        for data in data_list:
            sim_data_path = data['sim_data_path']
            base = os.path.basename(sim_data_path)
            simLabel = os.path.basename(os.path.dirname(sim_data_path))
            if param in base and not base.startswith('_'):
                clean_grid[param]['data'][simLabel] = data
        
        #assert len(clean_grid[param]['data']) == levels, f'Expected {levels} levels, found {len(clean_grid[param]["data"])}'
        
        row_data = clean_grid[param]['data']
        middle_idx = levels // 2
        new_row_data = {}
        for key, data in row_data.items():
            level_pos = int(re.search(r'\d+$', key).group())
            if level_pos >= middle_idx:
                level_pos += 1
            new_row_data[level_pos] = data
        return new_row_data

    def extract_metric_value(data, metric_path):
        """Recursively extracts a numerical metric value from a nested dictionary."""
        for path_part in metric_path:
            if 'network_metrics' in path_part:
                continue
            if isinstance(data, dict):
                data = data.get(path_part, np.nan)
            else:
                break
        return float(data) if isinstance(data, (int, float, np.number)) else np.nan
    
    def plot_metric_heatmap_v3(output_dir, metric_path, metric_name, network_metrics_data, clean_grid, levels):
        """
        Plots heatmaps for a specified network metric.
        """
        print(f"Plotting summary grid for {metric_name} with color gradient")
        data_list = [data['data'] for data in network_metrics_data]
        original_key, original_metric = extract_original_metric(data_list, metric_path)
        
        #metric_values = [float(data.get(path_part, np.nan)) for data in data_list for path_part in metric_path if 'network_metrics' not in path_part]
        metric_values = [extract_metric_value(data, metric_path) for data in data_list]
        min_metric, max_metric = compute_metric_bounds(metric_values, original_metric)
        
        #cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(0, 0, 1), (1, 1, 1), (1, 0, 0)], N=100)
        # use a softer/more offwhite color for the middle
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(0, 0, 1), (0.875, 0.875, 0.875), (1, 0, 0)], N=100)
        norm = (mcolors.CenteredNorm(vcenter=original_metric, halfrange=max(abs(min_metric - original_metric), abs(max_metric - original_metric)))
                if not np.isnan(original_metric) else mcolors.Normalize(vmin=min_metric, vmax=max_metric))
        
        for param in clean_grid:
            clean_grid[param]['data'] = prepare_clean_grid(clean_grid, data_list, param, levels)
        
        #fig, axs = plt.subplots(len(clean_grid), levels + 1, figsize=(2 * (levels + 1), len(clean_grid)))
        fig, axs = plt.subplots(len(clean_grid), levels + 1, figsize=(2* len(clean_grid), 2 * len(clean_grid)))

        for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
            for key, data in clean_grid[param]['data'].items():
                try:
                    metric_value = data.copy()
                    for path_part in metric_path:
                        if 'network_metrics' in path_part:
                            continue
                        metric_value = metric_value.get(path_part, np.nan)
                    
                    color = cmap(norm(metric_value))
                    axs[row_idx, key].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
                    #axs[row_idx, key].text(0.5, 0.5, f'{metric_value:.2f}', ha='center', va='center', fontsize=12)
                    axs[row_idx, key].axis('off')
                    
                    sim_data_path = data['sim_data_path']
                    permuted_value = next((netmet['cfg'].get(param, None) for netmet in network_metrics_data if netmet['data']['sim_data_path'] == sim_data_path), None)
                    
                    if row_idx == 0: # only on first ?
                        if permuted_value is not None:
                            #axs[row_idx, key].set_title(f'@{round(permuted_value, 3)}', fontsize=14)
                            # get the column position relative to origin (in the middle) and show that instead of perm_value
                            #level_pos = int(re.search(r'\d+$', key).group())
                            level_diff = key - 3
                            #if level_pos >= levels // 2: level_pos += 1
                            #level_diff = level_pos - levels // 2
                            if level_diff != 0: 
                                if level_diff<0: axs[row_idx, key].set_title(f'{level_diff}', fontsize=55, fontweight='bold')
                                elif level_diff>0: axs[row_idx, key].set_title(f'+{level_diff}', fontsize=55, fontweight='bold')
                            else: pass # dont print the origin value
                        
                    print(f"Plotted {param} in row {row_idx}, column {key}")    
                except Exception as e:
                    print(f"Error loading plot for key {key}: {e}")
                    
            #if any expected keys are missing from the row, fill them with black rectangles
            expected_keys = [0, 1, 2, 3, 4, 5, 6]
            for key in expected_keys:
                if key == 3:
                    # offwhite rect
                    axs[row_idx, key].add_patch(plt.Rectangle((0, 0), 1, 1, color=(0.875, 0.875, 0.875)))
                    axs[row_idx, key].axis('off')
                    continue
                if key not in clean_grid[param]['data']:
                    axs[row_idx, key].add_patch(plt.Rectangle((0, 0), 1, 1, color='black'))
                    axs[row_idx, key].axis('off')
            
            for col_idx in range(levels + 1):
                axs[row_idx, col_idx].axis('off')
                
            print(f"Plotted {param} in row {row_idx}")
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.90, top=0.925)
        
        # Configure Matplotlib to use LaTeX and include the bm package
        # import matplotlib
        # matplotlib.rc('text', usetex=True)
        # matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
        
        for row_idx, param in enumerate(clean_grid):
            pos = axs[row_idx, 0].get_position()
            #fig.text(pos.x0 - 0.025, pos.y0 + pos.height / 2, param, va='center', ha='right', fontsize=14, rotation=0)
            # bigger font and printed at a 45 degree angle from the top to save space
            
            param_text = param
            
            # if 'propVelocity' in param:
            #     # replace with 'v' with bar symbol over it
            #     param_text = param_text.replace('propVelocity', r'$v_{\mathrm{prop}}$')
            # #replace prob w P()            
            # if 'prob' in param and 'probLengthConst' not in param:
            #     param_text = param_text.replace('prob', 'P(')
            #     param_text = param_text + ')'
            # # replace LengthConst with lambda (λ)
            # if 'probLengthConst' in param:
            #     param_text = param_text.replace('LengthConst', 'λ')
            # if 'weight' in param:
            #     param_text = param_text.replace('weight', 'w(')
            #     param_text = param_text + ')'
            # if 'gnabar' in param:
            #     # replace with 'Na' with bar symbol over it
            #     param_text = param_text.replace('gnabar_', r'$g_{\mathrm{Na}}$')
            # if 'gkbar' in param:
            #     # replace with 'K' with bar symbol over it
            #     param_text = param_text.replace('gkbar_', r'$g_{\mathrm{K}}$')
            # if 'tau1_exc' in param:
            #     # replace with 'τ' with bar symbol over it
            #     param_text = param_text.replace('tau1_exc', r'$\tau1_{\mathrm{exc}}$')
            # if 'tau2_exc' in param:
            #     # replace with 'τ' with bar symbol over it
            #     param_text = param_text.replace('tau2_exc', r'$\tau2_{\mathrm{exc}}$')
            # if 'tau1_inh' in param:
            #     # replace with 'τ' with bar symbol over it
            #     param_text = param_text.replace('tau1_inh', r'$\tau1_{\mathrm{inh}}$')
            # if 'tau2_inh' in param:
            #     # replace with 'τ' with bar symbol over it
            #     param_text = param_text.replace('tau2_inh', r'$\tau2_{\mathrm{inh}}$')
            
            
            # Formatting replacements with bold symbols
            if 'propVelocity' in param:
                param_text = param_text.replace('propVelocity', r'$\mathbf{v_{\mathrm{prop}}}$')
            if 'prob' in param and 'probLengthConst' not in param:
                param_text = param_text.replace('prob', r'$\mathbf{P(}$')
                param_text += ')'
            if 'probLengthConst' in param:
                param_text = param_text.replace('probLengthConst', r'$\mathbf{\lambda}$')
            if 'weight' in param:
                param_text = param_text.replace('weight', r'$\mathbf{w(}$')
                param_text += ')'
            if 'gnabar' in param:
                param_text = param_text.replace('gnabar_', r'$\mathbf{g_{\mathrm{Na}}}$')
            if 'gkbar' in param:
                param_text = param_text.replace('gkbar_', r'$\mathbf{g_{\mathrm{K}}}$')
            if 'tau1_exc' in param:
                param_text = param_text.replace('tau1_exc', r'$\mathbf{\tau_{1,\mathrm{exc}}}$')
            if 'tau2_exc' in param:
                param_text = param_text.replace('tau2_exc', r'$\mathbf{\tau_{2,\mathrm{exc}}}$')
            if 'tau1_inh' in param:
                param_text = param_text.replace('tau1_inh', r'$\mathbf{\tau_{1,\mathrm{inh}}}$')
            if 'tau2_inh' in param:
                param_text = param_text.replace('tau2_inh', r'$\mathbf{\tau_{2,\mathrm{inh}}}$')

                
            #make param_text bold
            #param_text = param_text.replace(param_text, r'$\mathbf{' + param_text + '}$') 
            #param_text = f'$\\mathbf{{{param_text}}}$'
            #param_text = f'$\\textbf{{{param_text}}}$'
            #param_text = f'$\\mathbb{{{param_text}}}$'  # Use \mathbb{} as an alternative for bold text
            #param_text = f'$\\boldsymbol{{{param_text}}}$'
    
            fig.text(
                pos.x0 - 0.025, pos.y0 + pos.height / 2,
                param_text, va='center', ha='right',
                fontsize=55, 
                #rotation=45, 
                fontweight='bold'
            )
            
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
        fig.colorbar(sm, cax=cbar_ax)
        
        #increase color bar tick mark text
        cbar_ax.tick_params(labelsize=40)
        
        #fig.suptitle(f'Heatmap: {metric_name}', fontsize=50)
        #only use last part of path, define title as everything after 'metrics' in metric_name
        title = metric_name.split('metrics')[-1]
        # remove leading _ if present
        if title.startswith('_'): title = title[1:]
        fig.suptitle(title, fontsize=75, fontweight='bold')
            
        #plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'_heatmap_{metric_name}.png')
        plt.savefig(output_path, dpi=100)
        pdf_path = os.path.join(output_dir, f'_heatmap_{metric_name}.pdf')
        plt.savefig(pdf_path)
        print(f'Saved heatmap to {output_path}')
        print(f'Saved heatmap to {pdf_path}')
        return output_path
    
    def plot_metric_heatmap_v2(output_dir, metric_path, metric_name, network_metrics_data, clean_grid, levels):
        """
        Plots heatmaps for a specified network metric.
        """
        print(f"Plotting summary grid for {metric_name} with color gradient")
        data_list = [data['data'] for data in network_metrics_data]
        original_key, original_metric = extract_original_metric(data_list, metric_path)
        
        #metric_values = [float(data.get(path_part, np.nan)) for data in data_list for path_part in metric_path if 'network_metrics' not in path_part]
        metric_values = [extract_metric_value(data, metric_path) for data in data_list]
        min_metric, max_metric = compute_metric_bounds(metric_values, original_metric)
        
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(0, 0, 1), (1, 1, 1), (1, 0, 0)], N=100)
        norm = (mcolors.CenteredNorm(vcenter=original_metric, halfrange=max(abs(min_metric - original_metric), abs(max_metric - original_metric)))
                if not np.isnan(original_metric) else mcolors.Normalize(vmin=min_metric, vmax=max_metric))
        
        for param in clean_grid:
            clean_grid[param]['data'] = prepare_clean_grid(clean_grid, data_list, param, levels)
        
        fig, axs = plt.subplots(len(clean_grid), levels + 1, figsize=(2 * (levels + 1), len(clean_grid)))
        
        for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
            for key, data in clean_grid[param]['data'].items():
                try:
                    metric_value = data.copy()
                    for path_part in metric_path:
                        if 'network_metrics' in path_part:
                            continue
                        metric_value = metric_value.get(path_part, np.nan)
                    
                    color = cmap(norm(metric_value))
                    axs[row_idx, key].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
                    axs[row_idx, key].text(0.5, 0.5, f'{metric_value:.2f}', ha='center', va='center', fontsize=12)
                    axs[row_idx, key].axis('off')
                    
                    sim_data_path = data['sim_data_path']
                    permuted_value = next((netmet['cfg'].get(param, None) for netmet in network_metrics_data if netmet['data']['sim_data_path'] == sim_data_path), None)
                    
                    if permuted_value is not None:
                        axs[row_idx, key].set_title(f'@{round(permuted_value, 3)}', fontsize=14)
                        
                    print(f"Plotted {param} in row {row_idx}, column {key}")    
                except Exception as e:
                    print(f"Error loading plot for key {key}: {e}")
            
            for col_idx in range(levels + 1):
                axs[row_idx, col_idx].axis('off')
                
            print(f"Plotted {param} in row {row_idx}")
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.90, top=0.925)
        
        for row_idx, param in enumerate(clean_grid):
            pos = axs[row_idx, 0].get_position()
            fig.text(pos.x0 - 0.025, pos.y0 + pos.height / 2, param, va='center', ha='right', fontsize=14, rotation=0)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
        fig.colorbar(sm, cax=cbar_ax)
        
        fig.suptitle(f'Heatmap: {metric_name}', fontsize=16)
        output_path = os.path.join(output_dir, f'_heatmap_{metric_name}.png')
        plt.savefig(output_path, dpi=100)
        pdf_path = os.path.join(output_dir, f'_heatmap_{metric_name}.pdf')
        plt.savefig(pdf_path)
        print(f'Saved heatmap to {output_path}')
        print(f'Saved heatmap to {pdf_path}')
        return output_path
    
    def plot_metric_heatmap(output_dir, metric_path, metric_name, network_metrics_data, clean_grid, levels):
        """
        Generalized function to plot heatmaps for a specified network metric.
        
        Args:
            output_dir (str): Directory to save the heatmap.
            metric_path (list): List of keys to navigate the metric in the network_metrics_data dictionary.
            metric_name (str): Name of the metric to display in the title and filename.
            network_metrics_data (dict): Dictionary containing network metrics data.
            clean_grid (dict): Dictionary of parameters and their data paths.
            levels (int): Number of levels for each parameter.
        """
        print(f"Plotting summary grid for {metric_name} with color gradient")
        
        # Find the original metric value - dict
        # for key in network_metrics_data.keys():
        #     if key.startswith('_'):
        #         original_key = key
        #         original_metric = network_metrics_data[key]['data']
        #         for path_part in metric_path:
        #             original_metric = original_metric[path_part]
        #         break
        
        #network_metrics_data_copy = deepcopy(network_metrics_data)
        
        #make list
        data_list = []
        for data in network_metrics_data:
            data_list.append(data['data'])
            
        # replace list, lazy
        #network_metrics_data = data_list                
        
        # patch for list
        for i, data in enumerate(data_list):
            sim_data_path = data['sim_data_path']
            base = os.path.basename(sim_data_path)
            if base.startswith('_'):
                #if data.startswith('_'):
                #original_key = data
                #original_metric = network_metrics_data[data]['data']
                #original_metric = data[key]
                original_key = i
                original_metric = data.copy()
                for path_part in metric_path:
                    if 'network_metrics' in path_part: continue
                    original_metric = original_metric[path_part]
                break
        
        # # Determine min and max metric values
        # metric_list = []  # Initialize list to store metric values
        # min_metric = float('inf')
        # max_metric = float('-inf')
        # for key in network_metrics_data.keys():
        #     data = network_metrics_data[key]['data']
        #     metric_value = data
        #     for path_part in metric_path:
        #         metric_value = metric_value[path_part]
        #         #print(metric_value)
        #     metric_list.append(float(metric_value))
        #     min_metric = min(min_metric, metric_value)
        #     max_metric = max(max_metric, metric_value)
        
        # patch for list
        metric_list = []  # Initialize list to store metric values
        min_metric = float('inf')
        max_metric = float('-inf')
        for data in data_list:
            #data = network_metrics_data[key]['data']
            metric_value = data.copy()
            for path_part in metric_path:
                if 'network_metrics' in path_part: continue
                try: metric_value = metric_value[path_part]
                except:
                    #print(f"Error loading metric value for {path_part}")
                    metric_value = np.nan
                    continue
            metric_list.append(float(metric_value))
            min_metric = min(min_metric, metric_value)
            max_metric = max(max_metric, metric_value)
        
        # get min and max metric values within 2 std deviations to avoid outliers
        std_dev = np.std(metric_list)
        max_val = original_metric + 2 * std_dev
        min_val = original_metric - 2 * std_dev
        
        # now if min and max arre within 2 std deviations, use them, else use the std values
        min_metric = max(min_metric, min_val)
        max_metric = min(max_metric, max_val)
        
        # Define colormap and normalization
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
        
        # Handle the case where original_metric is NaN
        if not np.isnan(original_metric):
            #typical case
            norm = mcolors.CenteredNorm(vcenter=original_metric, halfrange=max(abs(min_metric - original_metric), abs(max_metric - original_metric)))
        else:
            # handle case where original_metric is NaN
            norm = mcolors.Normalize(vmin=min_metric, vmax=max_metric) # normalized without centering around original simulation
            
        # Prepare data dicts for clean_grid
        for param, summary_paths in clean_grid.items():
            clean_grid[param]['data'] = {}
            
        # Update clean_grid with network_metrics_data
        for param, summary_paths in clean_grid.items():
            #for key, data in network_metrics_data.items():
            for data in data_list: # patch for list
                sim_data_path = data['sim_data_path']
                base = os.path.basename(sim_data_path)
                simLabel = os.path.basename(os.path.dirname(sim_data_path))
                if param in base:
                    # skip og sim if it's based on param from previous SA
                    if base.startswith('_'): continue
                    clean_grid[param]['data'].update({simLabel: data})
        assert len(clean_grid[param]['data']) == levels, f'Expected {levels} levels, found {len(clean_grid[param]["data"])}'
            
        # Generate heatmap
        n_rows = len(clean_grid)
        n_cols = levels + 1
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1 * n_rows))
        
        for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
            
            # # aw 2025-03-04 16:32:38 - since we know the number of levels we have, we can adjust what the middle index is
            row_data = clean_grid[param]['data']
            middle_idx = levels // 2 # for 6 levels, idx is 3 (0, 1, 2, 3, 4, 5, 6) - 3 is the middle (perm, perm, perm, og, perm, perm, perm)
            new_row_data = {}
            for idx, (key, data) in enumerate(clean_grid[param]['data'].items()):
                # get correct idx for perm based on key. Each key should have a number value in it.
                # find number in key (which should be a string including a number at the end)
                # get the number from the key
                #if idx != middle_idx: # perm case
                level_pos = int(re.search(r'\d+$', key).group())
                if level_pos >= middle_idx: level_pos += 1
                #new_row_data[key] = data
                new_row_data[level_pos] = data                    
                # elif idx == middle_idx: # og case
                #     #new_row_data['original_data'] = data_list[original_key]
                #     new_row_data[idx] = data_list[original_key]
                #new_row_data[key] = data
            new_row_data[middle_idx] = data_list[original_key]
            clean_grid[param]['data'] = new_row_data
            
            # aw 2025-03-04 16:31:39 updating this block of code above to handle incomplete lists of data - incase some sims or analyses fail.
            # row_data = clean_grid[param]['data']
            # sorted_row_data = dict(sorted(row_data.items()))
            # middle_idx = len(sorted_row_data) // 2
            # new_row_data = {}
            # for idx, (key, value) in enumerate(sorted_row_data.items()):
            #     if idx == middle_idx:
            #         new_row_data['original_data'] = data_list[original_key]
            #     new_row_data[key] = value
            # clean_grid[param]['data'] = new_row_data
            
            # Plot each cell in the row
            for col_idx, (key, data) in enumerate(clean_grid[param]['data'].items()):
                try:
                    metric_value = data
                    for path_part in metric_path:
                        if 'network_metrics' in path_part: continue
                        try: metric_value = metric_value[path_part]
                        except: 
                            metric_value = np.nan
                            continue
                    color = cmap(norm(metric_value))
                    
                    # key is the real column index... #HACK
                    axs[row_idx, key].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
                    axs[row_idx, key].text(0.5, 0.5, f'{metric_value:.2f}', ha='center', va='center', fontsize=12)
                    axs[row_idx, key].axis('off')
                    
                    # axs[row_idx, col_idx].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
                    # axs[row_idx, col_idx].text(0.5, 0.5, f'{metric_value:.2f}', ha='center', va='center', fontsize=12)
                    # axs[row_idx, col_idx].axis('off')
                    permuted_param = param
                    #permuted_value = data['data']['simConfig'][param]
                    #permuted_value = data[param]
                    sim_data_path = data['sim_data_path']
                    for netmet in network_metrics_data:
                        sim_check = netmet['data']['sim_data_path']
                        if sim_data_path == sim_check:
                            permuted_value = netmet['cfg'][param]
                            break
                    #cfg = network_metrics_data[key]['cfg']
                    try:
                        permuted_value = round(permuted_value, 3)
                    except:
                        pass
                    axs[row_idx, key].set_title(f'@{permuted_value}', fontsize=14)
                except Exception as e:
                    print(f"Error loading plot for key {key}: {e}")
            #print(f"Plotted {param} in row {row_idx}")

            # remove axes for all subplots, even if nothing plotted
            for col_idx in range(levels + 1):
                axs[row_idx, col_idx].axis('off')
                
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.90, top=0.925)
        for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
            pos = axs[row_idx, 0].get_position()
            x = pos.x0 - 0.025
            fig.text(x, pos.y0 + pos.height / 2, param, va='center', ha='right', fontsize=14, rotation=0)
        
        # Add colorbar
        # NOTE: sm generated with norm based on original_metric = nan will result in stack overrflow when trying to generate the colorbar - to deal with this,
        # to deal with this, norm has a special case above for when original_metric is nan. norm will be set to a norm that is not centered on original simulaiton value.
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
        fig.colorbar(sm, cax=cbar_ax)
        
        # Add title and save
        fig.suptitle(f'Heatmap: {metric_name}', fontsize=16)
        output_path = os.path.join(output_dir, f'_heatmap_{metric_name}.png')
        plt.savefig(output_path, dpi=100)
        pdf_path = os.path.join(output_dir, f'_heatmap_{metric_name}.pdf')
        plt.savefig(pdf_path)
        print(f'Saved heatmap to {output_path}')
        print(f'Saved heatmap to {pdf_path}')
        #print(f'Saved heatmap to {output_path}')
        return output_path
    
    def find_metric_paths(network_metrics_data, parent_path="network_metrics"):
        """
        Recursively find paths that resolve to a single float/int 
        or a list of floats/ints, while treating dicts with only numeric keys as lists.
        """
        keys_to_ignore = [
            "std", 
            "cov",
            "median",
            "burst_ids",
            "burst_part",
            "burst_parts",
            ".data",
            "num_bursts", # burst rate is more informative with variable durations
            "unit_metrics",
            "gids",
            "spiking_metrics_by_unit",
            "spiking_times_by_unit",
            ".unit_metrics",
            "unit_types",
            "min",
            "max",
            "simData",
            "popData",
            "cellData",
            
            # fix this later
            'E_diam_mean',
            'I_diam_mean',
            'E_L_mean',
            'I_L_mean',
            'E_Ra_mean',
            'I_Ra_mean',
            
            ]
        metric_paths = set()

        def is_numeric_key_dict(d):
            """Check if all keys in the dictionary are numeric (i.e., should be treated as a list)."""
            return isinstance(d, dict) and all(re.match(r"^\d+$", str(k)) for k in d.keys())

        def recurse(d, path):
            if isinstance(d, dict):
                if is_numeric_key_dict(d):  # Treat as a list and stop recursion
                    if not any(ign in path for ign in keys_to_ignore):          
                        print(f"Found metric path: {path}")
                        metric_paths.add(path)
                    return
                for key, value in d.items():
                    new_path = f"{path}.{key}"
                    recurse(value, new_path)
            elif isinstance(d, (int, float)) or (isinstance(d, list) and all(isinstance(i, (int, float)) for i in d)):
                if not any(ign in path for ign in keys_to_ignore):          
                    print(f"Found metric path: {path}")
                    metric_paths.add(path)

            # metric paths

        # prep
        list_network_data = []
        for data in network_metrics_data:
            try:
                list_network_data.append(data['data'])
            except Exception as e:
                print(f"Error loading network data: {e}")
                continue     
        
        # 
        count = 0
        data = list_network_data
        for entry in data:  # Assuming network_metrics_data is a list of dicts
            recurse(entry, parent_path)
            count += 1
            if count > 3: break # really only need to do this once - but will do it three times to be sure
            #break # really only need to do this once
        return sorted(metric_paths)

    # main ===================================================================================================
    # get dict of paths for matrix
    clean_grid = get_clean_grid(input_dir, query='network_data.npy') # prepare grid of network_data.npy file paths
    network_metrics_data = load_network_metrics_v3(input_dir, num_workers=num_workers, use_threads=True, use_memmap=False) # Collect network_metrics.npy files and process #NOTE: parallel processing is used here
    metric_paths = find_metric_paths(network_metrics_data)
    print(f'{len(metric_paths)} metric paths found')
    
    # testing
    for metric_path in metric_paths:
        try:
            metric_path_parts = metric_path.split('.')
            metric_name = '_'.join(metric_path_parts)
            #plot_metric_heatmap(output_dir, metric_path_parts, metric_name, network_metrics_data, clean_grid, levels)
            #plot_metric_heatmap_v2(output_dir, metric_path_parts, metric_name, network_metrics_data, clean_grid, levels) #TODO: add reference data...
            plot_metric_heatmap_v3(output_dir, metric_path_parts, metric_name, network_metrics_data, clean_grid, levels)
        except Exception as e:
            traceback.print_exc()
            print(f"Error plotting heatmap for {metric_path}: {e}")
            continue
        
    print('done.')

def plot_sensitivity_analysis_v3(kwargs):    
    """
    Plots a grid of summary plots with color-modulated cells based on changes in burst rates.
    """    
    # main ===================================================================================================
    
    #unpack kwargs    
    og_simulation_data_path = kwargs['sim_data_path']
    sensitvity_analysis_output_dir = kwargs['output_dir']
    num_workers = kwargs['max_workers']
    levels = kwargs['levels']
    plot_heatmaps = kwargs['plot_heatmaps']
    
    # Set up paths and parameters
    input_dir = sensitvity_analysis_output_dir
    output_dir = os.path.join(sensitvity_analysis_output_dir, 'summary_plots')
    sim_data_path = og_simulation_data_path
    
    # assertions
    assert os.path.exists(input_dir), f'Input directory {input_dir} does not exist.'
    assert os.path.exists(sim_data_path), f'Simulation data path {sim_data_path} does not exist.'
    
    # make output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    # plot summary grid
    hkwargs = {
        'output_dir': output_dir,
        'input_dir': input_dir,
        'num_workers': num_workers,
        'levels': levels
    }
    if plot_heatmaps: plot_heat_maps(**hkwargs)
    
def metrics_loader_v2(network_metrics_file):
    """
    Helper function to process network metrics files and extract burst rate information.
    """
    try:
        start = time.time()
        
        network_data = np.load(network_metrics_file, allow_pickle=True).item()
        
        # get cfg path.
        perm_dir_parent = os.path.dirname(network_metrics_file)
        perm_dir_gp = os.path.dirname(perm_dir_parent)
        # use glob to look for file ending in _cfg.json
        cfg_file = glob.glob(f'{perm_dir_gp}/*_cfg.json')
        try:
            if len(cfg_file) == 0:
                print('No cfg file found for', network_metrics_file)
                raise FileNotFoundError('No cfg file found for', network_metrics_file)
            cfg_file = cfg_file[0]
            # load json
            import json
            with open(cfg_file, 'r') as f:
                cfg_file = json.load(f)
        except:
            # find pkl file instead
            sim_file = glob.glob(f'{perm_dir_gp}/*_data.pkl')
            sim.loadSimCfg(sim_file[0])
            cfg_file = sim.cfg
        
        return {
            'data': network_data,
            'cfg': cfg_file
        }
    except Exception as e:
        print('Error loading', network_metrics_file, ':', e)
        return e

def plot_sensitivity_grid_plots_v2(
    og_simulation_data_path, 
    sensitvity_analysis_output_dir,
    num_workers=None,
    burst_rates=None,
    original_burst_rate=None,
    format_option='long',
    levels=6,
    plot_grid=True,
    plot_heatmaps=True
    ):
    """
    Plots a grid of summary plots with color-modulated cells based on changes in burst rates.
    """    
    # subfunctions ===================================================================================================
    def get_clean_grid(
        input_dir,
        #query = 'summary_plot.png'
        query = None
        ):
        if query is None: raise ValueError('query must be specified')
        
        # get paths in grid format
        
        # walk through input_dir and find all permutations
        # for root, _, files in os.walk(input_dir):
        #     for file in files:
        #         if query in file:
        #             print('Found:', file) 
        
        # find the original summary plot - # HACK: this is a pretty shitty way to handle this - should probably just make a list or something to load
        # when running the perms func
        sim_label_parts = os.path.basename(input_dir).split('_')[1:-1]
        sim_label_parts.insert(0, '')
        #try to remove the word data if it is present
        if 'data' in sim_label_parts:
            sim_label_parts.remove('data')
        sim_label = '_'.join(sim_label_parts)
        print('sim_label:', sim_label)
        found=[]
        for root, _, files in os.walk(input_dir):
            #files = []
            #ignore archived files
            if '__archive' in root: continue
            for file in files:
                # simLabel = os.path.basename(input_dir)
                # simLable = sim
                file_path = os.path.join(root, file)
                parent_dir = os.path.basename(os.path.dirname(file_path))
                if sim_label in parent_dir and query in file_path:                 
                    #if query in file_path:
                    #og_sumamry_path = os.path.join(input_dir, file)
                    og_sumamry_path = file_path
                    #found.append(os.path.join(input_dir, file))
                    found.append(file_path)
                    print('Found original summary plot:', og_sumamry_path)
                    break
        assert len(found) == 1, f'Expected 1 original summary plot, found {len(files)}'
        
        # iterate through params, load associated plots, build a grid of paths
        grid = {} #store png paths here
        for param_idx, param in enumerate(params):
            
            #check if param value is a list, or tuple, of two values - if so dont skip, else skip
            param_val = params[param]
            if not isinstance(param_val, (list, tuple)):
                #print('skipping', param)
                continue
            #print() #print a line to separate outputs
            
            # Get permutations of param
            num_permutations, summary_paths = get_perms_per_param(param)
            
            # Arrange data into a grid
            grid[param] = {}
            middle_idx = num_permutations // 2
            #insert og_summary_plot in the middle of summary_paths list
            if len(summary_paths) > 0:
                summary_paths.insert(middle_idx, og_sumamry_path)
                for idx, slide_path in enumerate(summary_paths):
                    grid[param][idx] = slide_path
            print('num_permutations:', num_permutations)
            if num_permutations == 0: continue
            print() #print a line to separate outputs
            
            # quality check - make sure number of permutations is less than or equal to levels
            try:
                assert num_permutations <= levels, f'Expected {levels} permutations, found {num_permutations}'
            except Exception as e:
                print('Error:', e)                
            
        # remove empty rows
        clean_grid = {param: summary_paths for param, summary_paths in grid.items() if len(summary_paths) > 0}
        return clean_grid

    def get_perms_per_param(param):
        ''' get number of permutations for a given param '''
        # init file list
        #files = []
        found = []
        
        #iterate through files in input_dir, get number of permutations from filename context
        num_permutations = 0
        param_elements = param.split('_')
        
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                parent_dir = os.path.basename(os.path.dirname(file_path))
                perm_label = parent_dir
                if param in perm_label:
                    #if all([element in perm_label for element in param_elements]):
                    if '.npy' in file:
                        num_permutations += 1
                        #files.append(file_path)
                        found.append(file_path)
                        print('Found permutation for', param, 'in', file_path)
        
        # return number of permutations found
        return num_permutations, found
            
    def plot_summary_grid(
        output_dir,
        num_workers=None,
        #burst_rates=None,
        # original_burst_rate=None,
        # format_option = 'long' # aw 2025-01-11 17:02:34 - retiring this option
        format_option = 'matrix'
        ):
        
        # Collect network_metrics.npy files and process

        
        # Plot summary grid
        print('Plotting summary grid')        
        
        if format_option == 'matrix':
            '''
            #arrange data into a grid of plots
            # y axis = params - any number of parameters that were varied in the simulation
            # x axis =  simulation permutations - usually 2-6 permutations of the simulation (ideally an even number) 
                        # + 1 column for the original simulation in the middle. (which is why an even number of permutations is nice)
            '''
            
            # get dict of paths for matrix
            clean_grid = get_clean_grid(output_dir)
            
            # Create a grid of plots
            n_rows = len(clean_grid)
            n_cols = levels+1
            #fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 7.5 * n_rows))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
            for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                for col_idx, summary_path in summary_paths.items():
                    try:
                        img = mpimg.imread(summary_path)
                        axs[row_idx, col_idx].imshow(img)
                        axs[row_idx, col_idx].axis('off')
                        axs[row_idx, col_idx].set_title(f'param: {param} (perm: {col_idx})', fontsize=14)
                    except Exception as e:
                        print('Error loading plot:', e)
                print(f'Plotted {param} in row {row_idx}')
                        
            # Save the plot
            print('Saving summary grid to', output_dir + '/_summary_grid.png')
            plt.tight_layout()
            plt.savefig(output_dir + '/_summary_grid.png', dpi=100)
            output_path = os.path.join(output_dir, '_summary_grid.png')
            print('done.')
            
            # Return original burst rate, burst rates, and output path
            # return original_burst_rate, burst_rates, output_path
            return output_path, clean_grid, original_burst_rate
        
        # reject unknown format options
        else:
            raise ValueError(f"Unknown format_option: {format_option}")    
    
    def plot_heat_maps(
        output_dir,
        input_dir,
        num_workers=None,
        levels=6,
        ):
        
        # subfuncs ===================================================================================================
        
        def plot_metric_heatmap(output_dir, metric_path, metric_name, network_metrics_data, clean_grid, levels):
            """
            Generalized function to plot heatmaps for a specified network metric.
            
            Args:
                output_dir (str): Directory to save the heatmap.
                metric_path (list): List of keys to navigate the metric in the network_metrics_data dictionary.
                metric_name (str): Name of the metric to display in the title and filename.
                network_metrics_data (dict): Dictionary containing network metrics data.
                clean_grid (dict): Dictionary of parameters and their data paths.
                levels (int): Number of levels for each parameter.
            """
            print(f"Plotting summary grid for {metric_name} with color gradient")
            
            # Find the original metric value - dict
            # for key in network_metrics_data.keys():
            #     if key.startswith('_'):
            #         original_key = key
            #         original_metric = network_metrics_data[key]['data']
            #         for path_part in metric_path:
            #             original_metric = original_metric[path_part]
            #         break
            
            #network_metrics_data_copy = deepcopy(network_metrics_data)
            
            #make list
            data_list = []
            for data in network_metrics_data:
                data_list.append(data['data'])
                
            # replace list, lazy
            #network_metrics_data = data_list                
            
            # patch for list
            for i, data in enumerate(data_list):
                sim_data_path = data['sim_data_path']
                base = os.path.basename(sim_data_path)
                if base.startswith('_'):
                    #if data.startswith('_'):
                    #original_key = data
                    #original_metric = network_metrics_data[data]['data']
                    #original_metric = data[key]
                    original_key = i
                    original_metric = data.copy()
                    for path_part in metric_path:
                        if 'network_metrics' in path_part: continue
                        original_metric = original_metric[path_part]
                    break
            
            # # Determine min and max metric values
            # metric_list = []  # Initialize list to store metric values
            # min_metric = float('inf')
            # max_metric = float('-inf')
            # for key in network_metrics_data.keys():
            #     data = network_metrics_data[key]['data']
            #     metric_value = data
            #     for path_part in metric_path:
            #         metric_value = metric_value[path_part]
            #         #print(metric_value)
            #     metric_list.append(float(metric_value))
            #     min_metric = min(min_metric, metric_value)
            #     max_metric = max(max_metric, metric_value)
            
            # patch for list
            metric_list = []  # Initialize list to store metric values
            min_metric = float('inf')
            max_metric = float('-inf')
            for data in data_list:
                #data = network_metrics_data[key]['data']
                metric_value = data.copy()
                for path_part in metric_path:
                    if 'network_metrics' in path_part: continue
                    try: metric_value = metric_value[path_part]
                    except:
                        #print(f"Error loading metric value for {path_part}")
                        metric_value = np.nan
                        continue
                metric_list.append(float(metric_value))
                min_metric = min(min_metric, metric_value)
                max_metric = max(max_metric, metric_value)
            
            # get min and max metric values within 2 std deviations to avoid outliers
            std_dev = np.std(metric_list)
            max_val = original_metric + 2 * std_dev
            min_val = original_metric - 2 * std_dev
            
            # now if min and max arre within 2 std deviations, use them, else use the std values
            min_metric = max(min_metric, min_val)
            max_metric = min(max_metric, max_val)
            
            # Define colormap and normalization
            colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
            cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
            
            # Handle the case where original_metric is NaN
            if not np.isnan(original_metric):
                #typical case
                norm = mcolors.CenteredNorm(vcenter=original_metric, halfrange=max(abs(min_metric - original_metric), abs(max_metric - original_metric)))
            else:
                # handle case where original_metric is NaN
                norm = mcolors.Normalize(vmin=min_metric, vmax=max_metric) # normalized without centering around original simulation
                
            # Prepare data dicts for clean_grid
            for param, summary_paths in clean_grid.items():
                clean_grid[param]['data'] = {}
                
            # Update clean_grid with network_metrics_data
            for param, summary_paths in clean_grid.items():
                #for key, data in network_metrics_data.items():
                for data in data_list: # patch for list
                    sim_data_path = data['sim_data_path']
                    base = os.path.basename(sim_data_path)
                    simLabel = os.path.basename(os.path.dirname(sim_data_path))
                    if param in base:
                        # skip og sim if it's based on param from previous SA
                        if base.startswith('_'): continue
                        clean_grid[param]['data'].update({simLabel: data})
            assert len(clean_grid[param]['data']) == levels, f'Expected {levels} levels, found {len(clean_grid[param]["data"])}'
                
            # Generate heatmap
            n_rows = len(clean_grid)
            n_cols = levels + 1
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1 * n_rows))
            
            for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                
                # # aw 2025-03-04 16:32:38 - since we know the number of levels we have, we can adjust what the middle index is
                row_data = clean_grid[param]['data']
                middle_idx = levels // 2 # for 6 levels, idx is 3 (0, 1, 2, 3, 4, 5, 6) - 3 is the middle (perm, perm, perm, og, perm, perm, perm)
                new_row_data = {}
                for idx, (key, data) in enumerate(clean_grid[param]['data'].items()):
                    # get correct idx for perm based on key. Each key should have a number value in it.
                    # find number in key (which should be a string including a number at the end)
                    # get the number from the key
                    #if idx != middle_idx: # perm case
                    level_pos = int(re.search(r'\d+$', key).group())
                    if level_pos >= middle_idx: level_pos += 1
                    #new_row_data[key] = data
                    new_row_data[level_pos] = data                    
                    # elif idx == middle_idx: # og case
                    #     #new_row_data['original_data'] = data_list[original_key]
                    #     new_row_data[idx] = data_list[original_key]
                    #new_row_data[key] = data
                new_row_data[middle_idx] = data_list[original_key]
                clean_grid[param]['data'] = new_row_data
                
                # aw 2025-03-04 16:31:39 updating this block of code above to handle incomplete lists of data - incase some sims or analyses fail.
                # row_data = clean_grid[param]['data']
                # sorted_row_data = dict(sorted(row_data.items()))
                # middle_idx = len(sorted_row_data) // 2
                # new_row_data = {}
                # for idx, (key, value) in enumerate(sorted_row_data.items()):
                #     if idx == middle_idx:
                #         new_row_data['original_data'] = data_list[original_key]
                #     new_row_data[key] = value
                # clean_grid[param]['data'] = new_row_data
                
                # Plot each cell in the row
                for col_idx, (key, data) in enumerate(clean_grid[param]['data'].items()):
                    try:
                        metric_value = data
                        for path_part in metric_path:
                            if 'network_metrics' in path_part: continue
                            try: metric_value = metric_value[path_part]
                            except: 
                                metric_value = np.nan
                                continue
                        color = cmap(norm(metric_value))
                        
                        # key is the real column index... #HACK
                        axs[row_idx, key].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
                        axs[row_idx, key].text(0.5, 0.5, f'{metric_value:.2f}', ha='center', va='center', fontsize=12)
                        axs[row_idx, key].axis('off')
                        
                        # axs[row_idx, col_idx].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
                        # axs[row_idx, col_idx].text(0.5, 0.5, f'{metric_value:.2f}', ha='center', va='center', fontsize=12)
                        # axs[row_idx, col_idx].axis('off')
                        permuted_param = param
                        #permuted_value = data['data']['simConfig'][param]
                        #permuted_value = data[param]
                        sim_data_path = data['sim_data_path']
                        for netmet in network_metrics_data:
                            sim_check = netmet['data']['sim_data_path']
                            if sim_data_path == sim_check:
                                permuted_value = netmet['cfg'][param]
                                break
                        #cfg = network_metrics_data[key]['cfg']
                        try:
                            permuted_value = round(permuted_value, 3)
                        except:
                            pass
                        axs[row_idx, key].set_title(f'@{permuted_value}', fontsize=14)
                    except Exception as e:
                        print(f"Error loading plot for key {key}: {e}")
                #print(f"Plotted {param} in row {row_idx}")

                # remove axes for all subplots, even if nothing plotted
                for col_idx in range(levels + 1):
                    axs[row_idx, col_idx].axis('off')
                    
            plt.tight_layout()
            plt.subplots_adjust(left=0.15, right=0.90, top=0.925)
            for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                pos = axs[row_idx, 0].get_position()
                x = pos.x0 - 0.025
                fig.text(x, pos.y0 + pos.height / 2, param, va='center', ha='right', fontsize=14, rotation=0)
            
            # Add colorbar
            # NOTE: sm generated with norm based on original_metric = nan will result in stack overrflow when trying to generate the colorbar - to deal with this,
            # to deal with this, norm has a special case above for when original_metric is nan. norm will be set to a norm that is not centered on original simulaiton value.
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
            fig.colorbar(sm, cax=cbar_ax)
            
            # Add title and save
            fig.suptitle(f'Heatmap: {metric_name}', fontsize=16)
            output_path = os.path.join(output_dir, f'_heatmap_{metric_name}.png')
            plt.savefig(output_path, dpi=100)
            pdf_path = os.path.join(output_dir, f'_heatmap_{metric_name}.pdf')
            plt.savefig(pdf_path)
            print(f'Saved heatmap to {output_path}')
            print(f'Saved heatmap to {pdf_path}')
            #print(f'Saved heatmap to {output_path}')
            return output_path
        
        # main ===================================================================================================
        # get dict of paths for matrix
        clean_grid = get_clean_grid(
            input_dir,
            query='network_data.npy'
            )
        
        # Collect network_metrics.npy files and process #NOTE: parallel processing is used here
        #if burst_rates is None or original_burst_rate is None:
        network_metrics_data = load_network_metrics(
            input_dir, 
            sim_data_path,
            num_workers=num_workers,
            )
                
        # # Metric paths of interest #TODO - I guess I could automate this by looking for any metric that resolves as a single value or something like that
            # metric_paths = [
            #     'network_metrics.simulated_data.MeanFireRate_E',
            #     'network_metrics.simulated_data.MeanFireRate_I',
            #     'network_metrics.simulated_data.CoVFireRate_E',
            #     'network_metrics.simulated_data.CoVFireRate_I',
            #     'network_metrics.simulated_data.MeanISI_E',
            #     'network_metrics.simulated_data.MeanISI_I',
            #     'network_metrics.simulated_data.CoV_ISI_E',
            #     'network_metrics.simulated_data.CoV_ISI_I',
                
            #     'network_metrics.spiking_data.spiking_summary_data.MeanFireRate',
            #     'network_metrics.spiking_data.spiking_summary_data.CoVFireRate',
            #     'network_metrics.spiking_data.spiking_summary_data.MeanISI',
            #     'network_metrics.spiking_data.spiking_summary_data.CoV_ISI',
                
            #     'network_metrics.bursting_data.bursting_summary_data.MeanWithinBurstISI',
            #     'network_metrics.bursting_data.bursting_summary_data.CoVWithinBurstISI',
            #     'network_metrics.bursting_data.bursting_summary_data.MeanOutsideBurstISI',
            #     'network_metrics.bursting_data.bursting_summary_data.CoVOutsideBurstISI',
            #     'network_metrics.bursting_data.bursting_summary_data.MeanNetworkISI',
            #     'network_metrics.bursting_data.bursting_summary_data.CoVNetworkISI',
            #     'network_metrics.bursting_data.bursting_summary_data.mean_IBI',
            #     'network_metrics.bursting_data.bursting_summary_data.cov_IBI',
            #     'network_metrics.bursting_data.bursting_summary_data.mean_Burst_Rate',
            #     'network_metrics.bursting_data.bursting_summary_data.mean_Burst_Peak',
            #     'network_metrics.bursting_data.bursting_summary_data.fano_factor',
            #     'network_metrics.bursting_data.bursting_summary_data.baseline',
                
            #     'network_metrics.mega_bursting_data.bursting_summary_data.MeanWithinBurstISI',
            #     'network_metrics.mega_bursting_data.bursting_summary_data.CoVWithinBurstISI',
            #     'network_metrics.mega_bursting_data.bursting_summary_data.MeanOutsideBurstISI',
            #     'network_metrics.mega_bursting_data.bursting_summary_data.CoVOutsideBurstISI',
            #     'network_metrics.mega_bursting_data.bursting_summary_data.MeanNetworkISI',
            #     'network_metrics.mega_bursting_data.bursting_summary_data.CoVNetworkISI',
            #     'network_metrics.mega_bursting_data.bursting_summary_data.mean_IBI',
            #     'network_metrics.mega_bursting_data.bursting_summary_data.cov_IBI',
            #     'network_metrics.mega_bursting_data.bursting_summary_data.mean_Burst_Rate',
            #     'network_metrics.mega_bursting_data.bursting_summary_data.mean_Burst_Peak',
            #     'network_metrics.mega_bursting_data.bursting_summary_data.fano_factor',
            #     'network_metrics.mega_bursting_data.bursting_summary_data.baseline',            
            # ] 
        # aw 2025-03-02 21:30:07 - okay, automating this now lol...
        metric_paths = []

        def find_metric_paths(data, parent_path="network_metrics"):
            """
            Recursively find paths that resolve to a single float/int 
            or a list of floats/ints, while treating dicts with only numeric keys as lists.
            """
            keys_to_ignore = [
                "std", 
                "cov",
                "median",
                "burst_ids",
                "burst_part",
                "burst_parts",
                ".data",
                "num_bursts", # burst rate is more informative with variable durations
                "unit_metrics",
                "gids",
                "spiking_metrics_by_unit",
                "spiking_times_by_unit",
                ".unit_metrics",
                "unit_types",
                
                ]
            metric_paths = set()

            def is_numeric_key_dict(d):
                """Check if all keys in the dictionary are numeric (i.e., should be treated as a list)."""
                return isinstance(d, dict) and all(re.match(r"^\d+$", str(k)) for k in d.keys())

            def recurse(d, path):
                if isinstance(d, dict):
                    if is_numeric_key_dict(d):  # Treat as a list and stop recursion
                        if not any(ign in path for ign in keys_to_ignore):          
                            print(f"Found metric path: {path}")
                            metric_paths.add(path)
                        return
                    for key, value in d.items():
                        new_path = f"{path}.{key}"
                        recurse(value, new_path)
                elif isinstance(d, (int, float)) or (isinstance(d, list) and all(isinstance(i, (int, float)) for i in d)):
                    if not any(ign in path for ign in keys_to_ignore):          
                        print(f"Found metric path: {path}")
                        metric_paths.add(path)

            for entry in data:  # Assuming network_metrics_data is a list of dicts
                recurse(entry, parent_path)

            return sorted(metric_paths)

        
        # metric paths
        list_network_data = []
        for data in network_metrics_data:
            try:
                list_network_data.append(data['data'])
            except Exception as e:
                print(f"Error loading network data: {e}")
                continue            
        metric_paths = find_metric_paths(list_network_data)
        print(f'{len(metric_paths)} metric paths found')
        
        # testing
        for metric_path in metric_paths:
            try:
                metric_path_parts = metric_path.split('.')
                # if any('bursting' in part for part in metric_path_parts):
                #     metric_name = f'{metric_path_parts[-3]}_{metric_path_parts[-1]}'
                # else:
                #metric_name = f'{metric_path_parts[-4]}_{metric_path_parts[-2]}_{metric_path_parts[-1]}'
                # just string all path parts together
                metric_name = '_'.join(metric_path_parts)
                plot_metric_heatmap(output_dir, metric_path_parts, metric_name, network_metrics_data, clean_grid, levels) #TODO: add reference data...
            except Exception as e:
                traceback.print_exc()
                print(f"Error plotting heatmap for {metric_path}: {e}")
                continue
            
        print('done.')

    # main ===================================================================================================
    
    # Set up paths and parameters
    #output_dir = sensitvity_analysis_output_dir
    # input_dir = os.path.join(sensitvity_analysis_output_dir, 'simulations')
    input_dir = sensitvity_analysis_output_dir
    output_dir = os.path.join(sensitvity_analysis_output_dir, 'summary_plots')
    sim_data_path = og_simulation_data_path
    
    # assertions
    assert os.path.exists(input_dir), f'Input directory {input_dir} does not exist.'
    assert os.path.exists(sim_data_path), f'Simulation data path {sim_data_path} does not exist.'
    
    # dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if plot_heatmaps:
        plot_heat_maps(
            output_dir,
            input_dir, 
            num_workers=num_workers,
            levels=levels)

def plot_sensitivity_analysis_v2(kwargs):
    
    #unpack kwargs    
    og_simulation_data_path = kwargs['sim_data_path']
    sensitvity_analysis_output_dir = kwargs['output_dir']
    num_workers = kwargs['max_workers']
    # burst_rates = kwargs['burst_rates']
    # original_burst_rate = kwargs['original_burst_rate']
    # format_option = kwargs['format_option']
    levels = kwargs['levels']
    plot_grid = kwargs['plot_grid']
    plot_heatmaps = kwargs['plot_heatmaps']
    
    plot_sensitivity_grid_plots_v2(
        og_simulation_data_path,
        sensitvity_analysis_output_dir,
        num_workers=num_workers,
        levels=levels,
        plot_grid=plot_grid,
        plot_heatmaps=plot_heatmaps,      
    )
    
def plot_permutations(perm_network_data, kwargs):
    ''' plot network metrics for each permutation '''
    # subfuncs ===================================================================================================
    
    # main ===================================================================================================
    # init
    indent_increase()
    for network_data in perm_network_data:
        try:
            kwargs['network_summary_mode'] = '2p' #2 pannel
            plot_network_summary_v2(network_data, **kwargs)
            kwargs['network_summary_mode'] = '3p' #3 pannel
            plot_network_summary_v2(network_data, **kwargs)
        except Exception as e:
            print(f'Error plotting network metrics: {e}')
            traceback.print_exc()
            continue
    # end func
    indent_decrease()
    return
    #raise NotImplementedError('plot_permutations not implemented yet.')

def map_cfg_to_netparams_v2(simConfig, netParams):
    """
    Map attributes in simConfig to their corresponding locations in netParams based on values.
    
    Parameters:
        simConfig (dict): The configuration dictionary (cfg).
        netParams (object): The network parameters object.
    
    Returns:
        dict: A mapping from simConfig parameters to their paths in netParams.
    """
    # subfuncs ===================================================================================================
    def deterimine_strategy(cfg_param):
        #
        #strategies = []
        #cfg_param in cfg_param_list:
        
        #Typical case:
        strategy = 'by_value'
        
        #SPECIAL CASES: gnabar, gkbar, L, diam, Ra
        handle_by_name = ['gnabar', 'gkbar', 'L', 'diam', 'Ra']
        if any([name in cfg_param for name in handle_by_name]): 
            if '_' in cfg_param:
                elements = cfg_param.split('_')
                for element in elements:
                    if any([name==element for name in handle_by_name]):
                        strategy = 'by_name'
                        break
            elif any([name==cfg_param for name in handle_by_name]):
                strategy = 'by_name'
        #strategies.append(strategy)
        
        return strategy
    
    def find_value_in_netparams(value, netParams, current_path=""):
        """
        Recursively search for the value in netParams and return a list of matching paths.
        
        Parameters:
            value (any): The value to search for.
            netParams (object): The network parameters object.
            current_path (str): The current path in the recursive search.
        
        Returns:
            list: A list of paths to the matching value.
        """
        stack = [(netParams, current_path)]  # Stack for backtracking, contains (current_object, current_path)
        matching_paths = []  # To store all matching paths

        while stack:
            
            try:
                current_obj, current_path = stack.pop()
                
                # if 'connParams' in current_path:  # Debugging: specific context output
                #     print('found connParams')
                #     if 'I->E' in current_path:
                #         print('found I->E')

                if isinstance(current_obj, dict):
                    for key, val in current_obj.items():
                        new_path = f"{current_path}.{key}" if current_path else key
                        if isinstance(val, (int, float)):
                            if val == value:
                                matching_paths.append(new_path)
                        elif isinstance(val, str):  # Handle HOC string matches
                            if str(value) in val:
                                matching_paths.append(new_path)
                        elif isinstance(val, (dict, list)):
                            stack.append((val, new_path))  # Push deeper layer onto stack
                        else:
                            print('Unhandled type:', type(val))
                            raise ValueError(f"Unhandled type: {type(val)}")
                            
                elif isinstance(current_obj, list):
                    for i, item in enumerate(current_obj):
                        new_path = f"{current_path}[{i}]"
                        if isinstance(item, (int, float)):
                            if item == value:
                                matching_paths.append(new_path)
                        elif isinstance(item, str):  # Handle HOC string matches
                            if str(value) in item:
                                matching_paths.append(new_path)
                        elif isinstance(item, (dict, list)):
                            stack.append((item, new_path))  # Push list item onto stack
                        else:
                            print('Unhandled type:', type(item))
                            raise ValueError(f"Unhandled type: {type(item)}")
                            
                else:
                    print('Unhandled type:', type(current_obj))
                    raise ValueError(f"Unhandled type: {type(current_obj)}")    
            except Exception as e:
                traceback.print_exc()
                print(f"Error in find_value_in_netparams: {e}")
                continue

        return matching_paths  # Return all matching paths
    
    def find_name_in_netparams(name, netParams, current_path=""):
        """
        Recursively search for the name in netParams and return a list of matching paths.
        
        Parameters:
            name (str): The name to search for.
            netParams (object): The network parameters object.
            current_path (str): The current path in the recursive search.
        
        Returns:
            list: A list of paths to the matching name.
        """
        stack = [(netParams, current_path)]
        
        if '_' in name:
            elements = name.split('_')
            try: assert 'E' in elements or 'I' in elements
            except: elements = None
        else:
            elements = None
        
        matching_paths = []
        while stack:
            current_obj, current_path = stack.pop()
            
            # if 'cellParams' in current_path:  # Debugging: specific context output
            #     print('found cellParams')
                
            # if 'gnabar' in current_path:  # Debugging: specific context output
            #     print('found gnabar')
                
            # elements=None
            # #if _ in 
            
            if elements is not None:
                if isinstance(current_obj, dict):
                    for key, val in current_obj.items():
                        new_path = f"{current_path}.{key}" if current_path else key
                        if all([element in new_path for element in elements]):
                            matching_paths.append(new_path)
                        elif isinstance(val, (dict, list)):
                            stack.append((val, new_path))
                elif isinstance(current_obj, list):
                    for i, item in enumerate(current_obj):
                        new_path = f"{current_path}[{i}]"
                        if all([element in new_path for element in elements]):
                            matching_paths.append(new_path)
                        elif isinstance(item, (dict, list)):
                            stack.append((item, new_path))
                
            elif isinstance(current_obj, dict):
                for key, val in current_obj.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    #if key == name:
                    if key == name:
                        matching_paths.append(new_path)
                    elif isinstance(val, (dict, list)):
                        stack.append((val, new_path))
            elif isinstance(current_obj, list):
                for i, item in enumerate(current_obj):
                    new_path = f"{current_path}[{i}]"
                    if item == name:
                        matching_paths.append(new_path)
                    elif isinstance(item, (dict, list)):
                        stack.append((item, new_path))
        return matching_paths
    
    # main ===================================================================================================
    # Determine strategy
    strategy = deterimine_strategy(list(simConfig.keys())[0])
       
    # Generate the mapping
    mapping = {}
    #for param, value in simConfig.items():
    for param, value in simConfig.items():
        if strategy == 'by_name':
            #paths = find_value_in_netparams(param, netParams)
            paths = find_name_in_netparams(param, netParams)
        elif strategy == 'by_value':
            #paths = find_name_in_netparams(value, netParams)
            paths = find_value_in_netparams(value, netParams)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        mapping[param] = paths if paths else None  # Assign None if no path is found

    return mapping, strategy

def try_load_sim(cfg):
    saveFolder = cfg['saveFolder']
    simLabel = cfg['simLabel']
    expected_save_path = os.path.join(saveFolder, f'{simLabel}_data.pkl')
    exists = os.path.exists(expected_save_path)
    if exists: 
        print(f'Simulation data for {simLabel} already exists at {expected_save_path}. Attempting to load...')
        try:
            sim.load(expected_save_path)
            assert hasattr(sim, 'net'), "Simulation data loaded but 'net' attribute is missing."
            assert hasattr(sim, 'cfg'), "Simulation data loaded but 'simConfig' attribute is missing."
            #assert cfg == sim.cfg, "Loaded simulation data does not match the expected configuration."
            if not cfg == sim.cfg:
                error_count = 0
                # iterate through each element of cfg and check if it's in sim.cfg
                for key, val in cfg.items():
                    # aw 2025-03-04 14:06:40 - I think we can ignore filename as long as simLabel and simFolder are correct.
                    if key in ['filename']: continue
                    try:
                        assert sim.cfg[key] == val
                    except:
                        #print(f'Expected {key} = {val} but got {sim.cfg[key]}')
                        print(f'Loaded value: {key} = {sim.cfg[key]}')
                        print(f'Expected value: {key} = {val}')
                        error_count += 1
                if error_count > 0: raise AssertionError("Loaded simulation data does not match the expected configuration.")
                else: pass
            # # aw 2025-03-01 16:33:00 - checking cfg in this way doesnt work here since we're loading before prep config
            assert hasattr(sim, 'simData'), "Simulation data loaded but 'simData' attribute is missing."
            assert hasattr(sim, 'allSimData'), "Simulation data loaded but 'allSimData' attribute is missing."
            print(f'Simulation data for {simLabel} loaded successfully.')
            return True, expected_save_path
        except Exception as e:
            print(f'Error loading simulation data for {simLabel}: {e}')
            print(f'Will attempt to run the simulation instead.')
            try: sim.clearAll()
            except: pass
    return False, None

def prepare_permuted_sim_v2(pkwargs):
    
    # subfuncs ===================================================================================================
    def getNestedParam(netParams, paramLabel):
        if '.' in paramLabel: 
            paramLabel = paramLabel.split('.')
        if isinstance(paramLabel, list ) or isinstance(paramLabel, tuple):
            container = netParams
            for ip in range(len(paramLabel) - 1):
                if hasattr(container, paramLabel[ip]):
                    container = getattr(container, paramLabel[ip])
                else:
                    container = container[paramLabel[ip]]
            return container[paramLabel[-1]]
    
    # main ===================================================================================================
    # unpack pkwargs
    sim_data_path = pkwargs.get('sim_data_path', None) # og_simulation_data_path
    cfg = pkwargs.get('cfg', None)
    cfg_param = pkwargs.get('cfg_param', None)
    cfg_val = pkwargs.get('cfg_val', None)
    # try_loading = pkwargs.get('try_loading', False)
    
    # #try loading sim data if possible
    # if try_loading:
    #     success, sim_data_path = try_load_sim(cfg)
    
    # load netparams and permute
    sim.load(sim_data_path, simConfig=cfg) # load og sim_data w/ permuted cfg
    simConfig = specs.SimConfig(simConfigDict=cfg)
    # # comment out later.
    # # # netParams_eioverride = sim.loadNetParams(
    # # #     "/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/netParams.py", 
    # # #     setLoaded=False
    # # #     )
    # _, netParams_eioverride = sim.readCmdLineArgs(
    #                 simConfigDefault="/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/cfg.py",
    #                 netParamsDefault="/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/netParams.py", 
    #             )
    # # comment out later
    
    netParams = sim.loadNetParams(sim_data_path, setLoaded=False)
    
    # # #comment out later
    # # #make strategic deits to netparams
    
    
    # # Extract relevant sections
    # cellParams_or = netParams_eioverride.cellParams  # Override version
    # popParams_or = netParams_eioverride.popParams
    # cellParams = netParams.cellParams  # Original version
    # popParams = netParams.popParams

    # # Define variables to track count
    # num_e_cells = len([cell for cell in cellParams if 'E' in cell])
    # num_i_cells = len([cell for cell in cellParams if 'I' in cell])

    # # Adjust the balance: remove excess E cells, add missing I cells
    # num_e_target = len([cell for cell in cellParams_or if 'E' in cell])
    # num_i_target = len([cell for cell in cellParams_or if 'I' in cell])

    # # Remove excess E cells
    # if num_e_cells > num_e_target:
    #     excess = num_e_cells - num_e_target
    #     cellParams = {k: v for i, (k, v) in enumerate(cellParams.items()) if 'E' not in k or i >= excess}

    # final_num_e_cells = len([cell for cell in cellParams if 'E' in cell])
    
    # # Add missing I cells from override
    # if num_i_cells < num_i_target:
    #     needed = num_i_target - num_i_cells
    #     #only add the first needed cells
    #     i_cells_to_add = {k: v for k, v in cellParams_or.items() if 'I' in k}
    #     i_cells_to_add = dict(list(i_cells_to_add.items())[:needed])
    #     cellParams.update(i_cells_to_add)
    
    # final_num_i_cells = len([cell for cell in cellParams if 'I' in cell])       

    # # Align cell lists in popParams
    # cell_lists_E_or = popParams_or['E']['cellsList']
    # cell_lists_I_or = popParams_or['I']['cellsList']
    # cell_lists_E = popParams['E']['cellsList']
    # cell_lists_I = popParams['I']['cellsList']

    # # Ensure `cellType` and `cellLabel` in `cellsList` match `cell_label` in `cellParams`
    # for pop_label, pop in popParams.items():
    #     if 'cellsList' in pop:
    #         to_remove = []  # Collect cells to remove
    #         for cell in pop['cellsList']:
    #             cell_id = cell.get('cellLabel', None)  # Get the label if it exists
    #             if cell_id not in cellParams:  # If it's not in cellParams, remove it
    #                 to_remove.append(cell)

    #         # Now remove items safely after iterating
    #         for cell in to_remove:
    #             pop['cellsList'].remove(cell)
    #             print(f"Removed {cell.get('cellLabel', 'Unknown')} from {pop_label} cellsList")

    # # Ensure all cells from popParams_or exist in popParams
    # for pop_label, pop in popParams_or.items():
    #     if 'cellsList' in pop:
    #         for cell in pop['cellsList']:
    #             cell_id = cell.get('cellLabel', None)  # Get the label
    #             if cell_id in cellParams and cell not in popParams[pop_label]['cellsList']:
    #                 popParams[pop_label]['cellsList'].append(cell)
    #                 print(f"Added {cell_id} to {pop_label} cellsList")
                    
                    
    # # finally update pop params with the correct num cells
    # for pop_label, pop in popParams.items():
    #     if 'numCells' in pop:
    #         if 'E' in pop_label:
    #             pop['numCells'] = final_num_e_cells
    #         elif 'I' in pop_label:
    #             pop['numCells'] = final_num_i_cells
    
    # comment out later
    
    cfg_to_netparams_mapping, strategy = map_cfg_to_netparams_v2({cfg_param: cfg_val}, netParams.__dict__.copy())
    mapped_paths = cfg_to_netparams_mapping[cfg_param]
    
    if mapped_paths is None:
        print(f"WARNING: mapped paths is None.")
        print(f"No paths found for {cfg_param} = {cfg_val}")
        return

    # update permuted params
    for mapped_path in mapped_paths:    
        current_val = getNestedParam(netParams, mapped_path)
        #assert cfg_val == current_val, f"Expected {cfg_val} but got {current_val}"
        try:
            if isinstance(current_val, str): #handle hoc strings
                assert str(cfg_val) in current_val, f"Expected {cfg_val} to be in {current_val}"
                updated_func = current_val.replace(str(cfg_val), str(cfg[cfg_param]))                    
                #netParams.setNestedParam(mapped_path, updated_func)
                before_val = getNestedParam(netParams, mapped_path)
                netParams.setNestedParam(mapped_path, updated_func)
                after_val = getNestedParam(netParams, mapped_path)
                assert before_val != after_val, f"Failed to update {mapped_path} from {before_val} to {after_val}"
                print(f"Updated {mapped_path} from {before_val} to {after_val}")
            elif strategy == 'by_name': #special case
                original_val = cfg_val
                permuted_val = cfg[cfg_param]
                modifier = permuted_val / original_val  # NOTE: this should end up equal to one of the level multipliers
                before_val = getNestedParam(netParams, mapped_path)
                netParams.setNestedParam(mapped_path, current_val * modifier)
                after_val = getNestedParam(netParams, mapped_path)
                assert before_val != after_val, f"Failed to update {mapped_path} from {before_val} to {after_val}"
                print(f"Updated {mapped_path} from {before_val} to {after_val}")
            else:
                assert cfg_val == current_val, f"Expected {cfg_val} but got {current_val}"
                before_val = getNestedParam(netParams, mapped_path)
                netParams.setNestedParam(mapped_path, cfg[cfg_param])
                after_val = getNestedParam(netParams, mapped_path)
                assert before_val != after_val, f"Failed to update {mapped_path} from {before_val} to {after_val}"
                print(f"Updated {mapped_path} from {before_val} to {after_val}")  
        except:
            print(f'Error updating {mapped_path}: {e}')
            continue

    # remove previous data
    sim.clearAll()

    #remove mapping from netParams #TODO: figure out how to actually take advantage of this
    # if hasattr(netParams, 'mapping'):
    #     del netParams.mapping
    netParams.mapping = {}

    # run simulation
    # Create network and run simulation
    sim.initialize(                     # create network object and set cfg and net params
            simConfig = simConfig,          # pass simulation config and network params as arguments
            netParams = netParams)
    sim.net.createPops()                # instantiate network populations
    sim.net.createCells()               # instantiate network cells based on defined populations
    sim.net.connectCells()              # create connections between cells based on params
    sim.net.addStims()                  # add stimulation (usually there are none)
    sim.setupRecording()                # setup variables to record for each cell (spikes, V traces, etc)

def run_permutation_v2(simLabel, simFolder, cfg, cfg_param, cfg_val, tkwargs):
    
    # subfuncs ===================================================================================================
                
    # main ===================================================================================================
    # init
    #simLabel = cfg['simLabel']
    print(f'Running permutation {simLabel}...')
    indent_increase()
    
    try:
    
        # unpack tkwargs
        try_loading = tkwargs.get('try_load_sim_data', True)
        conv_params = tkwargs.get('conv_params', None)
        mega_params = tkwargs.get('mega_params', None)
        
        #assert reference_data_path is not None, "Reference data path must be provided for plotting."
        assert conv_params is not None, "Conversion parameters must be provided for plotting."
        assert mega_params is not None, "Mega parameters must be provided for plotting."

        # prep pkwargs for prepare_permuted_sim
        pkwargs = {
            'sim_data_path': tkwargs['sim_data_path'], #og_simulation_data_path,
            'simLabel': simLabel,
            'simFolder': simFolder, # simLabel + simFolder = new sim path
            'cfg_param': cfg_param,
            'cfg_val': cfg_val,
            'cfg': cfg,
            }
        
        ## load or run simulation
        
        #logic
        run_og_sim = cfg_param is None and cfg_val is None #bool, if True, run the original simConfig
        run_perm = cfg_param is not None and cfg_val is not None
        sim_loaded = False # init success flag
        if try_loading: sim_loaded, expected_save_path = try_load_sim(cfg) # try loading sim data
        
        # run permutation
        if run_perm and not sim_loaded: #if none, then it's the original simConfig            
            
            # prepare and run simulation permutation
            #if not sim_loaded:  # if loading fails or is not attempted, prepare and run simulation   
            prepare_permuted_sim_v2(pkwargs)          
            sim.runSim()                        # run parallel Neuron simulation
            sim.gatherData()                    # gather spiking data and cell info from each node
            
            # save data        
            permuted_data_paths = sim.saveData()                      # save params, cell info and sim output to file (pickle,mat,txt,etc)
            assert len(permuted_data_paths) == 1, "Expected only one data path, the .pkl file. Got more."
            perm_sim_data_path = permuted_data_paths[0]
        elif run_og_sim and not sim_loaded: #if both none, rerun the original simConfig (with new duration plausibly)
            
            # load and run original simulation
            og_sim_data_path = tkwargs['sim_data_path']
            sim.load(og_sim_data_path, simConfig=cfg)
            sim.runSim()                        # run parallel Neuron simulation
            sim.gatherData()                    # gather spiking data and cell info from each node
            
            # save data        
            permuted_data_paths = sim.saveData()                      # save params, cell info and sim output to file (pickle,mat,txt,etc)
            assert len(permuted_data_paths) == 1, "Expected only one data path, the .pkl file. Got more."
            perm_sim_data_path = permuted_data_paths[0]
        elif sim_loaded: 
            perm_sim_data_path = expected_save_path  # if sim loaded, use the expected save path
        else: 
            print('Not sure how you got here... Something went wrong.')
            raise ValueError('Invalid cfg_param and cfg_val combination.')
        
        # end func    
        indent_decrease()    
        print(f'Permutation {simLabel} successfully ran!')
        assert perm_sim_data_path is not None, "Permutation data path is None."
        return perm_sim_data_path
    
    except Exception as e:
        print(f'Error running permutation {simLabel}: {e}')
        traceback.print_exc()
        indent_decrease()
        return e  

def run_permutations_v2(kwargs):
    """
    Run all configuration permutations in parallel, limited by logical CPU availability.
    """
    # subfuncs ===================================================================================================
    def run_perms(tkwargs):
        #unpack tkwargs
        debug_mode = tkwargs.get('debug_mode', False)
        num_workers = tkwargs.get('num_workers', None)
        
        # force debug for now..
        # debug_mode = True
        
        # Prepare tasks
        tasks = []
        for cfg_tuple in cfg_permutations:
            if isinstance(cfg_tuple, tuple) and len(cfg_tuple) == 5: # aw 2025-03-04 12:55:40 added simLable and simFolder
            #if isinstance(cfg_tuple, tuple) and len(cfg_tuple) == 3:
                #cfg, cfg_param, cfg_val = cfg_tuple
                simLable, simFolder, cfg, cfg_param, cfg_val = cfg_tuple
                #tasks.append((cfg, cfg_param, cfg_val, tkwargs))
                tasks.append((simLable, simFolder, cfg, cfg_param, cfg_val, tkwargs))
            else: raise ValueError(f"Unexpected structure in cfg_permutations: {cfg_tuple}")
            
        # if debug mode, only run the first five tasks
        if debug_mode: tasks = tasks[:5]
        
        # if tasks is less than num_workers, set num_workers to len(tasks)
        if len(tasks) < num_workers: num_workers = len(tasks)

        # prepare for parallel processing
        # for debug
        #num_workers = 1

        # Evenly distribute threads among processes
        # NOTE: idek if this is necessary, but it's here.
        threads_per_worker = max(1, available_cpus // num_workers)
        os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
        os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)
        
        # Run tasks in parallel
        print(f'Running {len(tasks)} permutations...')
        # Print number of workers
        print(f'Using {num_workers} workers out of {available_cpus} available CPUs.')
        permuted_paths = [] # collect pkl data paths from each permutation
        #num_workers = 2 # DEBUG - force to 1
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(run_permutation_v2, *task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try: 
                    result = future.result()  # This will raise any exceptions from the worker
                    permuted_paths.append(result)
                except Exception as e:
                    cfg = task[1]  # Access the configuration from the task
                    sim_label = cfg.get("simLabel", "unknown") if isinstance(cfg, dict) else "unknown"
                    print(f"Unhandled exception in permutation {sim_label}: {e}")
                    traceback.print_exc()
                    permuted_paths.append(e)
        print(f'Permutations complete!')
        
        return permuted_paths
    
    def compute_netmets(permuted_data_paths, tkwargs):
        perm_network_data = []
        completed = 0
        failed = 0
        for path in permuted_data_paths:
            try:
                #loading
                print(f'Computing network metrics for {path}...')
                print(f'Loading...')
                
                # logic
                expected_network_data_path = path.replace('_data.pkl', '/network_data.npy')
                exists = os.path.exists(expected_network_data_path)
                try_load_network = tkwargs.get('try_load_network_data', False)
                try_load_plots = tkwargs.get('try_load_network_summary', False)
                expected_plots_path = path.replace('_data.pkl', '/network_summary_*')
                network_data_loaded = False # init success flag
                summary_plots_exist = False # init success flag
                
                # check for 2p and 3p network summary plots, png and pdf
                exist_plots_bool = []
                for suffix in ['2p', '3p']:
                    for ext in ['png', 'pdf']:
                        expected_plot_path = expected_plots_path.replace('*', suffix) + f'.{ext}'
                        simLabel = os.path.basename(os.path.dirname(path))
                        expected_plot_path = expected_plot_path.replace(f'/{simLabel}/{simLabel}', f'/{simLabel}') # HACK: just shitty code
                        plot_exists = os.path.exists(expected_plot_path)
                        
                        #print expected plot path and if it exists
                        print(f'Checking for {expected_plot_path}...')
                        print(f'Exists: {plot_exists}')
                        
                        
                        if plot_exists: exist_plots_bool.append(True)
                        else: exist_plots_bool.append(False)
                if all(exist_plots_bool): summary_plots_exist = True
                
                # try loading network data
                if try_load_network and exists:
                    try:
                        print(f'Network data for {path} already exists at {expected_network_data_path}. Attempting to load...')
                        network_data = np.load(expected_network_data_path, allow_pickle=True)
                        network_data = network_data.item()
                        print(f'Network data for {path} loaded successfully.')                            
                        perm_network_data.append(network_data)
                        network_data_loaded = True
                    except Exception as e:
                        traceback.print_exc()
                        print(f'Error loading network data for {path}: {e}')
                        print(f'Will attempt to compute network metrics instead.')
                        pass
                    
                # continue?
                print(f'Checking for existing network metrics and summary plots...')
                print(summary_plots_exist)
                print(try_load_plots)
                if network_data_loaded and summary_plots_exist and try_load_plots: 
                    print(f'Network metrics and summary plots already exist for {path}. Skipping...')
                    completed += 1
                    continue
                              
                # do computations?
                if not network_data_loaded:
                    source = 'simulated'
                    sim.clearAll()
                    sim.load(path)
                    
                    #unpack tkwargs
                    conv_params = tkwargs['conv_params']
                    mega_params = tkwargs['mega_params']
                    nkwargs = {
                        'simData': sim.allSimData,
                        'popData': sim.net.allPops,
                        'cellData': sim.net.allCells,
                        'run_parallel': True,
                        'debug_mode': False,
                        'max_workers': tkwargs.get('max_workers', None),
                        'sim_data_path': path,
                        'burst_sequencing': tkwargs.get('burst_sequencing', False),
                        'dtw_matrix': tkwargs.get('dtw_matrix', False),
                    }
                    network_data = compute_network_metrics(conv_params, mega_params, source, **nkwargs)
                    #perm_dir = os.path.dirname(path)
                    #remove the .pkl extension
                    perm_dir = path.replace('_data.pkl', '')
                    if not os.path.exists(perm_dir): os.makedirs(perm_dir)
                    perm_path = os.path.join(perm_dir, 'network_data.npy')
                    np.save(perm_path, network_data)
                    print(f'Computed network metrics for {path}')
                    print(f'Saved network metrics to {perm_path}')
                
                # plot network summary?
                if not summary_plots_exist or not try_load_plots:
                    temp_perm_network_data_list = []
                    temp_perm_network_data_list.append(network_data)
                    plot_permutations(temp_perm_network_data_list, tkwargs)
                
                ## append
                # # aw 2025-03-01 17:42:54 is deep copy needed? #TODO
                #network_copy = deepcopy(network_data)
                #perm_network_data.append(network_copy)
                
                #
                perm_network_data.append(network_data)
                completed += 1
            except Exception as e:
                print(f'Error computing network metrics for {path}: {e}')
                traceback.print_exc()
                perm_network_data.append(e)
                failed += 1
                
            #print()
            print(f'Completed: {completed}')
            print(f'Failed: {failed}')
            print(f'Remain: {len(permuted_data_paths) - completed - failed}')
            print()
                
        return perm_network_data    
    
    # main ===================================================================================================    
    #init
    indent_increase()
    available_cpus = os.cpu_count()
    num_workers = kwargs.get('max_workers', None)
    cfg_permutations = kwargs.get('cfg_permutations', None)
    if cfg_permutations is None: raise ValueError('cfg_permutations not in kwargs')
    if num_workers is None: num_workers = min(len(cfg_permutations), available_cpus)
    else: num_workers = min(num_workers, len(cfg_permutations))
    
    # prep tkwargs - task kwargs
    tkwargs = {
        'sim_data_path': kwargs.get('sim_data_path', None),
        'reference_data_path': kwargs.get('reference_data_path', None),
        'output_dir': kwargs.get('output_dir', None),
        'plot_permutations': kwargs.get('plot_permutations', False),
        'debug_mode': kwargs.get('debug_mode', False),
        'conv_params': kwargs.get('conv_params', None),
        'mega_params': kwargs.get('mega_params', None),
        'fitnessFuncArgs': kwargs.get('fitnessFuncArgs', None),
        #'try_loading': kwargs.get('try_loading', True),
        #'try_load_sim_data': kwargs.get('try_load_sim', True),
        'try_load_sim_data': kwargs.get('try_load_sim_data', False),
        'try_load_network_data': kwargs.get('try_load_network_data', False),
        'try_load_network_summary': kwargs.get('try_load_network_summary', False),
        'num_workers': num_workers,
        'max_workers': kwargs.get('max_workers', None),
        'burst_sequencing': kwargs.get('burst_sequencing', False),
        'dtw_matrix': kwargs.get('dtw_matrix', False),
        }

    # main processing steps
    permuted_data_paths = run_perms(tkwargs) # run permutations in parallel
    perm_network_data = compute_netmets(permuted_data_paths, tkwargs) # compute network metrics for each permutation
    
    # plot permutations
    # plot = kwargs.get('plot_permutations', False)
    # if plot:
    #     print('Plotting permutations...')
    #     pkwargs = kwargs.copy()
    #     # pkwargs['permuted_paths'] = permuted_data_paths
    #     pkwargs['perm_network_data'] = perm_network_data
    #     plot_permutations(perm_network_data, pkwargs)   
    
    # end func
    indent_decrease()
    print('All permutations complete!')

def generate_permutations_v2(kwargs):
    """ generate permutations for sensitivity analysis """
    # subfuncs ===================================================================================================
    def generate(simConfig, kwargs):
        # subfuncs ===================================================================================================
                    
        # main ===================================================================================================
        # init
        print('Generating permutations...')
        indent_increase()
        evol_params = kwargs['evol_params']
        verbose = kwargs.get('verbose', False)
        
        # generate permutations
        for evol_param, evol_val in evol_params.items():
            if hasattr(simConfig, evol_param):
                cfg_param = evol_param
                cfg_val = getattr(simConfig, evol_param)                
                if isinstance(evol_val, list): #if evol_val is a list, then it's a range from min to max allowed for the parameter
                    
                    #skip over certain params
                    excepted_param_keys = [
                        'std', # see rationale in notes at top of file
                        ]
                    if any([key in cfg_param for key in excepted_param_keys]):
                        if verbose: print(f'Skipping permutations for {cfg_param}...')
                        continue
                    
                    # generate permutations
                    print(f'Generating permutations for {cfg_param}...')
                    cfg_permutations = append_permutation_levels(cfg_param, cfg_val, simConfig, kwargs)                    
                    #print(f'Permutations generated for {cfg_param}!')
                    
        return cfg_permutations
    
    def append_permutation_levels(cfg_param, cfg_val, simConfig, kwargs):
        """
        Append permutations to the list of cfg permutations for a given parameter.
        """
        #init
        #print('Appending permutations...')
        indent_increase()
        levels = kwargs['levels']
        upper_bound = kwargs.get('upper_bound', 1.8)
        lower_bound = kwargs.get('lower_bound', 0.2)
        output_dir = kwargs.get('output_dir', None)
        if output_dir is None: raise ValueError('output_dir not in kwargs')
        
        #print(f'Generating levels for {cfg_param}...')
        for i in range(levels):
            original_cfg = simConfig.__dict__.copy()
            cfg_permutation = simConfig.__dict__.copy()
            #temp_upper, temp_lower = upper_bound, lower_bound # save original values for future iterations
            cfg_permutation, permuted_vals = permute_param(cfg_permutation, cfg_param, cfg_val, upper_bound, lower_bound, i, levels)
            cfg_permutation['simLabel'] = f'{cfg_param}_{i}'
            cfg_permutation['saveFolder'] = os.path.abspath(os.path.join(output_dir, cfg_permutation['simLabel']))
            cfg_simLabel = cfg_permutation['simLabel']
            cfg_saveFolder = cfg_permutation['saveFolder']
            cfg_permutations.append((
                cfg_simLabel,
                cfg_saveFolder,
                cfg_permutation,
                cfg_param,
                cfg_val,
                ))
            
            # save as json
            # sim.saveJson(cfg_permutation, cfg_permutation['saveFolder'])
            cfg_path = os.path.join(cfg_permutation['saveFolder'], cfg_permutation['simLabel'] + '_cfg.json')
            cfg_dir = os.path.dirname(cfg_path)
            if not os.path.exists(cfg_dir): os.makedirs(cfg_dir)
            sim.saveJSON(cfg_path, cfg_permutation)
            
            # quality
            assert cfg_permutation[cfg_param] == permuted_vals[i], f'Failed to permute {cfg_param} to {permuted_vals[i]}'
            assert original_cfg != cfg_permutation, f'Failed to permute {cfg_param} to {permuted_vals[i]}'
                   
        # end func
        indent_decrease()
        return cfg_permutations
    
    def permute_param(cfg_permutation, cfg_param, cfg_val, upper_bound, lower_bound, level, levels):                         
        """
        Handle special cases where the parameter should not be permuted.
        """
        # if verbose: print(f'Skipping permutations for {cfg_param}...')
        # return cfg_permutations
        # special cases
        #print(f'Generating levels for {cfg_param}...')
        if 'LengthConst' in cfg_param:
            # got to typical case, this isnt actually a probability based param
            # TODO: rename this to just length constant later.
            upper_value = cfg_val * upper_bound
            lower_value = cfg_val * lower_bound                       
        elif 'prob' in cfg_param:
            #modify upper and lower bounds such that probability based params 
            #dont go below 0 or above 1
            upper_value = cfg_val * upper_bound
            lower_value = cfg_val * lower_bound
            if upper_value > 1:
                upper_value = 1
            if lower_value < 0:
                lower_value = 0
            
            # #calculate new upper and lower bounds to be used in the permutations
            # upper_bound = 1 / cfg_val
            # lower_bound = 0 / cfg_val
        else:
            #typical case
            upper_value = cfg_val * upper_bound
            lower_value = cfg_val * lower_bound
            
        # do two linspaces and stitch them together to ensure cfg_val is centered.
        permuted_vals_1 = np.linspace(lower_value, cfg_val, levels // 2 + 1)[:-1] # returns all but the last value (exclude cfg_val)
        permuted_vals_2 = np.linspace(cfg_val, upper_value, levels // 2 + 1)[1:] ## returns all but the first value (exclude cfg_val)
        permuted_vals = np.concatenate((permuted_vals_1, permuted_vals_2))
        
        # quality
        assert permuted_vals.size == levels, f'Expected {levels} permuted values, got {permuted_vals.size}'
        assert np.all(np.diff(permuted_vals) > 0), f'Permuted values are not in ascending order: {permuted_vals}'
        assert cfg_val > permuted_vals[levels//2-1], f'Permuted value {permuted_vals[levels//2-1]} is not less than original value {cfg_val}'
        assert cfg_val < permuted_vals[levels//2], f'Permuted value {permuted_vals[levels//2]} is not greater than original value {cfg_val}'
            
        # return permuted value
        #permuted_vals = np.linspace(lower_value, upper_value, levels)
        permuted_val = permuted_vals[level]
        cfg_permutation[cfg_param] = permuted_val
        return cfg_permutation, permuted_vals
    
    # main ===================================================================================================
    
    # init
    indent_increase()
    sim_data_path = kwargs['sim_data_path']
    cfg_permutations = []
    saveFolder = kwargs['output_dir']
    duration_seconds = kwargs['duration_seconds']
    evol_params = kwargs['evol_params']    
    
    #apparently need to modify simcfg before loading
    simConfig = sim.loadSimCfg(sim_data_path, setLoaded=False)
    simLabel = simConfig.simLabel
    simLabel = '_'+simLabel # add underscore to simLabel so that original sim is easy to find in file system - 
                            # NOTE: this is only applied to the original simConfig - only because it isnt overwritten later when generating permutations.
    saveFolder = os.path.join(saveFolder, simLabel)
    
    #modify shared runtime options
    simConfig.simLabel = simLabel
    simConfig.saveFolder=saveFolder
    simConfig.duration = 1e3 * duration_seconds  # likewise, I think it's necessary to modify netParams, not net.params or net
    simConfig.verbose = kwargs.get('verbose', False) # NOTE: during connection formation, this will be VERY verbose
    simConfig.validateNetParams = kwargs.get('validateNetParams', True)
    simConfig.coreneuron = kwargs.get('use_coreneuron', False)
    simConfig.saveJson = kwargs.get('saveJson', False)
    simConfig.savePickle = kwargs.get('savePickle', True)
    simConfig.cvode_active = kwargs.get('cvode_active', False) # make sure variable time step is off...not sure why it was ever on.
    
    # turn recordings off - if any are present
    remove_recordCells = kwargs.get('remove_recordCells', True) # this just save time.
    if remove_recordCells: # remove recordCells from simConfig
        if hasattr(simConfig, 'recordCells'):
            delattr(simConfig, 'recordCells')
    
    # append original simConfig to permutations so that it is also run, plot, and saved with the others.
    # structure is maintained as a tuple to match the structure of the other permutations
    cfg_permutations.append((
                            simConfig.simLabel,
                            simConfig.saveFolder,
                            simConfig.__dict__.copy(), #cfg
                            None, #permuted param
                            None, #original value
                            ))
    
    # generate permutations
    perm_cfgs = generate(simConfig, kwargs)
    cfg_permutations.extend(perm_cfgs)
    
    # end func
    print(f'Generated {len(cfg_permutations)} cfg permutations.')
    indent_decrease()
    return cfg_permutations

def run_sensitivity_analysis_v2(kwargs):
    # subfuncs ===================================================================================================
    def validate_inputs(kwargs):
        # validate inputs
        assert 'sim_data_path' in kwargs, 'sim_data_path not in kwargs'
        assert 'output_dir' in kwargs, 'output_dir not in kwargs'
        assert 'reference_data_path' in kwargs, 'reference_data_path not in kwargs'
        assert 'run_analysis' in kwargs, 'run_analysis not in kwargs'
        assert 'plot_analysis' in kwargs, 'plot_analysis not in kwargs'
        assert 'plot_grid' in kwargs, 'plot_grid not in kwargs'
        assert 'plot_heatmaps' in kwargs, 'plot_heatmaps not in kwargs'
        assert 'levels' in kwargs, 'levels not in kwargs'
        assert 'conv_params' in kwargs, 'conv_params not in kwargs'
        assert 'mega_params' in kwargs, 'mega_params not in kwargs'
        #assert 'fitnessFuncArgs' in kwargs, 'fitnessFuncArgs not in kwargs'
        assert 'max_workers' in kwargs, 'max_workers not in kwargs'
        assert 'run_parallel' in kwargs, 'run_parallel not in kwargs'
        assert 'duration_seconds' in kwargs, 'duration_seconds not in kwargs'
        #assert 'try_loading' in kwargs, 'try_loading not in kwargs'
        
        # aw 2025-03-01 16:23:45 replacing try_loading with try_load_sim_data and try_load_** anything else
        assert 'try_load_sim_data' in kwargs, 'try_load_sim_data not in kwargs'
        assert 'try_load_network_summary' in kwargs, 'try_load_network_summary not in kwargs'
        
        assert 'debug_mode' in kwargs, 'debug_mode not in kwargs'
        
        # validate types
        assert isinstance(kwargs['sim_data_path'], str), 'sim_data_path must be a string'
        assert isinstance(kwargs['output_dir'], str), 'output_dir must be a string'
        assert isinstance(kwargs['reference_data_path'], str), 'reference_data_path must be a string'
        assert isinstance(kwargs['run_analysis'], bool), 'run_analysis must be a boolean'
        assert isinstance(kwargs['plot_analysis'], bool), 'plot_analysis must be a boolean'
        assert isinstance(kwargs['plot_grid'], bool), 'plot_grid must be a boolean'
        assert isinstance(kwargs['plot_heatmaps'], bool), 'plot_heatmaps must be a boolean'
        assert isinstance(kwargs['levels'], int), 'levels must be an integer'
        assert isinstance(kwargs['conv_params'], dict), 'conv_params must be a dictionary'
        assert isinstance(kwargs['mega_params'], dict), 'mega_params must be a dictionary'
        #assert isinstance(kwargs['fitnessFuncArgs'], dict), 'fitnessFuncArgs must be a dictionary'
        assert isinstance(kwargs['max_workers'], int), 'max_workers must be an integer'
        assert isinstance(kwargs['run_parallel'], bool), 'run_parallel must be a boolean'
    
    def label_pdf(pdf_path, perm_label):
        """
        Adds a bookmark with `perm_label` to each page of the given PDF.
        Adds text at the bottom right of each page with the given `perm_label`.
        """
        try:
            # Step 1: Add Bookmarks using PyPDF2
            input_pdf = PdfReader(pdf_path)
            output_pdf = PdfWriter()

            for i, page in enumerate(input_pdf.pages):
                output_pdf.add_page(page)
                output_pdf.add_outline_item(title=perm_label, page_number=i)

            # Write a temporary PDF with bookmarks (without text yet)
            temp_pdf_path = pdf_path.replace(".pdf", "_temp.pdf")
            with open(temp_pdf_path, "wb") as outputStream:
                output_pdf.write(outputStream)

            # Step 2: Open temp PDF and add text labels using PyMuPDF
            doc = fitz.open(temp_pdf_path)

            for page in doc:
                #text_position = (500, page.rect.height - 30)  # Bottom-right corner
                # get the size of the page
                page_width = page.rect.width
                page_height = page.rect.height
                # calculate the position
                text_position = (page_width - 100, page_height - 10)  # Bottom-right corner
                page.insert_text(text_position, perm_label, fontsize=14, fontname="helv", color=(0, 0, 0), fontfile="helvB")

            # Save the modified PDF
            output_pdf_path = pdf_path.replace(".pdf", "_labeled.pdf")
            doc.save(output_pdf_path)
            doc.close()

            # Remove temp file
            os.remove(temp_pdf_path)
            
            #return labeled path
            return output_pdf_path

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            
            return None

    def collect_and_combine_pdfs(output_pdf_path, **kwargs):
        """
        Finds all PDFs in `sim_output_dir` that contain '3p' in their name,
        labels them with their parent directory, and combines them into one PDF.
        """
        # unpack kwargs
        sim_output_dir = kwargs.get('sim_output_dir', None)
        
        
        combined_pdf = PdfWriter()
        
        output_dir = os.path.dirname(output_pdf_path)
        
        # delete any existing labeled PDFs - avoid double labeling
        for root, dirs, files in os.walk(output_dir):            
            # delete any pdfs with 'labeled' in the name
            for file in files:
                if "labeled" in file and file.endswith(".pdf"):
                    os.remove(os.path.join(root, file))
            
        for root, dirs, files in os.walk(output_dir):
            # sort dirs alphabetically so that original sim is first
            # also rlated files will be grouped together
            #dirs.sort(key=lambda x: x.lower())
            
            sorted_files = sorted(files, key=lambda x: x.lower())
            
            for file in sorted_files:
                if "3p" in file and file.endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    perm_label = os.path.basename(root)  # Parent directory name

                    # Label each page in the individual PDF
                    labeled_path = label_pdf(pdf_path, perm_label)

                    # Add the labeled PDF to the combined document
                    try:
                        #input_pdf = PdfReader(pdf_path)
                        input_pdf = PdfReader(labeled_path)
                        for page in input_pdf.pages:
                            combined_pdf.add_page(page)
                            print(f"Added {labeled_path} to combined PDF")
                    except Exception as e:
                        print(f"Error adding {pdf_path} to combined PDF: {e}")

        # Write the combined PDF
        if combined_pdf.pages:
            with open(output_pdf_path, "wb") as outputStream:
                combined_pdf.write(outputStream)
            print(f"Combined PDF saved to {output_pdf_path}")
        else:
            print("No valid PDFs found to combine.")
    
    # main ===================================================================================================
    
    # validate inputs
    validate_inputs(kwargs)
    
    #init paths
    sim_output_dir = os.path.join(kwargs['output_dir'], 'simulations')
    if not os.path.exists(sim_output_dir): os.makedirs(sim_output_dir)
    kwargs['sim_output_dir'] = sim_output_dir
    
    # main analysis steps
    cfg_permutations = generate_permutations_v2(kwargs) # generate permutations
    kwargs['cfg_permutations'] = cfg_permutations # add to kwargs
    run_permutations_v2(kwargs)  # run permutations
    
    # combine PDFs
    output_pdf_path = os.path.join(kwargs['output_dir'], 'combined_permutations.pdf')
    collect_and_combine_pdfs(output_pdf_path, **kwargs)
    
    
    # # collect all pdfs in output dir with 3p in the name, make one pdf with all of them.
    # # label each page with the perm_label wich should be the parent dir of the pdf path
    # for root, dirs, files in os.walk(sim_output_dir):
    #     for file in files:
    #         if '3p' in file and '.pdf' in file:
    #             pdf_path = os.path.join(root, file)
    #             perm_label = os.path.basename(root)
    #             label_pdf(pdf_path, perm_label)
                
''' older functions'''
# old code being phased out ======================================================================================================================
'''run sensitivity analysis'''
def run_sensitivity_analysis(
    
    # input parameters
    sim_data_path, 
    output_path,
    reference_data_path = None,
    
    #plotting parameters
    plot = False,
    conv_params = None,
    mega_params = None,
    fitnessFuncArgs = None,
    
    #sensitivity analysis parameters
    lower_bound=0.2,
    upper_bound=1.8,
    levels=2,
    duration_seconds=1,
    option='serial', #NOTE: options are 'serial' or 'parallel'
    num_workers=None, #NOTE: specify number of workers for parallel option, if None, will allocate as many as possible and distribute threads evenly
    debug=False,
    try_loading=True,
    ):
    
    #simulation output path
    output_path = os.path.join(output_path, 'simulations')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # generate permutations
    def generate_permutations(
        sim_data_path, 
        #evol_params_path, 
        saveFolder,
        lower_bound=0.2, 
        upper_bound=1.8, 
        levels=2,
        duration_seconds=1,
        verbose = False
        ):
        #from copy import deepcopy
        #from netpyne import sim    
        
        #cfg_permutations
        cfg_permutations = []
        
        #apparently need to modify simcfg before loading
        simConfig = sim.loadSimCfg(sim_data_path, setLoaded=False)
        
        #modify shared runtime options
        duration_seconds = duration_seconds
        simConfig.duration = 1e3 * duration_seconds  # likewise, I think it's necessary to modify netParams, not net.params or net
        simConfig.verbose = False
        #simConfig.verbose = True # NOTE: during connection formation, this will be VERY verbose
        simConfig.validateNetParams = True
        #simConfig.coreneuron = True
        simConfig.saveFolder=saveFolder
        simConfig.saveJson = False
        simConfig.savePickle = True
        #simConfig.coreneuron = True
        simConfig.cvode_active = False # make sure variable time step is off...not sure why it was ever on.
        simConfig.simLabel = '_'+simConfig.simLabel # NOTE this is only applied to the original simConfig 
                                                    # - only because it isnt overwritten later when generating permutations.
        
        # turn recordings off
        # remove recordCells from simConfig
        if hasattr(simConfig, 'recordCells'):
            delattr(simConfig, 'recordCells')
        
        # append original simConfig to permutations so that it is also run, plot, and saved with the others.
        # structure is maintained as a tuple to match the structure of the other permutations
        cfg_permutations.append((
            simConfig.__dict__.copy(), #cfg
            None, #permuted param
            None, #original value
            ))
        
        # load evol_params
        # evol_params_module = import_module_from_path(evol_params_path)
        # evol_params = evol_params_module.params
        #from DIV21.src.evol_params import params
        from RBS_network_models.CDKL5.DIV21.src.evol_params import params
        evol_params = params
        
        # generate permutations
        for evol_param, evol_val in evol_params.items():
            # for cfg_param, cfg_val in simConfig.items():
            #     if evol_param == cfg_param:
            if hasattr(simConfig, evol_param):
                cfg_param = evol_param
                cfg_val = getattr(simConfig, evol_param)
                #if evol_val is a list, then it's a range from min to max allowed for the parameter
                if isinstance(evol_val, list):
                    
                    #NOTE: it has occured to me that modifying certain params this way just isnt very practical or useful
                    # for example, modifying std of distribution for a param, would require getting all the values of
                    # the param, and somehow remapping them to the new std for each cell. 
                    # I dont think this would be very useful, and would be pretty complicated to implement.
                    # by contrast, if the mean of the distribution was modified, it would be much simpler to just 
                    # shift all the values by the same proportion.
                    
                    #the following if statement will skip over these kinds of params
                    excepted_param_keys = [
                        'std', # see rationale above
                        #'probLengthConst', # NOTE: this is included into a string that passed and evaluated in hoc,
                                            # 'probability': 'exp(-dist_3D / {})*{}'.format(cfg.probLengthConst, spec['prob']),
                                            # it's easier to modify probability to get a sense of how it affects the network
                                            
                                            # NOTE: nvm I figured it out. I can just modify the string directly.
                        ]
                    if any([key in cfg_param for key in excepted_param_keys]):
                        if verbose: print(f'Skipping permutations for {cfg_param}...')
                        continue
                    
                    if verbose: print(f'Generating permutations for {cfg_param}...')
                    
                    
                    # 2025-01-10 10:09:03 aw - original code. 2 levels of permutations, hard coded.
                    # going to implement code that will allow for any number of levels of permutations 
                    # between the lower and upper bounds.
                    
                    # #create two permutations of cfg in this param, 0.2 of the original, and 1.8 of the original
                    # cfg_permutation_1 = simConfig.__dict__.copy()
                    # cfg_permutation_1[cfg_param] = cfg_val * lower_bound
                    # cfg_permutation_1['simLabel'] = f'{cfg_param}_reduced'
                    # cfg_permutations.append((
                    #     cfg_permutation_1, #cfg
                    #     cfg_param, #permuted param
                    #     cfg_val, #original value                    
                    #     ))
                    
                    # cfg_permutation_2 = simConfig.__dict__.copy()
                    # cfg_permutation_2[cfg_param] = cfg_val * upper_bound
                    # cfg_permutation_2['simLabel'] = f'{cfg_param}_increased'
                    # cfg_permutations.append((
                    #     cfg_permutation_2,
                    #     cfg_param,
                    #     cfg_val,                    
                    #     ))
                    
                    
                    # 2025-01-10 10:10:08 aw new code. any number of levels of permutations between lower and upper bounds.
                    def append_permutation_levels(cfg_permutations, 
                                                  simConfig, 
                                                  cfg_param, 
                                                  cfg_val, 
                                                  lower_bound, 
                                                  upper_bound, 
                                                  levels):
                        """
                        Append permutations to the list of cfg permutations for a given parameter.
                        """
                        def permute_param(cfg_permutation, cfg_param, upper_bound, lower_bound, level, levels):                         
                            """
                            Handle special cases where the parameter should not be permuted.
                            """
                            # if verbose: print(f'Skipping permutations for {cfg_param}...')
                            # return cfg_permutations
                            # special cases
                            #print(f'Generating levels for {cfg_param}...')
                            if 'LengthConst' in cfg_param:
                                # got to typical case, this isnt actually a probability based param
                                # TODO: rename this to just length constant later.
                                upper_value = cfg_val * upper_bound
                                lower_value = cfg_val * lower_bound                       
                            elif 'prob' in cfg_param:
                                #modify upper and lower bounds such that probability based params 
                                #dont go below 0 or above 1
                                upper_value = cfg_val * upper_bound
                                lower_value = cfg_val * lower_bound
                                if upper_value > 1:
                                    upper_value = 1
                                if lower_value < 0:
                                    lower_value = 0
                                
                                # #calculate new upper and lower bounds to be used in the permutations
                                # upper_bound = 1 / cfg_val
                                # lower_bound = 0 / cfg_val
                            else:
                                #typical case
                                upper_value = cfg_val * upper_bound
                                lower_value = cfg_val * lower_bound
                                
                            # do two linspaces and stitch them together to ensure cfg_val is centered.
                            permuted_vals_1 = np.linspace(lower_value, cfg_val, levels // 2 + 1)[:-1] # returns all but the last value (exclude cfg_val)
                            permuted_vals_2 = np.linspace(cfg_val, upper_value, levels // 2 + 1)[1:] ## returns all but the first value (exclude cfg_val)
                            permuted_vals = np.concatenate((permuted_vals_1, permuted_vals_2))
                            
                            # quality
                            assert permuted_vals.size == levels, f'Expected {levels} permuted values, got {permuted_vals.size}'
                            assert np.all(np.diff(permuted_vals) > 0), f'Permuted values are not in ascending order: {permuted_vals}'
                            assert cfg_val > permuted_vals[levels//2-1], f'Permuted value {permuted_vals[levels//2-1]} is not less than original value {cfg_val}'
                            assert cfg_val < permuted_vals[levels//2], f'Permuted value {permuted_vals[levels//2]} is not greater than original value {cfg_val}'
                                
                            # return permuted value
                            #permuted_vals = np.linspace(lower_value, upper_value, levels)
                            permuted_val = permuted_vals[level]
                            cfg_permutation[cfg_param] = permuted_val
                            return cfg_permutation, permuted_vals
                        #print(f'Generating levels for {cfg_param}...')
                        for i in range(levels):
                            original_cfg = simConfig.__dict__.copy()
                            cfg_permutation = simConfig.__dict__.copy()
                            #temp_upper, temp_lower = upper_bound, lower_bound # save original values for future iterations
                            cfg_permutation, permuted_vals = permute_param(cfg_permutation, cfg_param, upper_bound, lower_bound, i, levels)
                            # cfg_permutation[cfg_param] = cfg_val * (lower_bound + (i * (upper_bound - lower_bound) / (levels - 1)))
                            # upper_bound, lower_bound = temp_upper, temp_lower # set back to original values for next iteration
                            cfg_permutation['simLabel'] = f'{cfg_param}_{i}'
                            cfg_permutations.append((
                                cfg_permutation,
                                cfg_param,
                                cfg_val,
                                ))
                            
                            # aw 2025-01-17 08:38:27 - adding validation controls to ensure permutations are correct
                            # seems like there's no issue in this step - however there may be issues in the net preprocessing
                            assert cfg_permutation[cfg_param] == permuted_vals[i], f'Failed to permute {cfg_param} to {permuted_vals[i]}'
                            assert original_cfg != cfg_permutation, f'Failed to permute {cfg_param} to {permuted_vals[i]}'
                        #print(f'Generated permuted vals: {permuted_vals}')
                        return cfg_permutations
                    
                    cfg_permutations = append_permutation_levels(cfg_permutations,
                                                simConfig,
                                                cfg_param,
                                                cfg_val,
                                                lower_bound,
                                                upper_bound,
                                                levels)
                    
                    if verbose: print(f'Permutations generated for {cfg_param}!')
        #debug
        #only keep cfgs where the simLabel contains 'gk' or 'gna'
        # #TODO: figure out the issue with these params
        # cfg_permutations = [cfg_permutation for cfg_permutation in cfg_permutations if 'gk' in cfg_permutation[0]['simLabel'] or 'gna' in cfg_permutation[0]['simLabel']]
        
        print(f'Generated {len(cfg_permutations)} cfg permutations.')
        return cfg_permutations
    cfg_permutations = generate_permutations(sim_data_path, 
                                             output_path,
                                             #evol_params_path, 
                                             #sensitivity_analysis_output_path, 
                                             lower_bound=lower_bound, 
                                             upper_bound=upper_bound,
                                             levels=levels,
                                             duration_seconds=duration_seconds
                                             )

    # run permutation, test individual perms as needed
    if option == 'serial':
        for perm_simConfig in cfg_permutations:
            try:
                run_permutation(
                    sim_data_path,
                    reference_data_path = reference_data_path,
                    plot = plot,
                    conv_params=conv_params,
                    mega_params=mega_params,
                    fitnessFuncArgs=fitnessFuncArgs,
                    debug = debug, 
                    *perm_simConfig
                    )
            except Exception as e:
                print(f'Error running permutation {perm_simConfig["simLabel"]}: {e}')
                print('Continuing to next permutation...')
            #break
    elif option == 'parallel':
        # run all permutations, parallelized        
        # TODO: validate this function
        # NOTE: this one seems to work, the ones above, do not.
                # running this one for now to see if it works.
        def run_all_permutations(
            sim_data_path, 
            cfg_permutations, 
            plot=None, 
            reference_data_path=None, 
            num_workers=None,
            debug=False,
            conv_params=None,
            mega_params=None,
            fitnessFuncArgs=None,
            try_loading=True      
            ):
            """
            Run all configuration permutations in parallel, limited by logical CPU availability.

            Args:
                sim_data_path (str): Path to simulation data.
                cfg_permutations (list): List of configuration permutations to run.
                plot: Plot-related information for `run_permutation` (optional).
                reference_data_path (str): Reference data path for `run_permutation` (optional).
            """
            import os
            from concurrent.futures import ProcessPoolExecutor, as_completed

            available_cpus = os.cpu_count()
            
            # set number of workers (i.e. processes, simultaneously running simulations)
            if num_workers is None:
                num_workers = min(len(cfg_permutations), available_cpus)
            else:
                num_workers = min(num_workers, available_cpus)
            print(f'Using {num_workers} workers out of {available_cpus} available CPUs.')

            # Evenly distribute threads among processes
            threads_per_worker = max(1, available_cpus // num_workers)
            os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
            os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
            os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
            os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)

            # Prepare tasks
            tasks = []
            for cfg_tuple in cfg_permutations:
                if isinstance(cfg_tuple, tuple) and len(cfg_tuple) == 3:
                    cfg, cfg_param, cfg_val = cfg_tuple
                    tasks.append((sim_data_path, 
                                  cfg, cfg_param, 
                                  cfg_val, 
                                  reference_data_path, 
                                  plot, 
                                  debug, 
                                  conv_params,
                                  mega_params, 
                                  fitnessFuncArgs,
                                  try_loading,
                                  ))
                else:
                    raise ValueError(f"Unexpected structure in cfg_permutations: {cfg_tuple}")

            # Run tasks in parallel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(run_permutation, *task): task for task in tasks}

                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        future.result()  # This will raise any exceptions from the worker
                    except Exception as e:
                        cfg = task[1]  # Access the configuration from the task
                        sim_label = cfg.get("simLabel", "unknown") if isinstance(cfg, dict) else "unknown"
                        print(f"Unhandled exception in permutation {sim_label}: {e}")
        run_all_permutations(
            sim_data_path, 
            cfg_permutations, 
            plot=plot, 
            reference_data_path=reference_data_path, 
            num_workers=num_workers,
            debug=debug,
            conv_params=conv_params,
            mega_params=mega_params,
            fitnessFuncArgs=fitnessFuncArgs,
            try_loading=try_loading
            )

def run_permutation(
    sim_data_path, 
    cfg, 
    cfg_param, 
    cfg_val,
    reference_data_path = None,
    plot = False,
    debug = False,
    conv_params = None,
    mega_params = None,
    fitnessFuncArgs = None,
    try_loading = True,
    *args
    ):
    
    if plot: assert reference_data_path is not None, "Reference data path must be provided for plotting."
    
    try: 
        simLabel = cfg['simLabel']
        print(f'Running permutation {simLabel}...')
        
        if cfg_param is not None and cfg_val is not None: #if none, then it's the original simConfig            
            prepare_permuted_sim(sim_data_path, 
                                 cfg, 
                                 cfg_param, 
                                 cfg_val, 
                                 try_loading=try_loading)
            
            # if try_loading is true, check if perpare_permuted_sim succeeded in loading all sim data
            # this should save a little time.
            if try_loading and hasattr(sim, 'allSimData'):
                pass
            else:            
                sim.runSim()                        # run parallel Neuron simulation
                sim.gatherData()                    # gather spiking data and cell info from each node
        elif not debug: # i.e. if original simConfig is being re-run for official 
                        # sensitivity analysis - confirming that the basic load and re-run method does
                        # in fact generate identical results.
            # #sim.clearAll()
            # if not debug_permuted_sims:
            sim.load(sim_data_path, simConfig=cfg)
            sim.runSim()                        # run parallel Neuron simulation
            sim.gatherData()                    # gather spiking data and cell info from each node
        elif debug: 
            print(f'Runing in debug mode. Only loading simConfig from {sim_data_path}')
            print(f'Simulation will not be re-run. it will, however, be saved to the expected location.')
            #if sim is None: #assume sim is already loaded
            sim.load(sim_data_path, simConfig=cfg)
            #else: pass
            print()
        else:
            print('Not sure how you got here... Something went wrong.')
                
        permuted_data_paths = sim.saveData()                      # save params, cell info and sim output to file (pickle,mat,txt,etc)
        assert len(permuted_data_paths) == 1, "Expected only one data path, the .pkl file. Got more."
        perm_sim_data_path = permuted_data_paths[0]
    except Exception as e:
        print(f'Error running permutation {simLabel}: {e}')
        return
        
    if plot:
        assert conv_params is not None, "Conversion parameters must be provided for plotting."
        assert mega_params is not None, "Mega parameters must be provided for plotting."
        assert fitnessFuncArgs is not None, "Fitness function arguments must be provided for plotting."
        try:        
            process_simulation(
                perm_sim_data_path, 
                reference_data_path, 
                DEBUG_MODE=debug,
                conv_params=conv_params,
                mega_params=mega_params,
                fitnessFuncArgs=fitnessFuncArgs,
                #try_loading=try_loading
                )
        except Exception as e:
            print(f'Error processing permutation {simLabel}: {e}')
            traceback.print_exc()
            return
        
    print(f'Permutation {simLabel} successfully ran!')  

def prepare_permuted_sim(
    sim_data_path, 
    cfg, 
    cfg_param, 
    cfg_val,
    try_loading=True):
    
    #try loading sim data if possible
    if try_loading:
        try:
            expected_save_path = os.path.join(cfg['saveFolder'], f'{cfg["simLabel"]}_data.pkl')
            exists = os.path.exists(expected_save_path)
            if exists:
                print(f'Simulation data for {cfg["simLabel"]} already exists at {expected_save_path}. Attempting to load...')
                sim.load(expected_save_path)
                assert hasattr(sim, 'net'), "Simulation data loaded but 'net' attribute is missing."
                #assert hasattr(sim, 'netParams'), "Simulation data loaded but 'netParams' attribute is missing."
                assert hasattr(sim, 'cfg'), "Simulation data loaded but 'simConfig' attribute is missing."
                #assert cfg == sim.cfg, "Loaded simulation data does not match the expected configuration."
                assert cfg[cfg_param]==sim.cfg[cfg_param], "Loaded simulation data does not match the expected configuration."
                assert hasattr(sim, 'simData'), "Simulation data loaded but 'simData' attribute is missing."
                assert hasattr(sim, 'allSimData'), "Simulation data loaded but 'allSimData' attribute is missing."
                #assert hasattr(sim, 'allCellData'), "Simulation data loaded but 'allCellData' attribute is missing."
                print(f'Simulation data for {cfg["simLabel"]} loaded successfully.')
                return
        except Exception as e:
            print(f'Error loading simulation data for {cfg["simLabel"]}: {e}')
            print(f'Will attempt to run the simulation instead.')
            try: sim.clearAll()
            except: pass   # continue to run the simulation if loading fails
    
    # load netparams and permute
    sim.load(sim_data_path, simConfig=cfg)
    simConfig = specs.SimConfig(simConfigDict=cfg)
    netParams = sim.loadNetParams(sim_data_path, setLoaded=False)
    
    #Typical case:
    strategy = 'by_value'
    #SPECIAL CASES: gnabar, gkbar, L, diam, Ra
    handle_by_name = ['gnabar', 'gkbar', 'L', 'diam', 'Ra']
    if any([name in cfg_param for name in handle_by_name]): 
        if '_' in cfg_param:
            elements = cfg_param.split('_')
            for element in elements:
                if any([name==element for name in handle_by_name]):
                    strategy = 'by_name'
                    break
        elif any([name==cfg_param for name in handle_by_name]):
            strategy = 'by_name'

    cfg_to_netparams_mapping = map_cfg_to_netparams(
        {cfg_param: cfg_val}, 
        netParams.__dict__.copy(),
        strategy=strategy
        )
    mapped_paths = cfg_to_netparams_mapping[cfg_param]
    
    if mapped_paths is None:
        print(f"WARNING: mapped paths is None.")
        print(f"No paths found for {cfg_param} = {cfg_val}")
        return

    # update permuted params
    def getNestedParam(netParams, paramLabel):
        if '.' in paramLabel: 
            paramLabel = paramLabel.split('.')
        if isinstance(paramLabel, list ) or isinstance(paramLabel, tuple):
            container = netParams
            for ip in range(len(paramLabel) - 1):
                if hasattr(container, paramLabel[ip]):
                    container = getattr(container, paramLabel[ip])
                else:
                    container = container[paramLabel[ip]]
            return container[paramLabel[-1]]
    for mapped_path in mapped_paths:    
        current_val = getNestedParam(netParams, mapped_path)
        #assert cfg_val == current_val, f"Expected {cfg_val} but got {current_val}"
        try:
            if isinstance(current_val, str): #handle hoc strings
                assert str(cfg_val) in current_val, f"Expected {cfg_val} to be in {current_val}"
                updated_func = current_val.replace(str(cfg_val), str(cfg[cfg_param]))                    
                #netParams.setNestedParam(mapped_path, updated_func)
                before_val = getNestedParam(netParams, mapped_path)
                netParams.setNestedParam(mapped_path, updated_func)
                after_val = getNestedParam(netParams, mapped_path)
                assert before_val != after_val, f"Failed to update {mapped_path} from {before_val} to {after_val}"
                print(f"Updated {mapped_path} from {before_val} to {after_val}")
            elif strategy == 'by_name': #special case
                #assert cfg_val == current_val, f"Expected {cfg_val} but got {current_val}"
                original_val = cfg_val
                permuted_val = cfg[cfg_param]
                modifier = permuted_val / original_val  # NOTE: this should end up equal to one of the 
                                                        #       level multipliers
                #proportion = cfg[cfg_param] / current_val
                
                #adjust mean proportinally 
                #netParams.setNestedParam(mapped_path, current_val * modifier)
                
                before_val = getNestedParam(netParams, mapped_path)
                netParams.setNestedParam(mapped_path, current_val * modifier)
                after_val = getNestedParam(netParams, mapped_path)
                assert before_val != after_val, f"Failed to update {mapped_path} from {before_val} to {after_val}"
                print(f"Updated {mapped_path} from {before_val} to {after_val}")
            else:
                assert cfg_val == current_val, f"Expected {cfg_val} but got {current_val}"
                before_val = getNestedParam(netParams, mapped_path)
                netParams.setNestedParam(mapped_path, cfg[cfg_param])
                after_val = getNestedParam(netParams, mapped_path)
                assert before_val != after_val, f"Failed to update {mapped_path} from {before_val} to {after_val}"
                print(f"Updated {mapped_path} from {before_val} to {after_val}")  
        except:
            print(f'Error updating {mapped_path}: {e}')
            continue

    # remove previous data
    sim.clearAll()

    #remove mapping from netParams #TODO: figure out how to actually take advantage of this
    # if hasattr(netParams, 'mapping'):
    #     del netParams.mapping
    netParams.mapping = {}

    # run simulation
    # Create network and run simulation
    sim.initialize(                     # create network object and set cfg and net params
            simConfig = simConfig,          # pass simulation config and network params as arguments
            netParams = netParams)
    sim.net.createPops()                # instantiate network populations
    sim.net.createCells()               # instantiate network cells based on defined populations
    sim.net.connectCells()              # create connections between cells based on params
    sim.net.addStims()                  # add stimulation (usually there are none)
    sim.setupRecording()                # setup variables to record for each cell (spikes, V traces, etc)

def map_cfg_to_netparams(simConfig, netParams, strategy='by_value'):
    """
    Map attributes in simConfig to their corresponding locations in netParams based on values.
    
    Parameters:
        simConfig (dict): The configuration dictionary (cfg).
        netParams (object): The network parameters object.
    
    Returns:
        dict: A mapping from simConfig parameters to their paths in netParams.
    """
    def find_value_in_netparams(value, netParams, current_path=""):
        """
        Recursively search for the value in netParams and return a list of matching paths.
        
        Parameters:
            value (any): The value to search for.
            netParams (object): The network parameters object.
            current_path (str): The current path in the recursive search.
        
        Returns:
            list: A list of paths to the matching value.
        """
        stack = [(netParams, current_path)]  # Stack for backtracking, contains (current_object, current_path)
        matching_paths = []  # To store all matching paths

        while stack:
            current_obj, current_path = stack.pop()
            
            # if 'connParams' in current_path:  # Debugging: specific context output
            #     print('found connParams')
            #     if 'I->E' in current_path:
            #         print('found I->E')

            if isinstance(current_obj, dict):
                for key, val in current_obj.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    if val == value:
                        matching_paths.append(new_path)
                    elif isinstance(val, str):  # Handle HOC string matches
                        if str(value) in val:
                            matching_paths.append(new_path)
                    elif isinstance(val, (dict, list)):
                        stack.append((val, new_path))  # Push deeper layer onto stack

            elif isinstance(current_obj, list):
                for i, item in enumerate(current_obj):
                    new_path = f"{current_path}[{i}]"
                    if item == value:
                        matching_paths.append(new_path)
                    elif isinstance(item, str):  # Handle HOC string matches
                        if str(value) in item:
                            matching_paths.append(new_path)
                    elif isinstance(item, (dict, list)):
                        stack.append((item, new_path))  # Push list item onto stack

        return matching_paths  # Return all matching paths
    
    def find_name_in_netparams(name, netParams, current_path=""):
        """
        Recursively search for the name in netParams and return a list of matching paths.
        
        Parameters:
            name (str): The name to search for.
            netParams (object): The network parameters object.
            current_path (str): The current path in the recursive search.
        
        Returns:
            list: A list of paths to the matching name.
        """
        stack = [(netParams, current_path)]
        
        if '_' in name:
            elements = name.split('_')
            try: assert 'E' in elements or 'I' in elements
            except: elements = None
        else:
            elements = None
        
        matching_paths = []
        while stack:
            current_obj, current_path = stack.pop()
            
            # if 'cellParams' in current_path:  # Debugging: specific context output
            #     print('found cellParams')
                
            # if 'gnabar' in current_path:  # Debugging: specific context output
            #     print('found gnabar')
                
            # elements=None
            # #if _ in 
            
            if elements is not None:
                if isinstance(current_obj, dict):
                    for key, val in current_obj.items():
                        new_path = f"{current_path}.{key}" if current_path else key
                        if all([element in new_path for element in elements]):
                            matching_paths.append(new_path)
                        elif isinstance(val, (dict, list)):
                            stack.append((val, new_path))
                elif isinstance(current_obj, list):
                    for i, item in enumerate(current_obj):
                        new_path = f"{current_path}[{i}]"
                        if all([element in new_path for element in elements]):
                            matching_paths.append(new_path)
                        elif isinstance(item, (dict, list)):
                            stack.append((item, new_path))
                
            elif isinstance(current_obj, dict):
                for key, val in current_obj.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    #if key == name:
                    if key == name:
                        matching_paths.append(new_path)
                    elif isinstance(val, (dict, list)):
                        stack.append((val, new_path))
            elif isinstance(current_obj, list):
                for i, item in enumerate(current_obj):
                    new_path = f"{current_path}[{i}]"
                    if item == name:
                        matching_paths.append(new_path)
                    elif isinstance(item, (dict, list)):
                        stack.append((item, new_path))
        return matching_paths
        
    # Generate the mapping
    mapping = {}
    for param, value in simConfig.items():
        if strategy == 'by_name':
            #paths = find_value_in_netparams(param, netParams)
            paths = find_name_in_netparams(param, netParams)
        elif strategy == 'by_value':
            #paths = find_name_in_netparams(value, netParams)
            paths = find_value_in_netparams(value, netParams)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        mapping[param] = paths if paths else None  # Assign None if no path is found

    return mapping

''' plot sensitivity analysis results'''
def plot_sensitivity_analysis(
    og_simulation_data_path, 
    sensitvity_analysis_output_dir,
    num_workers=None,
    burst_rates=None,
    original_burst_rate=None,
    format_option='long',
    levels=6,
    plot_grid=True,
    plot_heatmaps=True
    ):
    
    plot_sensitivity_grid_plots(
        og_simulation_data_path,
        sensitvity_analysis_output_dir,
        num_workers=num_workers,
        levels=levels,
        plot_grid=plot_grid,
        plot_heatmaps=plot_heatmaps,      
    )

def load_network_metrics(input_dir, og_simulation_data_path, num_workers=None):
    #network_metrics_files = glob.glob(input_dir + '/*network_metrics.npy')
    network_metrics_files = glob.glob(input_dir + '/**/**/network_data.npy')
    #burst_rates = {}
    network_metrics_data = {}
    
    # assert
    if len(network_metrics_files) == 0: raise ValueError(f"No network metrics files found in {input_dir}")
    
    # # for debug - for network_metric_files to only be first 10
    # network_metrics_files = network_metrics_files[:10]
    
    # set number of workers (i.e. processes, simultaneously running simulations)
    available_cpus = os.cpu_count()
    if num_workers is None:
        num_workers = min(len(network_metrics_files), available_cpus)
    else:
        num_workers = min(num_workers, len(network_metrics_files), available_cpus)
    print(f'Using {num_workers} workers to load {len(network_metrics_files)} network metrics files.')

    # Evenly distribute threads among processes
    threads_per_worker = max(1, available_cpus // num_workers)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
        
    total_files = len(network_metrics_files)
    completed_files = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(metrics_loader_v2, file) for file in network_metrics_files]
        for future in as_completed(futures):
            result = future.result()
            completed_files += 1
            print(f"Completed {completed_files} out of {total_files} processes") #NOTE: there's a delay between the completion of the process and return of the result
        results = [future.result() for future in futures]
    
    # aw 2025-01-13 11:14:09 - return network metrics instead of burst rates
    # for i, (data, summary_plot, basename) in enumerate(results):
    #     if data is not None:
    #         network_metrics_data[basename] = {'summary_plot': summary_plot, 'data': data}
            
    #return network_metrics_data
    
    return results    

def metrics_loader(network_metrics_file):
    """
    Helper function to process network metrics files and extract burst rate information.
    """
    try:
        start = time.time()
        basename = os.path.basename(network_metrics_file)
        #remove file extension from basename
        basename = os.path.splitext(basename)[0]
        #print('Loading', basename, '...')
        data = np.load(network_metrics_file, allow_pickle=True).item()
        #mean_burst_rate = data['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Rate']
        #summary_plot = network_metrics_file.replace('network_metrics.npy', 'summary_plot.png')
        summary_plot = network_metrics_file.replace('network_data.npy', 'summary_plot.png')
        
        # #just for debug
        # from netpyne import sim
        # sim_data_path = network_metrics_file.replace('_network_metrics.npy', '_data.pkl')
        # # netParams = sim.loadNetParams(sim_data_path, setLoaded=False)
        # # net = sim.loadNet(sim_data_path)
        # sim.load(sim_data_path)
        # netParams = sim.net.params
        # #just for debug #TODO: not sure if some changes to netparams didnt get applied before running the simulation
        # # TODO: Seems like some changes to netParams didn't get applied before running the simulation - at least tau params are not being applied
        
        # aw 2025-01-13 11:44:00 - sending the whole data object back is going really slow... I guess I should curate the data object to only include the necessary information
        print('Curating data from network_metrics file...')
        # oh it looks like I'm sending the whole simulation data object - in addition to network metrics data - back to the main function. I should curate the data object to only include the necessary information.
        curated_data = { #yea this is going quite a bit faster. Dont neet all the individual data objects, just the network metrics and simData.
            'simConfig': data['simConfig'],
            'simData': data['simData'],
            'network_metrics': data['network_metrics'] 
        }  
        print('Loaded', basename, 'in', round(time.time() - start, 2), 'seconds.')
        return curated_data, summary_plot, basename
    except Exception as e:
        print('Error loading', network_metrics_file, ':', e)
        return None, None, basename

def plot_sensitivity_grid_plots(
    og_simulation_data_path, 
    sensitvity_analysis_output_dir,
    num_workers=None,
    burst_rates=None,
    original_burst_rate=None,
    format_option='long',
    levels=6,
    plot_grid=True,
    plot_heatmaps=True
    ):
    """
    Plots a grid of summary plots with color-modulated cells based on changes in burst rates.
    """
    
    # Set up paths and parameters
    #output_dir = sensitvity_analysis_output_dir
    input_dir = os.path.join(sensitvity_analysis_output_dir, 'simulations')
    output_dir = os.path.join(sensitvity_analysis_output_dir, 'summary_plots')
    sim_data_path = og_simulation_data_path
    
    # assertions
    assert os.path.exists(input_dir), f'Input directory {input_dir} does not exist.'
    assert os.path.exists(sim_data_path), f'Simulation data path {sim_data_path} does not exist.'
    
    # dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # get paths in grid format
    def get_clean_grid(
        input_dir,
        query = 'summary_plot.png'
        ):
        # find the original summary plot
        files = []
        for file in os.listdir(input_dir):
            if file.startswith('_') and query in file:
                og_sumamry_path = os.path.join(input_dir, file)
                files.append(os.path.join(input_dir, file))
                print('Found original summary plot:', og_sumamry_path)
                break
        assert len(files) == 1, f'Expected 1 original summary plot, found {len(files)}'
        
        # iterate through params, load associated plots, build a grid of paths
        grid = {} #store png paths here
        for param_idx, param in enumerate(params):
            
            #check if param value is a list, or tuple, of two values - if so dont skip, else skip
            param_val = params[param]
            if not isinstance(param_val, (list, tuple)):
                #print('skipping', param)
                continue
            #print() #print a line to separate outputs
            
            # Arrange data into a grid
            def get_perms_per_param(param):
                
                # init file list
                files = []
                
                #iterate through files in input_dir, get number of permutations from filename context
                num_permutations = 0
                param_elements = param.split('_')
                
                # iterate through files in input_dir, get number of permutations from filename context
                for file in os.listdir(input_dir):
                    
                    # check if slide was generated for this param - indicateing all plotting was done
                    # if not, skip
                    # if not 'comparison_summary_slide.png' in file:
                    if not query in file:
                        continue
                    
                    #print('is param in file?', param, file)
                    file_elements = file.split('_')
                    if all([param_element in file_elements for param_element in param_elements]):
                        num_permutations += 1
                        files.append(os.path.join(input_dir, file))
                        print('Found permutation for', param, 'in', file)
                        
                # debug - print number of permutations found
                # if num_permutations>0:
                #     print('Found', num_permutations, 'permutations for', param)
                
                # return number of permutations found
                return num_permutations, files
            num_permutations, summary_paths = get_perms_per_param(param)
            grid[param] = {}
            middle_idx = num_permutations // 2
            #insert og_summary_plot in the middle of summary_paths list
            if len(summary_paths) > 0:
                summary_paths.insert(middle_idx, og_sumamry_path)
                for idx, slide_path in enumerate(summary_paths):
                    #if idx < middle_idx or idx > middle_idx:
                    grid[param][idx] = slide_path
                    # elif idx == middle_idx:
                    #     grid[param][idx] = og_sumamry_path
                    #     idx = idx + 1
            print('num_permutations:', num_permutations)
            if num_permutations == 0: continue
            print() #print a line to separate outputs
            
            # quality check - make sure number of permutations is less than or equal to levels
            try:
                assert num_permutations <= levels, f'Expected {levels} permutations, found {num_permutations}'
            except Exception as e:
                print('Error:', e)                
            
        # remove empty rows
        clean_grid = {param: summary_paths for param, summary_paths in grid.items() if len(summary_paths) > 0}
        return clean_grid

    # Collect network_metrics.npy files and process
    def plot_summary_grid(
        output_dir,
        num_workers=None,
        #burst_rates=None,
        # original_burst_rate=None,
        # format_option = 'long' # aw 2025-01-11 17:02:34 - retiring this option
        format_option = 'matrix'
        ):
        
        # Plot summary grid
        print('Plotting summary grid')        
        
        if format_option == 'matrix':
            '''
            #arrange data into a grid of plots
            # y axis = params - any number of parameters that were varied in the simulation
            # x axis =  simulation permutations - usually 2-6 permutations of the simulation (ideally an even number) 
                        # + 1 column for the original simulation in the middle. (which is why an even number of permutations is nice)
            '''
            
            # get dict of paths for matrix
            clean_grid = get_clean_grid(output_dir)
            
            # Create a grid of plots
            n_rows = len(clean_grid)
            n_cols = levels+1
            #fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 7.5 * n_rows))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
            for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                for col_idx, summary_path in summary_paths.items():
                    try:
                        img = mpimg.imread(summary_path)
                        axs[row_idx, col_idx].imshow(img)
                        axs[row_idx, col_idx].axis('off')
                        axs[row_idx, col_idx].set_title(f'param: {param} (perm: {col_idx})', fontsize=14)
                    except Exception as e:
                        print('Error loading plot:', e)
                print(f'Plotted {param} in row {row_idx}')
                        
            # Save the plot
            print('Saving summary grid to', output_dir + '/_summary_grid.png')
            plt.tight_layout()
            plt.savefig(output_dir + '/_summary_grid.png', dpi=100)
            output_path = os.path.join(output_dir, '_summary_grid.png')
            print('done.')
            
            # Return original burst rate, burst rates, and output path
            # return original_burst_rate, burst_rates, output_path
            return output_path, clean_grid, original_burst_rate
        
        # reject unknown format options
        else:
            raise ValueError(f"Unknown format_option: {format_option}")    
    if plot_grid:
        summary_grid_path, clean_grid, original_burst_rate = plot_summary_grid(
        #sim_data_path,
        output_dir,
        num_workers=num_workers,
        #burst_rates=burst_rates,
        #original_burst_rate=original_burst_rate,
        )
        
    # aw 2025-01-14 08:09:53 - depreciating the above fuc, going to copy paste it and make changes. New one will do more than just burst rates
    def plot_heat_maps(
        output_dir,
        input_dir,
        num_workers=None,
        levels=6,
        ):
        
        # get dict of paths for matrix
        clean_grid = get_clean_grid(
            input_dir,
            query='network_metrics.npy'
            )
        
        # Collect network_metrics.npy files and process #NOTE: parallel processing is used here
        #if burst_rates is None or original_burst_rate is None:
        network_metrics_data = load_network_metrics(
            input_dir, 
            sim_data_path,
            num_workers=num_workers,
            )

        # # plot mean_Burst_Rate heatmap
        # def plot_mean_Burst_Rate(output_dir):
        #     # Plot a grid of parameters with burst rate changes
        #     print('Plotting summary grid with color gradient')   
        
        #     # original burst rate, find a key with '_' in basename
        #     for key in network_metrics_data.keys():
        #         if key.startswith('_'):
        #             original_key = key
        #             original_burst_rate = network_metrics_data[key]['data']['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Rate']
        #             break
                
        #     # get min and max burst rates
        #     min_burst_rate = 0 #initialize min_burst_rate
        #     max_burst_rate = 0 #initialize max_burst_rate
        #     for key in network_metrics_data.keys():
        #         data = network_metrics_data[key]['data']
        #         burst_rate = data['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Rate']
        #         if burst_rate < original_burst_rate and burst_rate < min_burst_rate:
        #             min_burst_rate = burst_rate
        #         if burst_rate > original_burst_rate and burst_rate > max_burst_rate:
        #             max_burst_rate = burst_rate
                    
                    
        #     # create a color gradient using min, max, and original burst rates. 
        #     # closer to original = more white
        #     # closer to min = more blue
        #     # closer to max = more red
        #     colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
        #     cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)

        #     # create a norm object to center the color gradient around the original burst rate
        #     norm = mcolors.CenteredNorm(
        #         vcenter=original_burst_rate, 
        #         #halfrange=max(abs(min_burst_rate - original_burst_rate), abs(max_burst_rate - original_burst_rate)))
        #         halfrange = min(abs(min_burst_rate - original_burst_rate), abs(max_burst_rate - original_burst_rate))
        #         )
            
        #     #prepare data dicts in each grid cell
        #     for param, summary_paths in clean_grid.items():
        #         clean_grid[param]['data'] = {}
                    
        #     # add network metrics data to clean grid
        #     for param, summary_paths in clean_grid.items():
        #         # if clean_grid[param]['data'] is None:
        #         #     clean_grid[param]['data'] = {}
        #         for key, data in network_metrics_data.items():
        #             if param in key:
        #                 clean_grid[param]['data'].update({key: data})
        #                 #break
        #     #print('Added network metrics data to clean grid')
            
        #     # first, heatmap burst rates
        #     n_rows = len(clean_grid)
        #     n_cols = levels+1
        #     fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1 * n_rows))
        #     for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                
        #         row_data = clean_grid[param]['data']
        #         #sort row_data by key
        #         sorted_row_data = dict(sorted(row_data.items()))
                
        #         # insert original summary plot in the middle of row_data
        #         middle_idx = len(sorted_row_data) // 2
        #         new_row_data = {}
        #         for idx, (key, value) in enumerate(sorted_row_data.items()):
        #             if idx == middle_idx:
        #                 new_row_data['original_data'] = network_metrics_data[original_key]
        #             new_row_data[key] = value
                
        #         clean_grid[param]['data'] = new_row_data
        #         row_data = clean_grid[param]['data']            
                
        #         #plot each cell in the row
        #         for col_idx, (key, data) in enumerate(row_data.items()):
        #             try:
        #                 #print()
                        
        #                 burst_rate = clean_grid[param]['data'][key]['data']['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Rate']
                        
        #                 # # Set color based on burst rate
        #                 # norm_burst_rate = (burst_rate - min_burst_rate) / (original_burst_rate)
        #                 # if burst_rate == original_burst_rate:
        #                 #     color = (1, 1, 1)  # white for original burst rate
        #                 # else:
        #                 #     colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        #                 #     cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
        #                 #     color = cmap(norm_burst_rate)
                        
        #                 # set color based on burst rate
        #                 # norm_burst_rate = (burst_rate) / (original_burst_rate)
        #                 # color = cmap(norm_burst_rate)
                        
        #                 color = cmap(norm(burst_rate))
                        
        #                 axs[row_idx, col_idx].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        #                 axs[row_idx, col_idx].text(0.5, 0.5, f'{burst_rate:.2f}', ha='center', va='center', fontsize=12)
        #                 axs[row_idx, col_idx].axis('off')
        #                 #axs[row_idx, col_idx].set_title(f'param: {param} (perm: {col_idx})', fontsize=18)
        #                 permuted_param = param
        #                 permuted_value = data['data']['simConfig'][param]
        #                 #axs[row_idx, col_idx].set_title(f'(perm: {col_idx})', fontsize=14)
        #                 #round permuted value to 3 decimal places
        #                 try: permuted_value = round(permuted_value, 3)
        #                 except: pass
        #                 axs[row_idx, col_idx].set_title(f'@{permuted_value}', fontsize=14)
        #             except Exception as e:
        #                 print('Error loading plot:', e)
        #         print(f'Plotted {param} in row {row_idx}')
            
        #     #tight layout
        #     plt.tight_layout()
            
        #     #reveal y axis labels on the left side of the figure
        #     plt.subplots_adjust(left=0.15)
            
        #     # Add space to the right of the figure for the color bar
        #     fig.subplots_adjust(right=.90)
            
        #     # Add space at the time for title
        #     fig.subplots_adjust(top=0.925)
            
        #     #
        #     for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                
        #         # Get the position of the leftmost subplot
        #         pos = axs[row_idx, 0].get_position()
        #         x = pos.x0 - 0.025

        #         # Add row title to the left of the row
        #         fig.text(x, pos.y0 + pos.height / 2, param, va='center', ha='right', fontsize=14, rotation=0)
            
        #     # Add a color bar legend
        #     #sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_burst_rate, vmax=max_burst_rate))
        #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        #     sm.set_array([])
        #     #cbar_ax = fig.add_axes([0.86, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        #     cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        #     cbar = fig.colorbar(sm, cax=cbar_ax)
        #     #cbar.set_label('Burst Rate')
            
        #     # Add title
        #     metric = 'mean_Burst_Rate' #HACK: hardcoding this for now
        #     fig.suptitle(f'heatmap: {metric}', fontsize=16)
            
        #     # Save the plot
        #     print('Saving heat map to', output_dir + '/_BR_heat_map.png')
        #     plt.savefig(output_dir + '/_BR_heat_map.png', dpi=100)
        #     output_path = os.path.join(output_dir, '_BR_heat_map.png')
        #     print('done.')
            
        #     # return output_path
        #     return output_path
        # BR_heatmap_path = plot_mean_Burst_Rate(output_dir)

        # # plot mean_Burst_amplitude heatmap
        # def plot_mean_Burst_Peak(output_dir):
        #     # Plot a grid of parameters with burst rate changes
        #     print('Plotting summary grid with color gradient')   
        
        #     # original burst rate, find a key with '_' in basename
        #     for key in network_metrics_data.keys():
        #         if key.startswith('_'):
        #             original_key = key
        #             original_burst_peak = network_metrics_data[key]['data']['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        #             break
                
        #     # get min and max burst rates
        #     min_burst_peak = 0
        #     max_burst_peak = 0
        #     for key in network_metrics_data.keys():
        #         data = network_metrics_data[key]['data']
        #         burst_peak = data['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        #         if burst_peak < original_burst_peak and burst_peak < min_burst_peak:
        #             min_burst_peak = burst_peak
        #         if burst_peak > original_burst_peak and burst_peak > max_burst_peak:
        #             max_burst_peak = burst_peak
                    
        #     # create a color gradient using min, max, and original burst rates.
        #     colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        #     cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
            
        #     # create a norm object to center the color gradient around the original burst rate
        #     norm = mcolors.CenteredNorm(
        #         vcenter=original_burst_peak, 
        #         halfrange = min(abs(min_burst_peak - original_burst_peak), abs(max_burst_peak - original_burst_peak))
        #         )
            
        #     #prepare data dicts in each grid cell
        #     for param, summary_paths in clean_grid.items():
        #         clean_grid[param]['data'] = {}
                
        #     # add network metrics data to clean grid
        #     for param, summary_paths in clean_grid.items():
        #         for key, data in network_metrics_data.items():
        #             if param in key:
        #                 clean_grid[param]['data'].update({key: data})
                        
        #     # heat map for mean_Burst_Peak
        #     n_rows = len(clean_grid)
        #     n_cols = levels+1
        #     fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1 * n_rows))
        #     for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                
        #         row_data = clean_grid[param]['data']
        #         #sort row_data by key
        #         sorted_row_data = dict(sorted(row_data.items()))
                
        #         # insert original summary plot in the middle of row_data
        #         middle_idx = len(sorted_row_data) // 2
        #         new_row_data = {}
        #         for idx, (key, value) in enumerate(sorted_row_data.items()):
        #             if idx == middle_idx:
        #                 new_row_data['original_data'] = network_metrics_data[original_key]
        #             new_row_data[key] = value
                
        #         clean_grid[param]['data'] = new_row_data
        #         row_data = clean_grid[param]['data']            
                
        #         #plot each cell in the row
        #         for col_idx, (key, data) in enumerate(row_data.items()):
        #             try:
        #                 burst_peak = clean_grid[param]['data'][key]['data']['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        #                 color = cmap(norm(burst_peak))
        #                 axs[row_idx, col_idx].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        #                 axs[row_idx, col_idx].text(0.5, 0.5, f'{burst_peak:.2f}', ha='center', va='center', fontsize=12)
        #                 axs[row_idx, col_idx].axis('off')
        #                 permuted_param = param
        #                 permuted_value = data['data']['simConfig'][param]
        #                 try: permuted_value = round(permuted_value, 3)
        #                 except: pass
        #                 axs[row_idx, col_idx].set_title(f'@{permuted_value}', fontsize=14)
        #             except Exception as e:
        #                 print('Error loading plot:', e)
        #         print(f'Plotted {param} in row {row_idx}')
                
        #     plt.tight_layout()
        #     plt.subplots_adjust(left=0.15)
        #     fig.subplots_adjust(right=.90)
        #     fig.subplots_adjust(top=0.925)
        #     for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
        #         pos = axs[row_idx, 0].get_position()
        #         x = pos.x0 - 0.025
        #         fig.text(x, pos.y0 + pos.height / 2, param, va='center', ha='right', fontsize=14, rotation=0)
        #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        #     sm.set_array([])
        #     cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
        #     cbar = fig.colorbar(sm, cax=cbar_ax)
        #     metric = 'mean_Burst_Peak'
        #     fig.suptitle(f'heatmap: {metric}', fontsize=16)
        #     print('Saving heat map to', output_dir + '/_BP_heat_map.png')
        #     plt.savefig(output_dir + '/_BP_heat_map.png', dpi=100)
        #     output_path = os.path.join(output_dir, '_BP_heat_map.png')
        #     print('done.')
            
        #     return output_path
        # BP_heatmap_path = plot_mean_Burst_Peak(output_dir)

        # # plot fano_factor
        # def plot_fano_factor(output_dir):
        #     # Plot a grid of parameters with fano factor changes
        #     print('Plotting summary grid with color gradient')
            
        #     # original fano factor, find a key with '_' in basename
        #     for key in network_metrics_data.keys():
        #         if key.startswith('_'):
        #             original_key = key
        #             original_fano_factor = network_metrics_data[key]['data']['network_metrics']['bursting_data']['bursting_summary_data']['fano_factor']
        #             break
                
        #     # get min and max fano factors
        #     min_fano_factor = 0
        #     max_fano_factor = 0
        #     for key in network_metrics_data.keys():
        #         data = network_metrics_data[key]['data']
        #         fano_factor = data['network_metrics']['bursting_data']['bursting_summary_data']['fano_factor']
        #         if fano_factor < original_fano_factor and fano_factor < min_fano_factor:
        #             min_fano_factor = fano_factor
        #         if fano_factor > original_fano_factor and fano_factor > max_fano_factor:
        #             max_fano_factor = fano_factor
                    
        #     # create a color gradient using min, max, and original fano factors.
        #     colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        #     cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
            
        #     # create a norm object to center the color gradient around the original fano factor
        #     norm = mcolors.CenteredNorm(
        #         vcenter=original_fano_factor, 
        #         halfrange = min(abs(min_fano_factor - original_fano_factor), abs(max_fano_factor - original_fano_factor))
        #         )
            
        #     #prepare data dicts in each grid cell
        #     for param, summary_paths in clean_grid.items():
        #         clean_grid[param]['data'] = {}
                
        #     # add network metrics data to clean grid
        #     for param, summary_paths in clean_grid.items():
        #         for key, data in network_metrics_data.items():
        #             if param in key:
        #                 clean_grid[param]['data'].update({key: data})
                        
        #     # heat map for fano factor
        #     n_rows = len(clean_grid)
        #     n_cols = levels+1
        #     fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1 * n_rows))
        #     for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                
        #         row_data = clean_grid[param]['data']
        #         #sort row_data by key
        #         sorted_row_data = dict(sorted(row_data.items()))
                
        #         # insert original summary plot in the middle of row_data
        #         middle_idx = len(sorted_row_data) // 2
        #         new_row_data = {}
        #         for idx, (key, value) in enumerate(sorted_row_data.items()):
        #             if idx == middle_idx:
        #                 new_row_data['original_data'] = network_metrics_data[original_key]
        #             new_row_data[key] = value
                
        #         clean_grid[param]['data'] = new_row_data
        #         row_data = clean_grid[param]['data']            
                
        #         #plot each cell in the row
        #         for col_idx, (key, data) in enumerate(row_data.items()):
        #             try:
        #                 fano_factor = clean_grid[param]['data'][key]['data']['network_metrics']['bursting_data']['bursting_summary_data']['fano_factor']
        #                 color = cmap(norm(fano_factor))
        #                 axs[row_idx, col_idx].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        #                 axs[row_idx, col_idx].text(0.5, 0.5, f'{fano_factor:.2f}', ha='center', va='center', fontsize=12)
        #                 axs[row_idx, col_idx].axis('off')
        #                 permuted_param = param
        #                 permuted_value = data['data']['simConfig'][param]
        #                 try: permuted_value = round(permuted_value, 3)
        #                 except: pass
        #                 axs[row_idx, col_idx].set_title(f'@{permuted_value}', fontsize=14)
        #             except Exception as e:
        #                 print('Error loading plot:', e)
        #         print(f'Plotted {param} in row {row_idx}')

        #     plt.tight_layout()
        #     plt.subplots_adjust(left=0.15)
        #     fig.subplots_adjust(right=.90)
        #     fig.subplots_adjust(top=0.925)
        #     for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
        #         pos = axs[row_idx, 0].get_position()
        #         x = pos.x0 - 0.025
        #         fig.text(x, pos.y0 + pos.height / 2, param, va='center', ha='right', fontsize=14, rotation=0)
                
        #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        #     sm.set_array([])
        #     cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
        #     cbar = fig.colorbar(sm, cax=cbar_ax)
        #     metric = 'fano_factor'
        #     fig.suptitle(f'heatmap: {metric}', fontsize=16)
        #     print('Saving heat map to', output_dir + '/_FF_heat_map.png')
        #     plt.savefig(output_dir + '/_FF_heat_map.png', dpi=100)
        #     output_path = os.path.join(output_dir, '_FF_heat_map.png')
        #     print('done.')
            
        #     return output_path
        # FF_heatmap_path = plot_fano_factor(output_dir)
        
        # aw 2025-01-21 16:33:28 developing generalized function to plot heat maps for any metric
        # import os
        # import matplotlib.pyplot as plt
        # import matplotlib.colors as mcolors

        def plot_metric_heatmap(output_dir, metric_path, metric_name, network_metrics_data, clean_grid, levels):
            """
            Generalized function to plot heatmaps for a specified network metric.
            
            Args:
                output_dir (str): Directory to save the heatmap.
                metric_path (list): List of keys to navigate the metric in the network_metrics_data dictionary.
                metric_name (str): Name of the metric to display in the title and filename.
                network_metrics_data (dict): Dictionary containing network metrics data.
                clean_grid (dict): Dictionary of parameters and their data paths.
                levels (int): Number of levels for each parameter.
            """
            print(f"Plotting summary grid for {metric_name} with color gradient")
            
            # Find the original metric value
            for key in network_metrics_data.keys():
                if key.startswith('_'):
                    original_key = key
                    original_metric = network_metrics_data[key]['data']
                    for path_part in metric_path:
                        original_metric = original_metric[path_part]
                    break
            
            # Determine min and max metric values
            metric_list = []  # Initialize list to store metric values
            min_metric = float('inf')
            max_metric = float('-inf')
            for key in network_metrics_data.keys():
                data = network_metrics_data[key]['data']
                metric_value = data
                for path_part in metric_path:
                    metric_value = metric_value[path_part]
                    #print(metric_value)
                metric_list.append(float(metric_value))
                min_metric = min(min_metric, metric_value)
                max_metric = max(max_metric, metric_value)
            
            # get min and max metric values within 2 std deviations to avoid outliers
            std_dev = np.std(metric_list)
            max_val = original_metric + 2 * std_dev
            min_val = original_metric - 2 * std_dev
            
            # now if min and max arre within 2 std deviations, use them, else use the std values
            min_metric = max(min_metric, min_val)
            max_metric = min(max_metric, max_val)
            
            # Define colormap and normalization
            colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
            cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
            
            # Handle the case where original_metric is NaN
            if not np.isnan(original_metric):
                #typical case
                norm = mcolors.CenteredNorm(vcenter=original_metric, halfrange=max(abs(min_metric - original_metric), abs(max_metric - original_metric)))
            else:
                # handle case where original_metric is NaN
                norm = mcolors.Normalize(vmin=min_metric, vmax=max_metric) # normalized without centering around original simulation
                
            # Prepare data dicts for clean_grid
            for param, summary_paths in clean_grid.items():
                clean_grid[param]['data'] = {}
            for param, summary_paths in clean_grid.items():
                for key, data in network_metrics_data.items():
                    if param in key:
                        clean_grid[param]['data'].update({key: data})
            
            # Generate heatmap
            n_rows = len(clean_grid)
            n_cols = levels + 1
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1 * n_rows))
            
            for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                row_data = clean_grid[param]['data']
                sorted_row_data = dict(sorted(row_data.items()))
                middle_idx = len(sorted_row_data) // 2
                new_row_data = {}
                for idx, (key, value) in enumerate(sorted_row_data.items()):
                    if idx == middle_idx:
                        new_row_data['original_data'] = network_metrics_data[original_key]
                    new_row_data[key] = value
                clean_grid[param]['data'] = new_row_data
                
                # Plot each cell in the row
                for col_idx, (key, data) in enumerate(clean_grid[param]['data'].items()):
                    try:
                        metric_value = data['data']
                        for path_part in metric_path:
                            metric_value = metric_value[path_part]
                        color = cmap(norm(metric_value))
                        axs[row_idx, col_idx].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
                        axs[row_idx, col_idx].text(0.5, 0.5, f'{metric_value:.2f}', ha='center', va='center', fontsize=12)
                        axs[row_idx, col_idx].axis('off')
                        permuted_param = param
                        permuted_value = data['data']['simConfig'][param]
                        try:
                            permuted_value = round(permuted_value, 3)
                        except:
                            pass
                        axs[row_idx, col_idx].set_title(f'@{permuted_value}', fontsize=14)
                    except Exception as e:
                        print(f"Error loading plot for key {key}: {e}")
                #print(f"Plotted {param} in row {row_idx}")
            
            plt.tight_layout()
            plt.subplots_adjust(left=0.15, right=0.90, top=0.925)
            for row_idx, (param, summary_paths) in enumerate(clean_grid.items()):
                pos = axs[row_idx, 0].get_position()
                x = pos.x0 - 0.025
                fig.text(x, pos.y0 + pos.height / 2, param, va='center', ha='right', fontsize=14, rotation=0)
            
            # Add colorbar
            # NOTE: sm generated with norm based on original_metric = nan will result in stack overrflow when trying to generate the colorbar - to deal with this,
            # to deal with this, norm has a special case above for when original_metric is nan. norm will be set to a norm that is not centered on original simulaiton value.
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
            fig.colorbar(sm, cax=cbar_ax)
            
            # Add title and save
            fig.suptitle(f'Heatmap: {metric_name}', fontsize=16)
            output_path = os.path.join(output_dir, f'_heatmap_{metric_name}.png')
            plt.savefig(output_path, dpi=100)
            print(f'Saved heatmap to {output_path}')
            return output_path
        
        # # use this while debugging to select keys for metrics of interest
        # def walk_through_keys(data, indent=0, max_depth=3, skip_keys=None):
        #     """
        #     Recursively walk through the keys of a dictionary and print them with indentation for nested levels.
            
        #     Args:
        #         data (dict): The dictionary to traverse.
        #         indent (int): The current level of indentation.
        #     """            
        #     if indent > max_depth:
        #         return
            
        #     if not isinstance(data, dict):
        #         return  # Ensure the input is a dictionary

        #     for key, value in data.items():
        #         if skip_keys is not None:
        #             if any(skip_key in key for skip_key in skip_keys):
        #                 continue
                
        #         print("  " * indent + str(key))
        #         if isinstance(value, dict):  # If the value is a dictionary, recurse
        #             walk_through_keys(value, indent + 1, max_depth=max_depth, skip_keys=skip_keys)
                    
        #     # output should look something like:
        #     '''
        #     network_metrics
        #         source
        #         timeVector
        #         simulated_data
        #             soma_voltage
        #             E_Gids
        #             I_Gids
        #             MeanFireRate_E
        #             CoVFireRate_E
        #             MeanFireRate_I
        #             CoVFireRate_I
        #             MeanISI_E
        #             MeanISI_I
        #             CoV_ISI_E
        #             CoV_ISI_I
        #         spiking_data
        #             spike_times
        #             spiking_summary_data
        #             MeanFireRate
        #             CoVFireRate
        #             MeanISI
        #             CoV_ISI
        #         bursting_data
        #             bursting_summary_data
        #             MeanWithinBurstISI
        #             CoVWithinBurstISI
        #             MeanOutsideBurstISI
        #             CoVOutsideBurstISI
        #             MeanNetworkISI
        #             CoVNetworkISI
        #             NumUnits
        #             Number_Bursts
        #             mean_IBI
        #             cov_IBI
        #             mean_Burst_Rate
        #             mean_Burst_Peak
        #             cov_Burst_Peak
        #             fano_factor
        #             baseline
        #             ax
        #         mega_bursting_data
        #             bursting_summary_data
        #             MeanWithinBurstISI
        #             CoVWithinBurstISI
        #             MeanOutsideBurstISI
        #             CoVOutsideBurstISI
        #             MeanNetworkISI
        #             CoVNetworkISI
        #             NumUnits
        #             Number_Bursts
        #             mean_IBI
        #             cov_IBI
        #             mean_Burst_Rate
        #             mean_Burst_Peak
        #             cov_Burst_Peak
        #             fano_factor
        #             baseline
        #             ax
        #     '''
        # for key in network_metrics_data.keys():
        #     # Example usage
        #     walk_through_keys(network_metrics_data[key]['data'], max_depth=3, skip_keys=['_by_unit'])
        #     break
        
        # Metric paths of interest #TODO - I guess I could automate this by looking for any metric that resolves as a single value or something like that
        metric_paths = [
            'network_metrics.simulated_data.MeanFireRate_E',
            'network_metrics.simulated_data.MeanFireRate_I',
            'network_metrics.simulated_data.CoVFireRate_E',
            'network_metrics.simulated_data.CoVFireRate_I',
            'network_metrics.simulated_data.MeanISI_E',
            'network_metrics.simulated_data.MeanISI_I',
            'network_metrics.simulated_data.CoV_ISI_E',
            'network_metrics.simulated_data.CoV_ISI_I',
            
            'network_metrics.spiking_data.spiking_summary_data.MeanFireRate',
            'network_metrics.spiking_data.spiking_summary_data.CoVFireRate',
            'network_metrics.spiking_data.spiking_summary_data.MeanISI',
            'network_metrics.spiking_data.spiking_summary_data.CoV_ISI',
            
            'network_metrics.bursting_data.bursting_summary_data.MeanWithinBurstISI',
            'network_metrics.bursting_data.bursting_summary_data.CoVWithinBurstISI',
            'network_metrics.bursting_data.bursting_summary_data.MeanOutsideBurstISI',
            'network_metrics.bursting_data.bursting_summary_data.CoVOutsideBurstISI',
            'network_metrics.bursting_data.bursting_summary_data.MeanNetworkISI',
            'network_metrics.bursting_data.bursting_summary_data.CoVNetworkISI',
            'network_metrics.bursting_data.bursting_summary_data.mean_IBI',
            'network_metrics.bursting_data.bursting_summary_data.cov_IBI',
            'network_metrics.bursting_data.bursting_summary_data.mean_Burst_Rate',
            'network_metrics.bursting_data.bursting_summary_data.mean_Burst_Peak',
            'network_metrics.bursting_data.bursting_summary_data.fano_factor',
            'network_metrics.bursting_data.bursting_summary_data.baseline',
            
            'network_metrics.mega_bursting_data.bursting_summary_data.MeanWithinBurstISI',
            'network_metrics.mega_bursting_data.bursting_summary_data.CoVWithinBurstISI',
            'network_metrics.mega_bursting_data.bursting_summary_data.MeanOutsideBurstISI',
            'network_metrics.mega_bursting_data.bursting_summary_data.CoVOutsideBurstISI',
            'network_metrics.mega_bursting_data.bursting_summary_data.MeanNetworkISI',
            'network_metrics.mega_bursting_data.bursting_summary_data.CoVNetworkISI',
            'network_metrics.mega_bursting_data.bursting_summary_data.mean_IBI',
            'network_metrics.mega_bursting_data.bursting_summary_data.cov_IBI',
            'network_metrics.mega_bursting_data.bursting_summary_data.mean_Burst_Rate',
            'network_metrics.mega_bursting_data.bursting_summary_data.mean_Burst_Peak',
            'network_metrics.mega_bursting_data.bursting_summary_data.fano_factor',
            'network_metrics.mega_bursting_data.bursting_summary_data.baseline',            
        ] 
        
        # testing
        #metric_path = ['network_metrics', 'bursting_data', 'bursting_summary_data', 'mean_Burst_Rate']
        #output_dir = './output'
        #metric_name = 'mean_Burst_Rate'
        for metric_path in metric_paths:
            try:
                metric_path_parts = metric_path.split('.')
                if any('bursting' in part for part in metric_path_parts):
                    metric_name = f'{metric_path_parts[-3]}_{metric_path_parts[-1]}'
                else:
                    metric_name = f'{metric_path_parts[-2]}_{metric_path_parts[-1]}'
                plot_metric_heatmap(output_dir, metric_path_parts, metric_name, network_metrics_data, clean_grid, levels) #TODO: add reference data...
            except Exception as e:
                print(f"Error plotting heatmap for {metric_path}: {e}")
                continue


    if plot_heatmaps:
        plot_heat_maps(
            output_dir,
            input_dir, 
            num_workers=num_workers,
            levels=levels)
        
    # # combine the plots. put comp_grid on top, heat_map on bottom
    # from PIL import Image
    # comp_grid = Image.open(comp_grid_path)
    # heat_map = Image.open(heat_map_path)
    # combined = Image.new('RGB', (comp_grid.width, comp_grid.height + heat_map.height))
    # combined.paste(comp_grid, (0, 0))
    # combined.paste(heat_map, (0, comp_grid.height))
    # combined_path = os.path.join(output_dir, '_combined_summary_grid.png')
    # combined.save(combined_path)
    # print('Combined summary grid saved to', combined_path)
    # return combined_path