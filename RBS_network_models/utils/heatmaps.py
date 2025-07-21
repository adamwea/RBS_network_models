from RBS_network_models.utils.netpyne_helpers import import_evol_params
import os
import json
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
import time
import traceback
from fpdf import FPDF
from PIL import Image
import math
from os.path import dirname, basename


# subfunctions ===================================================================================================

def _derive_level_strings(levels):
    # make list of strings -level/2 to +level/2
    level_strings = [f'-{i}' for i in range(levels//2, 0, -1)] + [f'+{i}' for i in range(1, levels//2 + 1)]
    return level_strings

def _sort_found(found, levels):
    """
    Sorts the found file paths based on their level.
    
    Args:
        found (list): List of file paths to sort.
        levels (int): Number of levels to consider for sorting.
    
    Returns:
        list: Sorted list of file paths.
    """
    # sort by level in filename
    #found_sorted = sorted(found, key=lambda x: int(re.search(r'[-+]\d+', os.path.basename(x)).group()))
    
    # derive level strings
    level_strings = _derive_level_strings(levels)
    
    # remake found list by paths with level_strings in filename
    found_sorted = []
    for level_str in level_strings:
        fnd = False
        for file_path in found:
            if level_str in file_path:
                found_sorted.append(file_path)
                fnd = True
        if not fnd:
            # if no file found for this level, append a placeholder (e.g., None or empty string)
            found_sorted.append(None)
    
    # if more than levels, remove the rest
    # if len(found_sorted) > levels:
    #     found_sorted = found_sorted[:levels]
    
    return found_sorted

def _get_perms_per_param(param, permutations_dir, levels, query=None):
    ''' get number of permutations for a given param '''
    # init file list
    #files = []
    found = []
    
    #iterate through files in permutations_dir, get number of permutations from filename context
    num_permutations = 0
    param_elements = param.split('_')
    
    # debug
    # if 'std' in param and 'stdev' not in param:
    #     print(f"Debug: Searching for permutations of parameter '{param}' with elements {param_elements}")
    
    for root, _, files in os.walk(permutations_dir):
        for file in files:
            file_path = os.path.join(root, file)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            perm_label = parent_dir
            # check if all param_elements in perm_label
            #if all(element in perm_label for element in param_elements):
            if param in perm_label:
                #if all([element in perm_label for element in param_elements]):
                if '.npy' in file:
                    if query is None or query in file:
                        
                        # debug
                        if 'std' in file_path and 'std' not in param:
                            #print(f"Debug: Found permutation for parameter '{param}' in file '{file_path}'")
                            # HACK
                            continue # only add std files if param is std
                        
                        num_permutations += 1
                        found.append(file_path)
                        print('Found permutation for', param, 'in', file_path)
    
    # sort found
    found = _sort_found(found, levels)
    
    # return number of permutations found
    return num_permutations, found
        
def _get_clean_grid(input_dir, origin_dir, permutations_dir, levels, params=None, query=None):
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
    
    if params is None:
        raise ValueError("params must be specified")
    
    origin_network_data_path = os.path.join(origin_dir, query)
    if not os.path.exists(origin_network_data_path):
        raise FileNotFoundError(f"Original network data file not found at {origin_network_data_path}")
    
    # Initialize the grid for storing summary plot paths
    grid = {}
    
    params_to_exclude = [
        #'E_diam_mean', 'I_diam_mean', 'E_L_mean', 'I_L_mean', 'E_Ra_mean', 'I_Ra_mean',
    ]
    
    for param_name, param_value in params.items():
        # Skip parameters that are not lists or tuples of length 2
        if not isinstance(param_value, (list, tuple)):
            continue
        
        # Skip parameters that are not within the specified range
        if param_name in params_to_exclude:
            continue
        
        # Retrieve permutations for the current parameter
        #num_permutations, summary_paths = get_perms_per_param(param_name)
        _, summary_paths = _get_perms_per_param(param_name, permutations_dir, levels, query=query)
        
        num_permutations = len(summary_paths)
        
        if num_permutations == 0:
            continue  # Skip parameters with no variations
        
        # Insert the original summary plot at the middle index of the variations
        middle_idx = num_permutations // 2
        summary_paths.insert(middle_idx, origin_network_data_path)
        
        #debug - assert summary paths before and after origin network data path are equal
        if middle_idx > 0:
            assert len(summary_paths[:middle_idx]) == len(summary_paths[middle_idx + 1:]), \
                f"Mismatch in number of summary paths before and after origin network data path for {param_name}"
        
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

def _load_data_list(filepaths, param, mapped_keys):
    
    print(f"Processing network metric: {param}")
    heat_dict = {}
    # sort file paths alphabetically
    #filepaths = {k: v for k, v in sorted(filepaths.items(), key=lambda item: item[0])}
    
    for filepath in filepaths.values():
        try:
            print(f"Loading data from {filepath} for {param}")
            #path_parts = nestpath.split('.')
            full_data = np.load(filepath, allow_pickle=True).item()
            
            for path in mapped_keys:
                
                # Initialize the data list for this path
                data_list = []
                path_parts = path.split('.')

                # Traverse the nested dictionary structure to get the data
                data = full_data
                for part in path_parts:
                    data = data[part]
                
                # assert we only deal with data that are ints, floats, etc. No lists or dicts.
                if not isinstance(data, (int, float, np.number)):
                    print(f"Skipping {path} in {filepath} because it is not a numerical value")
                    continue
                
                # Create a dictionary entry for the path if it doesn't exist
                if path not in heat_dict:
                    heat_dict[path] = [data]
                else:
                    heat_dict[path].append(data)
                
            print(f"Loaded data for {param} from {filepath}")
            
        except Exception as e:
            #print(f"KeyError in network_metrics_data: {e}")
            print(f"Error accessing data in network_metrics_data: {e}")
            continue
    
    #print(f"Loaded {len(data_list)} data entries for {param} from {filepath}")
    print(f"Loaded {len(heat_dict)} data entries for {param}")
    
    #return data_list
    return heat_dict

def _extract_original_metric(data_list, metric_path):
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

def _compute_metric_bounds(heat_dict):
    
    all_metrics = []
    origin_metrics = []
    for param, data in heat_dict.items():
        # compute min and max metric values within 2 standard deviations
        if not data['metrics']:
            continue
        metric_values = data['metrics']
        all_metrics.extend(metric_values)
        
        # get origin metric, it will always be the middle element in the list
        if len(metric_values) > 0:
            origin_idx = len(metric_values) // 2
            origin_metric = metric_values[origin_idx]
            assert len(metric_values[:origin_idx]) == len(metric_values[origin_idx + 1:]), \
                f"Expected even number of metric values for {param}, found {len(metric_values)}"
            origin_metrics.append(origin_metric)
        
    if len(all_metrics) == 0:
        return np.nan, np.nan
    
    # assert that all values in origin_metrics are the same
    if len(set(origin_metrics)) > 1:
        print(f"Warning: Found multiple origin metrics: {set(origin_metrics)}")
    
    # choose origin metric, just get the first one
    original_metric = origin_metrics[0] if origin_metrics else np.nan
    
    # get min and max metric values
    max_metric = np.nanmax(all_metrics)
    min_metric = np.nanmin(all_metrics)
    
    
    return min_metric, max_metric, original_metric

def _plot_heatmap(heat_dict, mapped_key, output_dir, hwkargs=None):
    
    def _plot_boxes():
        
        def _print_values():
            """Prints the values of the parameters and metrics in the heatmap."""
            # format each part (2-decimal for numbers, else str)
            if isinstance(p, (int, float)):
                #p_str = f"{p:.2f}"
                # do scientific notation if necessary
                p_str = f"{p:.2e}" if abs(p) < 1e-3 or abs(p) > 1e3 else f"{p:.2f}"
            else:
                p_str = str(p)

            if isinstance(m, (int, float)):
                #m_str = f"{m:.2f}"
                m_str = f"{m:.2e}" if abs(m) < 1e-3 or abs(m) > 1e3 else f"{m:.2f}"
            else:
                m_str = str(m)

            # single label with arrow mapping
            if row_idx == 0:
                label = "param → metric\n" + f"{p_str} → {m_str}"
            else:
                label = f"{p_str} → {m_str}"

            # determine text color based on box color
            text_color = 'white' if color == 'black' else 'black'

            # draw it centered in the box
            axs[row_idx, col_idx].text(
                0.5, 0.5, label,
                ha="center", va="center",
                fontsize=10,
                color=text_color
            )
            
        # iterate through the heat_dict and plot each parameter
        row_idx = 0
        for param, data in heat_dict.items():
            metric_levels = data['metrics']
            param_levels = data['params']
            levels = len(metric_levels)-1
            
            # derive level strings
            level_strings = _derive_level_strings(levels)
            middle_level = levels // 2
            
            #insert 0 at the middle of level_strings
            if "0" not in level_strings:
                level_strings.insert(middle_level, '0')
            
            # iterate through param_levels and metric_levels
            col_idx = 0
            for lvl, p, m in zip(level_strings, param_levels, metric_levels):
                try:
                    
                    # get color for the metric value
                    if np.isnan(m):
                        color = 'black' # if metric is NaN, use black
                    else:
                        color = cmap(norm(m))
                    
                    # add a rectangle with the color to the subplot
                    axs[row_idx, col_idx].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
                    
                    # check if print_values is set to True in hwkargs
                    print_values = hwkargs.get('print_values', False) if hwkargs else False
                    if print_values: _print_values()
                    # turn off the axis for each subplot
                    axs[row_idx, col_idx].axis('off')
                    
                    # on the first row only, add -level/2 -> +level/2 as titles
                    if row_idx == 0: 
                        axs[row_idx, col_idx].set_title(lvl, fontsize=25, fontweight='bold')
                    else: 
                        pass
                        
                    print(f"Plotted {param} in row {row_idx}, column {col_idx} with value {m} and color {color}")
                    
                    # increment column index
                    col_idx += 1   
                except Exception as e:
                    #print(f"Error loading plot for key {key}: {e}")
                    print(f"Error loading plot for row {row_idx}, column {col_idx}: {e}")
                    traceback.print_exc()
                    print()
            
            #print
            print(f"Finished plotting {param} in row {row_idx}, with {col_idx} columns")        
            
            #increment row index
            row_idx += 1
            
            # #if any expected keys are missing from the row, fill them with black rectangles
            # #expected_keys = [0, 1, 2, 3, 4, 5, 6]
            # expected_keys = list(range(levels + 1))
            # # get middle most key of expected_keys
            # middle_key = expected_keys[len(expected_keys) // 2]
            # for key in expected_keys:
            #     #if key == 3:
            #     if key == middle_key:
            #         # offwhite rect
            #         axs[row_idx, key].add_patch(plt.Rectangle((0, 0), 1, 1, color=(0.875, 0.875, 0.875)))
            #         axs[row_idx, key].axis('off')
            #         continue
            #     if key not in clean_grid[param]['data']:
            #         axs[row_idx, key].add_patch(plt.Rectangle((0, 0), 1, 1, color='black'))
            #         axs[row_idx, key].axis('off')
            
            # for col_idx in range(levels + 1):
            #     axs[row_idx, col_idx].axis('off')
                
            # print(f"Plotted {param} in row {row_idx}")
    
    # get bounds for color mapping
    min_metric, max_metric, original_metric = _compute_metric_bounds(heat_dict)
    
    # apply pf color logic if needed
    # if doing pathfinder heatmap, adjust min and max values to maintain positive change -> red
    # negative change -> blue
    pf_color = hwkargs.get('pf_color', None) if hwkargs else None
    if pf_color:
        if min_metric >= 0:
            min_metric = -0.5
        if max_metric <= 0:
            max_metric = 0.5
        # NOTE: in this case, original_metrics should be zero, so this works.
    
    # Custom colormap: blue → offwhite → red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_cmap',
        [(0, 0, 1), (0.875, 0.875, 0.875), (1, 0, 0)],
        N=100
    )
    
    # create a colormap that goes from blue to offwhite to red - symmetric around the original metric
    # norm = (mcolors.CenteredNorm(vcenter=original_metric, halfrange=max(abs(min_metric - original_metric), abs(max_metric - original_metric)))
    #         if not np.isnan(original_metric) else mcolors.Normalize(vmin=min_metric, vmax=max_metric))

    # Use TwoSlopeNorm to allow asymmetric stretch with center at original_metric
    if not np.isnan(original_metric) and min_metric < original_metric < max_metric:
        norm = mcolors.TwoSlopeNorm(vmin=min_metric, vcenter=original_metric, vmax=max_metric)
    else:
        norm = mcolors.Normalize(vmin=min_metric, vmax=max_metric)
    
    # init figure and axes
    levels = len(heat_dict[list(heat_dict.keys())[0]]['metrics'])
    fig, axs = plt.subplots(len(heat_dict), levels, figsize=(2.25 * (levels), len(heat_dict)*.75))
    
    # adjust space between subplots
    fig.subplots_adjust(wspace=0.05, hspace=0.05)  # Adjust width and height spacing

    # plot color coded subplots for each parameter at each level
    _plot_boxes()
    
    # apply tight layout and adjust spacing for incoming labels
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.90, top=0.925)
    
    #improvements to labeling
    for row_idx, param in enumerate(heat_dict):
        pos = axs[row_idx, 0].get_position()
        param_text = param  # raw param

        # === Base replacements (these are cumulative) ===
        param_text = param_text.replace('propVelocity', r'\mathrm{v_{\mathrm{prop}}}')
        if 'prob' in param and 'probLengthConst' not in param:
            param_text = param_text.replace('prob', r'\mathrm{P(}') + ')'
        param_text = param_text.replace('probLengthConst', r'\mathrm{\lambda}')
        if 'weight' in param:
            param_text = param_text.replace('weight', r'\mathrm{w(}') + ')'
            
        # assume these are mean values, even if they are not labeled as such
        param_text = param_text.replace('gnabar_', r'\mathrm{\overline{g}_{\mathrm{Na}}}')
        param_text = param_text.replace('gkbar_', r'\mathrm{\overline{g}_{\mathrm{K}}}')
            
        # flat values assigned to tau1_exc, tau2_exc, tau1_inh, tau2_inh
        param_text = param_text.replace('tau1_exc', r'\mathrm{\tau_{1,\mathrm{exc}}}')
        param_text = param_text.replace('tau2_exc', r'\mathrm{\tau_{2,\mathrm{exc}}}')
        param_text = param_text.replace('tau1_inh', r'\mathrm{\tau_{1,\mathrm{inh}}}')
        param_text = param_text.replace('tau2_inh', r'\mathrm{\tau_{2,\mathrm{inh}}}')
        
        # replace _diam, _L, _Ra with LaTeX formatted strings
        param_text = param_text.replace('_diam', r'\mathrm{_{diam}}')
        param_text = param_text.replace('_L', r'\mathrm{_L}')
        param_text = param_text.replace('_Ra', r'\mathrm{_{Ra}}')

        # === Apply statistical wrapping ===
        base_text = param_text
        if '_mean' in param:
            base_text = base_text.replace('_mean', '')  # remove tag, keep LaTeX
            param_text = rf'$\mu_{{{base_text}}}$'
        elif '_std' in param or '_stdev' in param:
            base_text = base_text.replace('_stdev', '').replace('_std', '')
            param_text = rf'$\sigma_{{{base_text}}}$'
        # assume gnabar and gkbar are mu values if not std and not labeled otherwise
        elif 'gnabar' in param and '_std' not in param and '_stdev' not in param:
            #param_text = rf'$\mu_{{\mathrm{{{param_text}}}}}$'
            #base_text = latex_escape(base_text)
            base_text = param_text
            param_text = rf'$\mu_{{{base_text}}}$'
        elif 'gkbar' in param and '_std' not in param and '_stdev' not in param:
            #param_text = rf'$\mu_{{\mathrm{{{param_text}}}}}$'
            #base_text = latex_escape(base_text)
            base_text = param_text
            param_text = rf'$\mu_{{{base_text}}}$'
        else:
            # wrap the entire thing in LaTeX if not already wrapped
            param_text = rf'${param_text}$'

        fig.text(
            pos.x0 - 0.0125,
            pos.y0 + pos.height / 2,
            param_text, va='center', ha='right',
            fontsize=25,
            #fontweight='bold'
        )
    
    # add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)  # capture the colorbar!
    
    # Set colorbar ticks and labels
    cbar.set_ticks([min_metric, original_metric, max_metric])
    cbar.set_ticklabels([f'{min_metric:.2f}', f'{original_metric:.2f}', f'{max_metric:.2f}'])

    # Style tick marks and labels on the colorbar's Axes
    cbar.ax.tick_params(axis='y', labelsize=20, labelcolor='black', color='black')
    
    # set title for the figure
    #title = metric_name.split('metrics')[-1]
    title = mapped_key
    if title.startswith('_'): title = title[1:]
    fig.suptitle(title, fontsize=25, fontweight='bold')
    
    # debug
    # plt.savefig('debug_heatmap.png', dpi=100)
    # import sys
    # sys.exit()
    
    #output_path = os.path.join(output_dir, f'_heatmap_{metric_name}.png')
    #output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'heatmap_{mapped_key}.png')
    plt.savefig(output_path, dpi=300)
    #pdf_path = os.path.join(output_dir, f'heatmap_{mapped_key}.pdf')
    #plt.savefig(pdf_path)
    print(f'Saved heatmap to {output_path}')
    #print(f'Saved heatmap to {pdf_path}')
    return output_path  

def _prep_heat_dict(data_grid, mapped_key):
    """
    Prepares a dictionary for plotting heatmaps.
    
    Args:
        data (dict): The data to prepare.
        param (str): The parameter name.
        mapped_keys (list): The list of keys to map.
    
    Returns:
        dict: A dictionary with keys as paths and values as lists of data.
    """
    heat_dict = {}
    
    for param, data in data_grid.items():
        heat_dict[param] = {}
        
        # initialize lists for param_val and metric_value
        param_list = []
        metric_list = []
        
        # length of data
        data_length = len(data)
        print(f"Preparing heat_dict for {param} with {data_length} data points")
        
        for idx, datum in data.items():
            if datum is not None:
                                
                # get metric_value
                mapped_key_parts = mapped_key.split('.')
                d = datum
                for part in mapped_key_parts:
                    if part in d:
                        d = d[part]
                    else:
                        d = np.nan
                metric_value = d
                # if isinstance(d, (int, float, np.number)):
                #     metric_value = d
                # else:
                #     continue  # Skip if not a numerical value
                
                # get cfg_path                
                cfg_path = datum['cfg_path']
                
                # get param_val from cfg_path
                try:
                    with open(cfg_path, 'r') as f:
                        param_val = json.load(f)['simConfig'].get(param, np.nan)
                        #param_val = param_val.get(param, np.nan)  # Get the parameter value from the config
                except Exception as e:
                    print(f"Error loading config for {cfg_path}: {e}")
                    #param_val = "Error"
                    param_val = np.nan
                
                
                # debug
                if idx == 5 and param_val is np.nan:
                    print(f"Debug: param_val for {param} at index {idx} is NaN, cfg_path: {cfg_path}")
                
                # quality
                try:
                    #assert isinstance(param_val, (int, float, np.number)), f"Expected numerical value for {param}, got {param_val}"
                    assert isinstance(d, (int, float, np.number)), f"Expected numerical value for {mapped_key}, got {d}"
                    
                        
                    # append to lists
                    metric_list.append(metric_value)
                    param_list.append(param_val)
                except AssertionError as e:
                    #print(f"Error in data for {param} at index {idx}: {e}")
                    # append NaN if assertion fails
                    print(f"Skipping non-numerical data for {param} at index {idx}: {e}")
                    #param_list.append(np.nan)
                    param_list.append(param_val)  # still append the param_val, even if it's NaN
                    metric_list.append(np.nan)
                

            else:
                # If no data found for this index, append None
                param_list.append(np.nan)
                metric_list.append(np.nan)
                
        #if param_list and metric_lists are shorter than data_length, throw warning
        if len(param_list) < data_length or len(metric_list) < data_length:
            print(f"Warning: Parameter list or metric list for {param} is shorter than expected ({data_length}).")
            # fill with NaN to match data_length
            # param_list += [np.nan] * (data_length - len(param_list))
            # metric_list += [np.nan] * (data_length - len(metric_list))
            
        # debug
        if param_list[5] is np.nan:
            print(f"Debug: param_list for {param} at index 5 is NaN")
                
        # add to heat_dict
        heat_dict[param] = {
            'params': param_list,
            'metrics': metric_list,
        }
                
                
            
            
    
    return heat_dict

def _save_heatmap_to_cache(heat_dict, mapped_key, cache_dir=None):
    """
    Saves the heatmap data to a cache file.
    
    Args:
        heat_dict (dict): The heatmap data to save.
        mapped_key (str): The key for the heatmap data.
        cache_dir (str): The directory to save the cache file.
    """
    if cache_dir is None:
        print("No cache directory specified, skipping saving to cache.")
        return
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f'heatmap_{mapped_key}.json')
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(heat_dict, f)
        print(f"Saved heatmap data to cache at {cache_file}")
    except Exception as e:
        print(f"Error saving heatmap data to cache: {e}")

def _plot_heatmap_worker(mapped_key, heat_dict, output_dir, hkwargs):
    
    # use heat_dict to plot heatmap
    try:
        print(f"[Worker] Plotting heatmap for {mapped_key}")
        output_path = _plot_heatmap(heat_dict, mapped_key, output_dir, hwkargs=hkwargs)
        return output_path
    except Exception as e:
        print(f"[Worker] Error plotting heatmap for {mapped_key}: {e}")
        traceback.print_exc()
        return None

def _plot_heatmaps(data_grid, mapped_keys, output_dir, num_workers, parallel=False, hkwargs=None):
    """
    Plots heatmaps for each parameter in the grid.

    Args:
        data_grid (dict): Mapping from parameters to summary data.
        mapped_keys (list): Keys to plot heatmaps for.
        output_dir (str): Where to save the plots.
        num_workers (int): Number of parallel workers.
        parallel (bool): Whether to use multiprocessing.
        hkwargs (dict): Optional kwargs to pass into _plot_heatmap.
    """
    print("Plotting heatmaps for each parameter in the grid...")
    output_paths = []

    # Step 1: Prepare all heat_dicts serially
    jobs = []
    for mapped_key in mapped_keys:

        #init
        heat_dict = None
        exist_in_cache = False
        overwrite_cache = hkwargs.get('overwrite_cache', False) if hkwargs else False
        cache = hkwargs.get('cache', False) if hkwargs else False

        # if cache
        if cache:
            # Check if heat_dict is already prepared
            try:
                cache_dir = hkwargs.get('cache_dir', None)
                assert cache_dir is not None, "Cache directory must be specified in hkwargs"
                cache_file = os.path.join(cache_dir, f'heatmap_{mapped_key}.json')
                if os.path.exists(cache_file): exist_in_cache = True
            except AssertionError as e:
                print(f"Error checking cache for {mapped_key}: {e}")
                if cache_dir is None:
                    print("No cache directory specified, skipping cache check.")
            
            # if heat_dict exists in cache, load it
            if exist_in_cache and overwrite_cache is False:
                try:
                    print(f"Loading heat_dict for {mapped_key} from cache...")
                    with open(cache_file, 'r') as f:
                        heat_dict = json.load(f)
                except Exception as e:
                    print(f"Error loading heat_dict from cache for {mapped_key}: {e}")
                    traceback.print_exc()
                    print()
                    continue
        
        # if heat_dict could not be loaded
        if heat_dict==None:
            try:
                print(f"Generating heat_dict for {mapped_key}...")
                assert data_grid is not None, "data_grid must not be None for heatmap generation"
                heat_dict = _prep_heat_dict(data_grid, mapped_key)
                
                # save heat_dict to cache if cache_dir is specified
                if cache_dir is not None:
                    _save_heatmap_to_cache(heat_dict, mapped_key, cache_dir)
                
                jobs.append((mapped_key, heat_dict))
            except Exception as e:
                print(f"Error preparing heat_dict for {mapped_key}: {e}")
                traceback.print_exc()
                print()
                continue

    # Step 2: Plot heatmaps
    if parallel:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_plot_heatmap_worker, mapped_key, heat_dict, output_dir, hkwargs)
                for mapped_key, heat_dict in jobs
            ]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    output_paths.append(result)
    else:
        for mapped_key, heat_dict in jobs:
            try:
                print(f"Plotting heatmap for {mapped_key}")
                output_path = _plot_heatmap(heat_dict, mapped_key, output_dir, hwkargs=hkwargs)
                output_paths.append(output_path)
            except Exception as e:
                print(f"Error plotting heatmap for {mapped_key}: {e}")
                traceback.print_exc()
                print()
                continue

    return output_paths

def plot_metric_heatmap_v3(output_dir, metric_path, metric_name, network_metrics_data, clean_grid, levels):
    """
    Plots heatmaps for a specified network metric.
    """
    print(f"Plotting summary grid for {metric_name} with color gradient")
    #data_list = [data['data'] for data in network_metrics_data]
    data_list = []
    for netmet in network_metrics_data:
        try:
            data_list.append(netmet['data'])
        except Exception as e:
            #print(f"KeyError in network_metrics_data: {e}")
            print(f"Error accessing data in network_metrics_data: {e}")
            continue
    if len(data_list) == 0:
        print(f"No data found for {metric_name} in network_metrics_data")
        raise ValueError(f"No data found for {metric_name} in network_metrics_data")
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
                sim_data_path = data['sim_data_path']
                sim_cfg_path = sim_data_path.replace('_data.pkl', '_cfg.json')
                with open(sim_cfg_path, 'r') as f:
                    sim_cfg = json.load(f)
                param_value = sim_cfg.get(param, None)
                    
                for path_part in metric_path:
                    if 'network_metrics' in path_part:
                        continue
                    metric_value = metric_value.get(path_part, np.nan)
                
                color = cmap(norm(metric_value))
                axs[row_idx, key].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
                #axs[row_idx, key].text(0.5, 0.5, f'{metric_value:.2f}', ha='center', va='center', fontsize=12)
                
                # inside your loop, after you’ve defined param_value and metric_value…

                # format each part (2-decimal for numbers, else str)
                if isinstance(param_value, (int, float)):
                    #p_str = f"{param_value:.2f}"
                    # do scientific notation if necessary
                    p_str = f"{param_value:.2e}" if abs(param_value) < 1e-3 or abs(param_value) > 1e3 else f"{param_value:.2f}"
                else:
                    p_str = str(param_value)

                if isinstance(metric_value, (int, float)):
                    #m_str = f"{metric_value:.2f}"
                    m_str = f"{metric_value:.2e}" if abs(metric_value) < 1e-3 or abs(metric_value) > 1e3 else f"{metric_value:.2f}"
                else:
                    m_str = str(metric_value)

                # single label with arrow mapping
                label = "param → metric\n" + f"{p_str} → {m_str}"

                # draw it centered in the box
                axs[row_idx, key].text(
                    0.5, 0.5, label,
                    ha="center", va="center",
                    fontsize=12
                )

                axs[row_idx, key].axis('off')
                
                sim_data_path = data['sim_data_path']
                permuted_value = next((netmet['cfg'].get(param, None) for netmet in network_metrics_data if netmet['data']['sim_data_path'] == sim_data_path), None)
                
                middle_level = levels // 2
                if row_idx == 0: # only on first ?
                    if permuted_value is not None:
                        #axs[row_idx, key].set_title(f'@{round(permuted_value, 3)}', fontsize=14)
                        # get the column position relative to origin (in the middle) and show that instead of perm_value
                        #level_pos = int(re.search(r'\d+$', key).group())
                        #level_diff = key - 3
                        level_diff = key - middle_level
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
        #expected_keys = [0, 1, 2, 3, 4, 5, 6]
        expected_keys = list(range(levels + 1))
        # get middle most key of expected_keys
        middle_key = expected_keys[len(expected_keys) // 2]
        for key in expected_keys:
            #if key == 3:
            if key == middle_key:
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
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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

def _find_metric_paths(grid, max_samples=3, parent_path="network_data"):
    """
    Extract metric paths from a grid of .npy file paths without fully loading large data.
    
    Args:
        grid (dict): Dictionary where values are dicts of {idx: path to .npy files}.
        max_samples (int): Max number of files per param to inspect.
        parent_path (str): Base path prefix used in metric path construction.
        
    Returns:
        dict: Mapping of parameter names to list of detected metric paths.
    """
    keys_to_ignore = {
        "std", "cov", "median", "burst_ids", "burst_part", "burst_parts",
        ".data", "num_bursts", "unit_metrics", "gids", "spiking_metrics_by_unit",
        "spiking_times_by_unit", ".unit_metrics", "unit_types", "min", "max",
        "simData", "popData", "cellData", 
        #'E_diam_mean', 'I_diam_mean',
        #'E_L_mean', 'I_L_mean', 'E_Ra_mean', 'I_Ra_mean'
    }

    def is_numeric_key_dict(d):
        return isinstance(d, dict) and all(re.match(r"^\d+$", str(k)) for k in d)

    def recurse(obj, path, found):
        if isinstance(obj, dict):
            if is_numeric_key_dict(obj):
                if not any(ign in path for ign in keys_to_ignore):
                    found.add(path)
                return
            for k, v in obj.items():
                recurse(v, f"{path}.{k}", found)
        elif isinstance(obj, (int, float)):
            if not any(ign in path for ign in keys_to_ignore):
                found.add(path)
        elif isinstance(obj, list):
            if all(isinstance(x, (int, float)) for x in obj):
                if not any(ign in path for ign in keys_to_ignore):
                    found.add(path)

    # Main processing
    all_metric_paths = {}

    for param, idx_path_map in grid.items():
        found_paths = set()
        sample_count = 0

        for idx in sorted(idx_path_map.keys()):
            if sample_count >= max_samples:
                break

            path = idx_path_map[idx]
            if not os.path.isfile(path):
                continue

            try:
                loaded = np.load(path, allow_pickle=True) 
                obj = loaded.item() if isinstance(loaded, np.ndarray) and loaded.shape == () else loaded
                # assume structure: {'data': {...}}
                if isinstance(obj, dict): #and 'data' in obj:
                    recurse(obj, parent_path, found_paths)
                    sample_count += 1
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue

        all_metric_paths[param] = sorted(found_paths)

    return all_metric_paths

def _find_origin_paths(origin_dir, max_samples=3, parent_path="network_data"):
    """
    Extract metric paths from a grid of .npy file paths without fully loading large data.
    
    Args:
        grid (dict): Dictionary where values are dicts of {idx: path to .npy files}.
        max_samples (int): Max number of files per param to inspect.
        parent_path (str): Base path prefix used in metric path construction.
        
    Returns:
        dict: Mapping of parameter names to list of detected metric paths.
    """
    keys_to_ignore = {
        "std", "cov", "median", "burst_ids", "burst_part", "burst_parts", ".data", 
        #"num_bursts", 
        "unit_metrics", 
        "gids", 
        "spiking_metrics_by_unit",
        "spiking_times_by_unit", ".unit_metrics", "unit_types", 
        "min", "max",
        "simData", "popData", "cellData", 
        #'E_diam_mean', 'I_diam_mean',
        #'E_L_mean', 'I_L_mean', 'E_Ra_mean', 'I_Ra_mean'
    }

    def is_numeric_key_dict(d):
        return isinstance(d, dict) and all(re.match(r"^\d+$", str(k)) for k in d)

    def recurse(obj, path, found):
        if isinstance(obj, dict):
            if is_numeric_key_dict(obj):
                if not any(ign in path for ign in keys_to_ignore):
                    found.add(path)
                return
            for k, v in obj.items():
                if path is not None:
                    recurse(v, f"{path}.{k}", found)
                else:
                    recurse(v, k, found)
        elif isinstance(obj, (int, float)):
            if not any(ign in path for ign in keys_to_ignore):
                found.add(path)
        elif isinstance(obj, list):
            if all(isinstance(x, (int, float)) for x in obj):
                if not any(ign in path for ign in keys_to_ignore):
                    found.add(path)

    # Main processing
    all_metric_paths = {}
    
    found_paths = set()

    # for param, idx_path_map in grid.items():
    #     found_paths = set()
    #     sample_count = 0

    #     for idx in sorted(idx_path_map.keys()):
    #         if sample_count >= max_samples:
    #             break

    #path = idx_path_map[idx]
    path = os.path.join(origin_dir, 'network_data.npy')
    if not os.path.isfile(path):
        #continue
        raise FileNotFoundError(f"File {path} does not exist.")

    try:
        loaded = np.load(path, allow_pickle=True) 
        obj = loaded.item() if isinstance(loaded, np.ndarray) and loaded.shape == () else loaded
        # assume structure: {'data': {...}}
        if isinstance(obj, dict): #and 'data' in obj:
            recurse(obj, parent_path, found_paths)
            #sample_count += 1
    except Exception as e:
        print(f"Error reading {path}: {e}")
        #continue

    #     all_metric_paths[param] = sorted(found_paths)

    # return all_metric_paths

    return sorted(found_paths)

def _metrics_loader(network_metrics_file, use_memmap=True):
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
        print(f'Loading network metrics from {network_metrics_file}...')
        
        # Load network data with optional memory mapping
        network_data = np.load(network_metrics_file, mmap_mode='r' if use_memmap else None, allow_pickle=True).item()
        
        # if sim_data_path is not in network_data, add it
        # sim_data_path = network_data['inputs'].get('sim_data_path', None)
        # if sim_data_path is None:
        #     sim_data_dir = os.path.dirname(os.path.dirname(network_metrics_file))
        #     # find a file in sim_data_dir that ends with '_data.pkl'
        #     sim_data_files = glob.glob(os.path.join(sim_data_dir, '*_data.pkl'))
        #     if sim_data_files:
        #         sim_data_path = sim_data_files[0]
        #     else:
        #         raise FileNotFoundError(f'No simulation data file found in {sim_data_dir} matching *_data.pkl')
        #     network_data['sim_data_path'] = sim_data_path
        
        # update filepath in network_data to itself
        network_data['filepath'] = network_metrics_file
        
        # derive sim_data_path from the network_metrics_file, confirm it exists, put it in network_data
        sim_data_dir = os.path.dirname(network_metrics_file)
        sim_data_path = glob.glob(os.path.join(sim_data_dir, '*_data.pkl'))
        if not sim_data_path:
            print(f'No simulation data file found in {sim_data_dir} matching *_data.pkl')
            print(f'Looking for *_data.json instead...')
            sim_data_path = glob.glob(os.path.join(sim_data_dir, '*_data.json'))
        if not sim_data_path:
            raise FileNotFoundError(f'No simulation data file found in {sim_data_dir} matching *_data.pkl or *_data.json')
        sim_data_path = sim_data_path[0]  # Take the first match
        network_data['sim_data_path'] = sim_data_path
        
        # derive cfg_path from the network_metrics_file, confirm it exists, put it in network_data
        cfg_path = glob.glob(os.path.join(sim_data_dir, '*_cfg.json'))
        if not cfg_path:
            raise FileNotFoundError(f'No configuration file found in {sim_data_dir} matching *_cfg.json')
        cfg_path = cfg_path[0]  # Take the first match
        network_data['cfg_path'] = cfg_path
        
        return network_data
        
        # # Locate configuration file
        # perm_dir_parent = os.path.dirname(network_metrics_file)
        # perm_dir_gp = os.path.dirname(perm_dir_parent)
        
        # # Attempt to find JSON config file
        # cfg_file = glob.glob(f'{perm_dir_gp}/*_cfg.json')
        # try:
        #     if not cfg_file:
        #         raise FileNotFoundError(f'No cfg file found for {network_metrics_file}')
        #     with open(cfg_file[0], 'r') as f:
        #         cfg_file = json.load(f)
        # except:
        #     # Fall back to finding and loading a pickle file
        #     sim_file = glob.glob(f'{perm_dir_gp}/*_data.pkl')
        #     if not sim_file:
        #         raise FileNotFoundError(f'No alternative config found for {network_metrics_file}')
        #     from netpyne import sim  # Ensure netpyne is imported properly
        #     #sim.loadSimCfg(sim_file[0])
        #     sim.load(sim_file[0])
        #     cfg_file = deepcopy(sim.cfg.todict())
        #     sim.clearAll()
        
        # return {
        #     'data': network_data,
        #     'cfg': cfg_file
        # }
    except Exception as e:
        print(f'Error loading {network_metrics_file}: {e}')
        return {'error': str(e)}

def _load_network_data(origin_dir, permutation_dir, num_workers=None, use_threads=False, use_memmap=False, debug_limited_load=False, limited_load_num=5):
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
    
    # Locate network metrics file
    network_metrics_files = glob.glob(os.path.join(permutation_dir, '**', 'network_data.npy'), recursive=True)
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
    
    # Initialize counters and data structures
    total_files = len(network_metrics_files)
    completed_files = 0
    network_metrics_data = {}
    
    # Limit the number of files to load for debugging purposes
    if debug_limited_load:
        network_metrics_files = network_metrics_files[:limited_load_num-1]  # Limit to limited_load_num for testing
        print(f"Debug mode: limiting to {len(network_metrics_files)} files for testing.")
        
    # append origin data to the list
    origin_file = os.path.join(origin_dir, 'network_data.npy')
    assert os.path.isfile(origin_file), f"Origin file {origin_file} does not exist."
    network_metrics_files.append(origin_file)
     
    # run the loading process   
    num_workers = total_files if total_files < available_cpus else available_cpus
    results = []
    print(f"Using {num_workers} {'threads' if use_threads else 'processes'} to load {len(network_metrics_files)} network metrics files.")
    with Executor(max_workers=num_workers) as executor:
        futures = {executor.submit(_metrics_loader, file, use_memmap): file for file in network_metrics_files}
        for future in as_completed(futures):
            result = future.result()
            #if 'error' not in result:
                #network_metrics_data[os.path.basename(futures[future])] = result
            results.append(result)
            completed_files += 1
            print(f"Completed {completed_files} out of {total_files}")
    
    #return network_metrics_data
    return results

def _map_data_to_grid(network_data_results, grid):
    data_grid = {}
    for param in grid:
        data_grid[param] = {}
        for idx, path in grid[param].items():
            # find the corresponding data in the loaded results
            for result in network_data_results:
                if result['filepath'] == path:
                    data_grid[param][idx] = result
                    break
            else:
                data_grid[param][idx] = None  # No data found for this path
                #print(f"Warning: No data found for {param} at index {idx} with path {path}")
    # if cache:
    #     # Cache the data grid to avoid recomputing it
    #     import pickle
    #     cache_file = os.path.join(os.getcwd(), 'data_grid_cache.pkl')
    #     with open(cache_file, 'wb') as f:
    #         pickle.dump(data_grid, f)
    #     print(f"Data grid cached to {cache_file}")
                
    return data_grid

def _convert_png_to_jpeg(png_path, temp_folder="temp_jpegs"):
    os.makedirs(temp_folder, exist_ok=True)
    with Image.open(png_path) as img:
        rgb_img = img.convert('RGB')
        jpeg_path = os.path.join(temp_folder, os.path.basename(png_path).replace('.png', '.jpg'))
        rgb_img.save(jpeg_path, 'JPEG', quality=95)
        return jpeg_path

def _find_best_grid(num_images, available_width, available_height, label_height=5, spacing=1):
    best_layout = None
    best_score = float('inf')  # score based on squareness

    for rows in range(1, num_images + 1):
        cols = math.ceil(num_images / rows)

        cell_width = (available_width - (cols - 1) * spacing) / cols
        cell_height = (available_height - (rows - 1) * spacing) / rows

        if cell_width <= 0 or cell_height <= label_height:
            continue  # can't fit even the image+label

        # Heuristic: prefer grid with near-1 aspect ratio cells
        score = abs(rows - cols)

        if score < best_score:
            best_score = score
            best_layout = (rows, cols, cell_width, cell_height)

    if best_layout is None:
        raise ValueError("Cannot fit images on page with given margin/spacing.")

    return best_layout

def _add_landscape_summary_page(pdf, jpeg_paths, png_paths, page_size=(297, 210), margin=5, spacing=1):
    """
    Adds a landscape summary page with a grid of thumbnails and labels.
    Dynamically fits the grid based on number of images and page dimensions.
    """
    from os.path import dirname, basename
    from PIL import Image

    assert len(jpeg_paths) == len(png_paths), "jpeg_paths and png_paths must match in order and length"

    page_width, page_height = page_size
    pdf.add_page(orientation='L')

    LABEL_HEIGHT = 5
    BOTTOM_MARGIN_PAD = 5
    SAFE_LABEL_SPACE = LABEL_HEIGHT + BOTTOM_MARGIN_PAD

    available_width = page_width - 2 * margin
    available_height = page_height - 2 * margin - SAFE_LABEL_SPACE  # reserve bottom space

    num_images = len(jpeg_paths)

    # Dynamically find best (rows, cols) layout
    try:
        best_layout = _find_best_grid(
            num_images=num_images,
            available_width=available_width,
            available_height=available_height,
            label_height=LABEL_HEIGHT,
            spacing=spacing
        )
    except ValueError as e:
        print(f"Error finding best grid layout: {e}")
        raise ValueError("Cannot fit images in the summary page. Please check your image sizes or margins.")

    rows, cols, cell_width, cell_height = best_layout

    pdf.set_font("Arial", size=5)

    for idx, (jpeg_path, png_path) in enumerate(zip(jpeg_paths, png_paths)):
        row = idx // cols
        col = idx % cols

        x_cell = margin + col * (cell_width + spacing)
        y_cell = margin + row * (cell_height + spacing)

        with Image.open(jpeg_path) as img:
            img_width, img_height = img.size
            img_aspect = img_width / img_height
            max_img_height = cell_height - LABEL_HEIGHT
            cell_aspect = cell_width / max_img_height

            if img_aspect > cell_aspect:
                display_width = cell_width
                display_height = display_width / img_aspect
            else:
                display_height = max_img_height
                display_width = display_height * img_aspect

            x_offset = x_cell + (cell_width - display_width) / 2
            y_offset = y_cell

            pdf.image(jpeg_path, x=x_offset, y=y_offset, w=display_width, h=display_height)

            # Add label inside the cell
            label_y = y_offset + display_height
            if label_y + LABEL_HEIGHT > page_height - margin:
                label_y = page_height - margin - LABEL_HEIGHT

            label = basename(dirname(png_path))
            # pdf.set_xy(x_cell, label_y)
            # pdf.cell(cell_width, LABEL_HEIGHT, label, align='C')
            
            label_x = x_cell + (cell_width / 2) - (pdf.get_string_width(label) / 2)
            pdf.text(label_x, label_y + LABEL_HEIGHT, label)  # draw at exact position
            print(f"Added image {jpeg_path} at ({x_cell}, {y_cell}) with label '{label}'")

def _generate_pathfinder_report(png_files, output_pdf_path="pathfinder_report.pdf"):
    if not png_files:
        print("No simulation plots found for pathfinder report. Skipping report generation.")
        raise ValueError("No simulation plots found for pathfinder report. Skipping report generation.")

    pdf = FPDF(unit="mm", format="A4")
    page_width, page_height = 210, 297

    # Step 1: Convert all PNGs to JPEGs first
    jpeg_paths = [_convert_png_to_jpeg(p) for p in png_files]

    # Step 2: First page — first image (portrait)
    first_jpeg = jpeg_paths[0]
    first_png = png_files[0]

    with Image.open(first_jpeg) as img:
        img_width, img_height = img.size
        img_aspect = img_width / img_height
        page_aspect = page_width / page_height

        if img_aspect > page_aspect:
            display_width = page_width
            display_height = display_width / img_aspect
        else:
            display_height = page_height
            display_width = display_height * img_aspect

        x_offset = (page_width - display_width) / 2
        y_offset = 10

        pdf.add_page()
        pdf.image(first_jpeg, x=x_offset, y=y_offset, w=display_width, h=display_height)
        pdf.set_font("Arial", size=7)
        pdf.set_y(y_offset + display_height + 5)
        pdf.multi_cell(0, 10, f"Source: {os.path.dirname(first_png)}", align='L')

    print(f"Added first image: {first_jpeg}")

    # Step 3: Second page — landscape summary of all
    _add_landscape_summary_page(pdf, jpeg_paths[1:], png_files[1:]) # remove query image which should be first
    print("Added landscape summary page.")

    # Step 4: Remaining individual images (starting from 2nd)
    for png_path, jpeg_path in zip(png_files[1:], jpeg_paths[1:]):
        with Image.open(jpeg_path) as img:
            img_width, img_height = img.size
            img_aspect = img_width / img_height
            page_aspect = page_width / page_height

            if img_aspect > page_aspect:
                display_width = page_width
                display_height = display_width / img_aspect
            else:
                display_height = page_height
                display_width = display_height * img_aspect

            x_offset = (page_width - display_width) / 2
            y_offset = 10

            pdf.add_page()
            pdf.image(jpeg_path, x=x_offset, y=y_offset, w=display_width, h=display_height)
            pdf.set_font("Arial", size=7)
            pdf.set_y(y_offset + display_height + 5)
            pdf.multi_cell(0, 10, f"Source: {os.path.dirname(png_path)}", align='L')

        print(f"Added image: {jpeg_path}")

    pdf.output(output_pdf_path)
    print(f"Report saved to: {output_pdf_path}")

# main ============================================================================================

def plot_heatmaps(output_dir, input_dir, num_workers=None, levels=6, params=None, **hkwargs):
    
    # main ===================================================================================================
    
    # init output directory
    #output_dir = hkwargs.get('output_dir', None)
    if output_dir is None:
        raise ValueError("output_dir must be specified in hkwargs.")
    
    # import params if passed as a path string
    if isinstance(params, str):
        params = import_evol_params(params)
        
    # look for expected folder structure
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    
    ## look for '_origin' dir and 'permutations' dir
    permutations_dir = os.path.join(input_dir, 'permutations')
    origin_dir = os.path.join(input_dir, '_origin')
    if not os.path.exists(permutations_dir):
        raise FileNotFoundError(f"Permutations directory {permutations_dir} does not exist.")
    if not os.path.exists(origin_dir):
        raise FileNotFoundError(f"Origin directory {origin_dir} does not exist.")
    
    # sort param keys alphabetically
    if params is not None:
        if isinstance(params, dict):
            params = {k: v for k, v in sorted(params.items())}
        else:
            raise TypeError(f"params should be a dict or a list, got {type(params)}")
    
    # get dict of paths for matrix
    grid = _get_clean_grid(input_dir, origin_dir, permutations_dir, params=params, levels=levels, query='network_data.npy') # prepare grid of network_data.npy file paths
    
    # map out nested keys in .npy files without fully loading them
    # this is to avoid loading large files into memory, which can be slow and memory-intensive
    # new function goes here ....
    #mapped_keys = _find_metric_paths(grid)
    
    # map out nested keys based on origin data
    mapped_keys = _find_origin_paths(origin_dir, max_samples=3, parent_path=None)
    
    # remove paths including keys '.inputs.'
    mapped_keys = [key for key in mapped_keys if 'inputs.' not in key]
        
    # if cache is not available, load network data from files
    print("Loading network data from files...")
    
    # cache logic
    cache = hkwargs.get('cache', False)
    overwrite_cache = hkwargs.get('overwrite_cache', False)
    if not cache or overwrite_cache:
    
        # load all .npy objects in the grid
        # this is to ensure that we have all the data we need to plot the heatmaps
        num_workers = hkwargs.get('num_workers', num_workers)
        network_data_results = _load_network_data(
            origin_dir,
            permutations_dir,
            num_workers=num_workers,
            use_threads=True,
            use_memmap=False,
            debug_limited_load=hkwargs.get('debug_limited_heatmap', False),
            limited_load_num=hkwargs.get('limited_load_num', 5)
            )
        
        # map the loaded data to a dictionary matching the grid structure
        data_grid = _map_data_to_grid(network_data_results, grid)

        # print the number of metric paths found
        print(f"Found {len(mapped_keys)} metric paths in origin data.")
        
    else:
        data_grid = None # heat_maps will try to be loaded from cache if available, so we can pass None here to skip loading data again.
        print("Using cached data grid. If you want to reload the data, set cache=False in hkwargs.")
        print("WARNING: if cached data is not available, this will raise an error.")
        
    # plot heatmaps
    parallel = hkwargs.get('parallel', False)
    # NOTE: data_grid can be passed as None if loading from cache
    output_paths = _plot_heatmaps(data_grid, mapped_keys, output_dir, num_workers, parallel=parallel, hkwargs=hkwargs)
    
    print(f"Heatmaps saved to {output_dir}")
    return output_paths

def plot_pathfinder(queries, **pkwargs):
    
    # load cached files
    cache_dir = pkwargs.get('cache_dir', None)
    assert cache_dir is not None, "cache_dir not specified in pkwargs. Cached files are required for pathfinder plotting."
    if not os.path.exists(cache_dir):
        print(f"Cached directory {cache_dir} does not exist. Please provide a valid cached directory.")
        raise FileNotFoundError(f"Cached directory {cache_dir} does not exist. Please provide a valid cached directory.")

    # get json files in cache_dir
    json_files = glob.glob(os.path.join(cache_dir, '*.json'))
    if not json_files:
        print(f"No JSON files found in cache directory {cache_dir}. Please provide a valid cache directory with JSON files.")
        raise FileNotFoundError(f"No JSON files found in cache directory {cache_dir}. Please provide a valid cache directory with JSON files.")

    # load json files
    json_data = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                key = os.path.basename(json_file).replace('.json', '')
                key = key.replace('heatmap_', '')  # remove heatmap_ prefix if present
                json_data[key] = data
        except Exception as e:
            print(f"Error loading JSON file {json_file}: {e}")
            continue

    print(f"Loaded {len(json_data)} JSON files from cache directory {cache_dir}.")
    
    
    # filter data based on queries
    heats = {}
    for mapped_key, query in queries.items():
        if mapped_key not in json_data:
            #print(f"Warning: No data found for query '{mapped_key}' in cached JSON files.")
            continue
        
        # get the data for the query
        #data = json_data[mapped_key]
        heats[mapped_key] = json_data[mapped_key]
        
        # plot the heatmap
        #output_path = _plot_heatmap_from_data(data, query, pkwargs)
        #print(f"Heatmap for query '{mapped_key}' saved to {output_path}")
    
    # normalize data to origin in middle most position of each row
    # this is done by finding the original simulation data and using it as the center of the
    # heatmap, with the permuted data on either side.
    norm_heats = {}
    for mapped_key, data in heats.items():
        # find the original simulation data
        
        norm_heats[mapped_key] = {}
        for param, datum in data.items():
            m = datum['metrics']
            p = datum['params']
            
            # get the length of m to determine the middle index, total length should be odd with a distinct center idx
            m_length = len(m)
            if m_length % 2 == 0:
                print(f"Warning: Metric length for {mapped_key} is even ({m_length}), which may not center correctly.")
            middle_idx = m_length // 2
            # find the original simulation data
            origin = m[middle_idx]
            
            # normalize the entire list to the origin
            for idx, value in enumerate(m):
                if isinstance(value, (int, float)):
                    #m[idx] = value/origin * 100 - 100  # normalize to origin, so origin is 0%
                    m[idx] = (value - origin) / origin * 100
                # elif isinstance(value, list):
                #     m[idx] = [v/origin * 100 for v in value]
                else:
                    print(f"Warning: Unsupported data type {type(value)} in metric list for {mapped_key}. Skipping normalization.")
            normalized_m = m
            #print()
            norm_heats[mapped_key][param] = {
                'params': p,
                'metrics': normalized_m,
                'origin': origin
            }
            
    # now apply query filters
    paths_found = {}
    for query_key, query in queries.items():
        for mapped_key, norm_heat in norm_heats.items():
            if mapped_key == query_key:
                paths_found[mapped_key] = {}
                for param, data in norm_heat.items():
                    filtered_metrics = data['metrics'].copy()  # start with a copy of the metrics
                    m_length = len(filtered_metrics)
                    if m_length % 2 == 0:
                        print(f"Warning: Filtered metric length for {mapped_key} is even ({m_length}), which may not center correctly.")
                    middle_idx = m_length // 2
                    origin = filtered_metrics[middle_idx]  # this is the original simulation data

                    # apply filters based on the query
                    if 'greater_than' in query:
                        # filter metrics greater than a certain value
                        threshold = query['greater_than']
                        
                        #data['metrics'] = [m for m in data['metrics'] if m > threshold]
                        filtered_metrics = [m if m > threshold else np.nan for m in filtered_metrics]
                        # paths_found[mapped_key][param] = {
                        #     'params': data['params'],
                        #     'metrics': filtered_metrics,
                        #     'origin': data['origin']
                        # }
                    if 'less_than' in query:
                        # filter metrics less than a certain value
                        threshold = query['less_than']
                        #for param, data in norm_heat.items():
                            #data['metrics'] = [m for m in data['metrics'] if m < threshold]
                        filtered_metrics = [m if m < threshold else np.nan for m in filtered_metrics]
                            # paths_found[mapped_key][param] = {
                            #     'params': data['params'],
                            #     'metrics': filtered_metrics,
                            #     'origin': data['origin']
                            # }
                    if 'abs_less_than' in query:
                        # filter metrics based on absolute change
                        threshold = query['abs_less_than']
                        #for param, data in norm_heat.items():
                        filtered_metrics = [m if abs(m) < threshold else np.nan for m in filtered_metrics]
                    
                    if 'abs_greater_than' in query:
                        # filter metrics based on absolute change
                        threshold = query['abs_greater_than']
                        #for param, data in norm_heat.items():
                        filtered_metrics = [m if abs(m) > threshold else np.nan for m in filtered_metrics]
                
                    # make sure filter doesnt change origin

                    filtered_metrics[middle_idx] = origin # should be zero, but just to be sure.
                    
                    # store the filtered metrics in paths_found
                    paths_found[mapped_key][param] = {
                        'params': data['params'],
                        'metrics': filtered_metrics,
                        'origin': data['origin']
                    }
            
    # now create a synthesized heatmap for the query where filters are applied
    paths_found['query'] = {}
    for mapped_key, pf in paths_found.items():
        if mapped_key == 'query':
            continue
        for param, data in pf.items():
            m = data['metrics']
            val_count = 0 # number of values contributing to average at each position, for running average.
            if param not in paths_found['query']:
                
                # create dummy list full of NaNs for param
                if isinstance(data['params'], list):
                    p = [np.nan] * len(data['metrics'])
                
                paths_found['query'][param] = {
                    'params': p,
                    'metrics': m,
                    #'origin': data['origin']
                }
                val_count += 1
            else:
                path_metrics = paths_found['query'][param]['metrics']
                
                # update path_metrics one element at a time
                # where both current vals and incoming vals are not NaN, average them out.
                # else if one or both is NaN, update the value to NaN.
                for i in range(len(path_metrics)):
                    
                    # debug
                    if np.isnan(path_metrics[i]) or np.isnan(m[i]) and i == 6:
                        print(f"Debug: NaN found at index {i} for param {param}")
                        
                    if not np.isnan(path_metrics[i]) and not np.isnan(m[i]):
                        #path_metrics[i] = (path_metrics[i] + metrics[i]) / 2
                        #running average
                        path_metrics[i] = (path_metrics[i] * val_count + m[i]) / (val_count + 1)
                        val_count += 1
                    else:
                        path_metrics[i] = np.nan
            paths_found['query'][param]['metrics']=paths_found['query'][param]['metrics']
            
    # finally, use _plot_heatmaps to plot the pathfinder heatmap
    output_dir = pkwargs.get('output_dir', None)
    if output_dir is None:
        raise ValueError("output_dir must be specified in pkwargs.")
    
    # get the number of dirs in the output_dir and create a new dir for the pathfinder heatmap
    # naming scheme q1, q2, q3, etc. for sequential queries
    existing_dirs = glob.glob(os.path.join(output_dir, 'q*'))
    query_num = len(existing_dirs) + 1
    output_dir = os.path.join(output_dir, f'q{query_num}')
    #os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating output directory for pathfinder heatmap: {output_dir}")
    
    # plot the heatmap
    # for mapped_key, heat_dict in jobs:
    heat_dict = paths_found['query']
    mapped_key = 'query'  # since this is a synthesized heatmap for the query
    pkwargs['pf_color'] = True # tell _plot_heatmap to use simpler pathfinder color scheme - red for positive change, blue for negative change.
    try:
        print(f"Plotting pathfinder heatmap")
        output_path = _plot_heatmap(heat_dict, mapped_key, output_dir, hwkargs=pkwargs)
        #output_paths.append(output_path)
    except Exception as e:
        print(f"Error plotting heatmap for {mapped_key}: {e}")
        traceback.print_exc()
        print()
        
    print(f"Pathfinder heatmap saved to {output_path}")
    
    # get simulation plots contributing to the pathfinder heatmap and generate report.
    try:
        print(f"Generating pathfinder report...")
        perm_dir = pkwargs.get('permutation_dir', None)
        assert perm_dir is not None, "permutation_dir must be specified in pkwargs."
        if not os.path.exists(perm_dir):
            raise FileNotFoundError(f"Permutations directory {perm_dir} does not exist.")
        png_files = []
        for param, data in heat_dict.items():
            # get the simulation plots for this param if they exist.
            levels = len(data['metrics'])
            levels_list = _derive_level_strings(levels)
            m = data['metrics']
            
            # remove origin from m for this to work
            middle_idx = len(m) // 2
            assert len(m) % 2 == 1, f"Metric length for {param} is even ({len(m)}), which cannot not center correctly."
            m = m[:middle_idx] + m[middle_idx+1:]  # remove the origin value
            
            for el, level in zip(m, levels_list):
                if not np.isnan(el):
                    perm_subir = os.path.join(perm_dir, f"{param}{level}")
                    if not os.path.exists(perm_subir):
                        print(f"[WARNING] Permutation subdirectory {perm_subir} does not exist for param {param} at level {level}. Skipping.")
                        continue
                    # use glob to find paths ending in '_2p.png'
                    # #HACK if more than one found, just take the first one
                    sim_plots = glob.glob(os.path.join(perm_subir, '*_2p.png'))
                    if not sim_plots:
                        print(f"[WARNING] No simulation plots found for param {param} at level {level} in {perm_subir}. Skipping.")
                        continue
                    sim_plot = sim_plots[0]
                    png_files.append(sim_plot)
                    print(f"Found simulation plot for param {param} at level {level}: {sim_plot}")
                    # print(f"Found non-NaN value in metrics for param {param}: {el}")
                    # break
            #print(f"Generating simulation plots for param {param} with levels {levels_list}")
        if not png_files:
            print(f"No simulation plots found for pathfinder report. Skipping report generation.")
            raise ValueError("No simulation plots found for pathfinder report. Skipping report generation.")
        
        # add newly generated heatmap to png_files, at the top of the list
        png_files.insert(0, output_path)  # add the pathfinder heatmap itself
        
        # if png_files collected, generate the report
        print(f"Generating pathfinder report with {len(png_files)} simulation plots.")
        output_pdf_path = os.path.join(output_dir, 'pathfinder_report.pdf')
        _generate_pathfinder_report(png_files, output_pdf_path=output_pdf_path)        
        
    except Exception as e:
        print(f"Error generating pathfinder report: {e}")
        traceback.print_exc()
        print("Skipping report generation.")
    
    
    return output_path