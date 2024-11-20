import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')  # or 'Agg' if you don't need interactive plots

# Data extraction functions

def extract_pops_data(file_path, simulation_run):
    """Extract pops data from batch_config.json."""
    with open(file_path, 'r') as f:
        try:
            # Load the JSON data
            batch_data = json.load(f)

            # Extract pops data
            pops = batch_data.get('evolCfg', {}).get('fitnessFuncArgs', {}).get('pops', {})
            # Prepare entry for pops data
            pops_entry = {
                'Simulation_Run': simulation_run,
                # 'Gen': gen,
                # 'Cand': cand,
                'num_bursts_target': pops.get('num_bursts_target', None),
                'E_rate_target': pops.get('E_rate_target', {}),
                'I_rate_target': pops.get('I_rate_target', {}),
                'E_ISI_target': pops.get('E_ISI_target', {}),
                'I_ISI_target': pops.get('I_ISI_target', {}),
                'baseline_target': pops.get('baseline_target', {}),
                'big-small_cutoff': pops.get('big-small_cutoff', None),
                'big_burst_target': pops.get('big_burst_target', {}),
                'small_burst_target': pops.get('small_burst_target', {}),
                'bimodal_burst_target': pops.get('bimodal_burst_target', {}),
                'IBI_target': pops.get('IBI_target', {}),
                'burst_frequency_target': pops.get('burst_frequency_target', {}),
                'threshold_target': pops.get('threshold_target', {}),
                'sustained_activity_target': pops.get('sustained_activity_target', {}),
                'slope_target': pops.get('slope_target', {})
            }
            return pops_entry

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return None

def extract_fitness_data(file_path, simulation_run, gen, cand):
    """Extract fitness data from a fitness JSON file."""
    with open(file_path, 'r') as f:
        try:
            # Load the JSON data
            json_data = json.load(f)

            # Extract relevant fitness data
            fitness_entries = []
            for key, value in json_data.items():
                if isinstance(value, dict):
                    # Flatten the data structure
                    entry = {
                        'Simulation_Run': simulation_run,
                        'Gen': gen,
                        'Cand': cand,
                        'Type': key,
                        'Value': value.get('Value'),
                        'Fit': value.get('Fit'),
                        'deprioritized': value.get('deprioritized', False)
                    }
                    # Add features if they exist
                    if 'Features' in value:
                        entry.update(value['Features'])
                    fitness_entries.append(entry)
            return fitness_entries

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return []

def post_process_dataframes(fitness_df, pops_df):
    """Post-process dataframes to exclude simulation runs with empty target values in pops."""
    # Identify simulation runs with empty target values in pops
    empty_target_runs = pops_df[pops_df.isnull().any(axis=1)]['Simulation_Run'].unique()

    # Exclude these simulation runs from both dataframes
    fitness_df = fitness_df[~fitness_df['Simulation_Run'].isin(empty_target_runs)]
    pops_df = pops_df[~pops_df['Simulation_Run'].isin(empty_target_runs)]

    return fitness_df, pops_df

def traverse_dir_tree_and_make_dfs(base_dir):
    # Traverse the directory tree
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                # Extract simulation run, gen, and cand from the file path
                parts = file_path.split(os.sep)


                # Check if it's a fitness JSON file
                if 'Fitness' in file:
                    simulation_run = parts[-3]  # Assuming the structure is consistent
                    gen = parts[-2]
                    cand = file.split('_')[3]  # Extracting 'cand' from the filename
                    fitness_entries = extract_fitness_data(file_path, simulation_run, gen, cand)
                    fitness_data_list.extend(fitness_entries)

                # Check if it's a batch_config JSON file
                elif 'batch_config.json' in file:
                    simulation_run = parts[-2]  # Assuming the structure is consistent
                    pops_entry = extract_pops_data(file_path, simulation_run)
                    if pops_entry:
                        pops_data_list.append(pops_entry)

    # Create DataFrames from the lists of data
    fitness_df = pd.DataFrame(fitness_data_list)
    pops_df = pd.DataFrame(pops_data_list)

    return fitness_df, pops_df

# Histogram plotting functions

def plot_burst_frequency_fitness(fitness_df, pops_df, exclude_no_value=True):
    """Plot Value and Fit for burst_frequency_fitness for each simulation run in separate subplots."""
    # Filter the DataFrame for burst_frequency_fitness
    burst_freq_df = fitness_df[fitness_df['Type'] == 'burst_frequency_fitness']

    # Optionally exclude candidates with no burst frequency value
    if exclude_no_value:
        burst_freq_df = burst_freq_df[burst_freq_df['Value'].notnull()]
        # Data frame size before and after filtering
        print(f"Data frame size before filtering: {len(fitness_df)}")
        print(f"Data frame size after filtering: {len(burst_freq_df)}")

    # Get unique simulation runs
    simulation_runs = burst_freq_df['Simulation_Run'].unique()

    # Create a plot for each simulation run and generation
    for run in simulation_runs:
        # Filter for the current simulation run
        run_data = burst_freq_df[burst_freq_df['Simulation_Run'] == run]

        # Get unique generations
        generations = run_data['Gen'].sort_values().unique()

        # Get the burst frequency target value from pops_df
        target_value = pops_df[pops_df['Simulation_Run'] == run]['burst_frequency_target'].values[0]
        target_value = target_value['target']

        # Plot each generation in a separate figure
        for gen in generations:
            # Filter for the current generation
            gen_data = run_data[run_data['Gen'] == gen]

            # Sort by candidate number
            gen_data = gen_data.sort_values(by='Cand')

            # Create a new figure
            fig, axes = plt.subplots(2, 1, figsize=(24, 12), sharex=True)  # Adjusted size

            # Plot Value in the top subplot
            axes[0].bar(gen_data['Cand'].astype(str), gen_data['Value'], color='skyblue', alpha=0.7)
            axes[0].axhline(y=target_value, color='red', linestyle='--', label='target')
            axes[0].set_title(f'Value (Hz) {run} - Generation {gen}')
            axes[0].set_ylabel('Value')
            axes[0].grid(axis='y')
            axes[0].legend()

            # Plot Fit in the bottom subplot as the negative space between the fitness value and 1000
            inverted_fit = 1000 - gen_data['Fit']
            axes[1].bar(gen_data['Cand'].astype(str), inverted_fit, color='orange', alpha=0.7)
            axes[1].set_title(f'Inverted Fit {run} - Generation {gen}')
            axes[1].set_ylabel('Inverted Fit (1000 - Fit)')
            axes[1].grid(axis='y')

            # Create a secondary y-axis for the non-inverted Fit values
            ax2 = axes[1].twinx()
            ax2.set_ylabel('Fit')
            ax2.set_ylim(0, 1000)
            ax2.set_yticks([1000 - tick for tick in axes[1].get_yticks()])
            ax2.set_yticklabels([str(1000 - tick) for tick in axes[1].get_yticks()])

            axes[0].tick_params(axis='x', labelsize=8)  # Make candidate labels smaller
            axes[1].tick_params(axis='x', labelsize=8)  # Make candidate labels smaller

            # Adjust layout
            plt.tight_layout()

            # Save the plot to a file
            plot_file = f"{run}_generation_{gen}_burst_frequency_fitness.png"
            plt.savefig(plot_file)
            print(f"Plot saved to {plot_file}")

            # Show the plot in VS Code
            # plt.show()

def plot_candidates_per_generation(fitness_df, fitness_type=None, exclude_no_value=True):
    """Plot the number of candidates per generation for each simulation run, optionally filtered by fitness type."""
    # Optionally filter by fitness type
    if fitness_type:
        fitness_df = fitness_df[fitness_df['Type'] == fitness_type]

    # Optionally exclude candidates with no value
    if exclude_no_value:
        fitness_df = fitness_df[fitness_df['Value'].notnull()]
        # Data frame size before and after filtering
        print(f"Data frame size before filtering: {len(fitness_df)}")
        print(f"Data frame size after filtering: {len(fitness_df)}")

    # Group by Simulation_Run and Gen, and count the number of candidates
    candidates_per_gen = fitness_df.groupby(['Simulation_Run', 'Gen']).size().reset_index(name='Num_Candidates')

    # Get unique simulation runs
    simulation_runs = candidates_per_gen['Simulation_Run'].unique()

    # Create a plot for each simulation run
    for run in simulation_runs:
        # Filter for the current simulation run
        run_data = candidates_per_gen[candidates_per_gen['Simulation_Run'] == run]

        # Create a new figure
        plt.figure(figsize=(12, 6))

        # Plot number of candidates per generation
        plt.bar(run_data['Gen'].astype(str), run_data['Num_Candidates'], color='green', alpha=0.7)
        title = f'Survivors per Generation - {run}'
        if fitness_type:
            title += f' ({fitness_type})'
        plt.title(title)
        plt.xlabel('Generation')
        plt.ylabel('Number of Candidates')
        plt.grid(axis='y')

        # Adjust layout
        plt.tight_layout()

        # Save the plot to a file
        plot_file = f"{run}_candidates_per_generation"
        if fitness_type:
            plot_file += f"_{fitness_type}"
        plot_file += ".png"
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")

        # Show the plot in VS Code
        # plt.show()
        
def plot_fitness_histogram(fitness_df, fitness_type, pops_df, target_type, units=None, bin_width=None, mode='value', base_dir=None, num_std=4):
    """Plot histogram for a specific fitness type with modular bin width and target features.
    
    Args:
        fitness_df (pd.DataFrame): DataFrame containing fitness data.
        fitness_type (str): The type of fitness to plot.
        pops_df (pd.DataFrame): DataFrame containing pops data.
        target_type (str): The type of target to plot.
        bin_width (float, optional): Width of the bins in the histogram. Defaults to None.
        mode (str, optional): Mode of the histogram ('value' or 'rms_diff'). Defaults to 'value'.
    """
    # Filter the DataFrame for the specified fitness type
    fitness_type_df = fitness_df[fitness_df['Type'] == fitness_type]

    # Exclude candidates with no value
    fitness_type_df = fitness_type_df[fitness_type_df['Value'].notnull()]
    
    # Save the data to a CSV file
    if base_dir is not None: 
        csv_dir = os.path.dirname(base_dir)
        csv_dir = os.path.join(csv_dir, 'csv_data')
        os.makedirs(csv_dir, exist_ok=True)
        fitness_type_df.to_csv(os.path.join(csv_dir, f'{fitness_type}_data.csv'), index=False)
    else: 
        fitness_type_df.to_csv(f'{fitness_type}_data.csv', index=False)
    
    # If bin width is not specified, calculate it based on the data
    if bin_width is None:
        data_range = fitness_type_df['Value'].max() - fitness_type_df['Value'].min()
        bin_width = data_range / 100  # 1% increments of the range

    # Calculate the number of bins
    num_bins = int(data_range / bin_width) + 1

    # Calculate the mean and standard deviation of the fitness_type_df data
    sample_mean = fitness_type_df['Value'].mean()
    sample_std = fitness_type_df['Value'].std()

    # Filter the plot data to be within num_std standard deviations of the target mean
    lower_bound = sample_mean - num_std * sample_std
    upper_bound = sample_mean + num_std * sample_std
    filtered_df = fitness_type_df[(fitness_type_df['Value'] >= lower_bound) & (fitness_type_df['Value'] <= upper_bound)]

    # Calculate the mean value of the filtered data
    filtered_mean = filtered_df['Value'].mean()

    #
    if units is None:
        units = ''
    else:
        units = f'[{units}]'
    
    # Prepare the data for plotting
    if mode == 'rms_diff':
        fitness_type_df['RMS_Diff'] = fitness_type_df.apply(lambda row: abs(row['Value'] - pops_df[pops_df['Simulation_Run'] == row['Simulation_Run']][target_type].values[0]['target']), axis=1)
        plot_data = fitness_type_df['RMS_Diff']
        plot_title = f'Abs RMS Diff for {fitness_type} ({num_std} std, mode: {mode})'
        x_label = f'Abs RMS Diff {units}'
        
        # Calculate RMS difference for the mean value
        mean_rms_diff = abs(filtered_mean - pops_df[pops_df['Simulation_Run'] == fitness_type_df['Simulation_Run'].iloc[0]][target_type].values[0]['target'])
    else:
        plot_data = fitness_type_df['Value']
        plot_title = f'Histogram of {fitness_type} Values'
        x_label = f'Value {units}'
        mean_rms_diff = filtered_mean

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(plot_data, bins=num_bins, color='blue', alpha=0.7)
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel('Candidate Count')
    plt.grid(axis='y')

    # Plot target features as vertical lines
    real_target_value = pops_df[pops_df['Simulation_Run'] == fitness_type_df['Simulation_Run'].iloc[0]][target_type].values[0]['target']
    if mode == 'rms_diff':
        plt.axvline(x=0, color='red', linestyle='--', label=f'target ({real_target_value} {units})')
    else:
        for run in fitness_type_df['Simulation_Run'].unique():
            target_info = pops_df[pops_df['Simulation_Run'] == run][target_type].values[0]
            if isinstance(target_info, dict):
                min_value = target_info.get('min', None)
                max_value = target_info.get('max', None)
                target_value = target_info.get('target', None)

                if min_value is not None:
                    plt.axvline(x=min_value, color='black', linestyle='--', label='min' if run == fitness_type_df['Simulation_Run'].unique()[0] else "")
                if max_value is not None:
                    plt.axvline(x=max_value, color='black', linestyle='--', label='max' if run == fitness_type_df['Simulation_Run'].unique()[0] else "")
                if target_value is not None:
                    plt.axvline(x=target_value, color='red', linestyle='--', label='target' if run == fitness_type_df['Simulation_Run'].unique()[0] else "")

    # Plot the mean value as a green line
    plt.axvline(x=mean_rms_diff, color='green', linestyle='--', label=f'mean ({filtered_mean:.2f} {units})')

    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a file
    if base_dir is not None: 
        plot_dir = os.path.dirname(base_dir)
        plot_dir = os.path.join(plot_dir, f'histograms_{num_std}_std')
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = os.path.join(plot_dir, f"{fitness_type}_histogram_{mode}.png")
    else: 
        plot_file = f"{fitness_type}_histogram_{mode}.png"
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

    # Show the plot in VS Code
    # plt.show()
    
def plot_histogram_with_error_handling(fitness_type, target, units, fitness_df, pops_df, base_dir, num_std=4, mode='rms_diff'):
    try:
        plot_fitness_histogram(fitness_df, fitness_type, pops_df, target, units=units, mode=mode, base_dir=base_dir, num_std=num_std)
    except Exception as e:
        print(f"Error plotting {fitness_type}: {e}")
        
# import matplotlib.pyplot as plt
# import pandas as pd

import pandas as pd

import os
import pandas as pd

def save_filtered_means_to_csv(filtered_means, base_dir):
    # Ensure the base directory is created
    base_dir = os.path.dirname(base_dir)
    csv_dir = os.path.join(base_dir, 'csv_data')
    os.makedirs(csv_dir, exist_ok=True)
    
    # File paths for partial and final CSVs
    partial_csv_path = os.path.join(csv_dir, 'partial_n_factors.csv')
    csv_path = os.path.join(csv_dir, 'n_factors.csv')
    
    # Convert the filtered means list to a DataFrame
    filtered_means_df = pd.DataFrame(filtered_means, columns=['Generation', 'Fit_Type', 'Filtered_Mean'])
    
    # Save the partial filtered means DataFrame to a CSV
    filtered_means_df.to_csv(partial_csv_path, index=False)
    
    # Group by 'Fit_Type' and calculate the mean of 'Filtered_Mean'
    filtered_means_summary = filtered_means_df.groupby('Fit_Type')['Filtered_Mean'].mean().reset_index()
    
    # Save the collapsed data to a final CSV file
    filtered_means_summary.to_csv(csv_path, index=False)

    
def extract_rms_diff_values(fitness_df, pops_df, fitness_types, target_types, num_std=2, base_dir=None):
    # Initialize a dictionary to hold the rms diff values for each fitness type
    rms_diff_values = {fit_type: [] for fit_type in fitness_types}
    
    # This list will hold tuples of (Generation, Fit_Type, Filtered_Mean) for the CSV
    filtered_means = []
    
    # Get the list of unique generations
    generations = sorted(fitness_df['Gen'].unique())
    
    # Iterate over each generation
    for generation in generations:
        # Filter data for the current generation
        gen_df = fitness_df[fitness_df['Gen'] == generation]
        
        # Iterate over fitness types and corresponding target types
        for fit_type, target_type in zip(fitness_types, target_types):
            # Filter data for the current fitness type
            gen_fit_df = gen_df[gen_df['Type'] == fit_type]
            
            if not gen_fit_df.empty:
                # Compute RMS_Diff values for each row
                gen_fit_df['RMS_Diff'] = gen_fit_df.apply(
                    lambda row: abs(row['Value'] - pops_df[pops_df['Simulation_Run'] == row['Simulation_Run']][target_type].values[0]['target']),
                    axis=1
                )
                
                # Calculate the mean and standard deviation of the RMS differences
                sample_mean = gen_fit_df['RMS_Diff'].mean()
                sample_std = gen_fit_df['RMS_Diff'].std()

                if sample_std == 0:
                    # If standard deviation is 0, skip this iteration to avoid division by zero
                    continue

                # Filter the data to be within num_std standard deviations of the mean
                lower_bound = sample_mean - num_std * sample_std
                upper_bound = sample_mean + num_std * sample_std
                filtered_df = gen_fit_df[(gen_fit_df['RMS_Diff'] >= lower_bound) & (gen_fit_df['RMS_Diff'] <= upper_bound)]

                # Only proceed if there is any data left after filtering
                if not filtered_df.empty:
                    # Calculate the mean of the filtered data (before RMS normalization)
                    filtered_mean = filtered_df['RMS_Diff'].mean()

                    # Append the (generation, fit_type, filtered_mean) tuple to the filtered_means list
                    filtered_means.append((generation, fit_type, filtered_mean))

                    # Normalize the filtered mean RMS difference by the unfiltered sample mean
                    normalized_rms_diff = filtered_mean / sample_mean if sample_mean != 0 else 0
                    
                    # Store the result in the rms_diff_values dictionary
                    rms_diff_values[fit_type].append((generation, normalized_rms_diff))

    # Call the function to save the filtered means to CSV
    if base_dir is not None:
        save_filtered_means_to_csv(filtered_means, base_dir)

    return rms_diff_values


def plot_rms_diff_over_generations(rms_diff_values, base_dir):
    plt.figure(figsize=(12, 6))
    for fit_type, values in rms_diff_values.items():
        generations, mean_rms_diffs = zip(*values)
        plt.plot(generations, mean_rms_diffs, label=fit_type)

    plt.title('Normalized Mean RMS Difference Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Normalized Mean RMS Difference')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a file
    plot_dir = os.path.dirname(base_dir)
    plot_dir = os.path.join(plot_dir, 'generation_plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_file = os.path.join(plot_dir, "normalized_mean_rms_diff_over_generations.png")
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

# Example usage
fitness_types = [
    'burst_frequency_fitness', 'thresh_fit', 
    #'slope_fit', 
    'baseline_fit',
    'big_burst_fit', 'small_burst_fit', 'bimodal_burst_fit', 'IBI_fitness',
    'E_rate_fit', 'I_rate_fit', 'E_ISI_fit', 'I_ISI_fit'
]
target_types = [
    'burst_frequency_target', 'threshold_target', 
    #'slope_target', 
    'baseline_target',
    'big_burst_target', 'small_burst_target', 'bimodal_burst_target', 'IBI_target',
    'E_rate_target', 'I_rate_target', 'E_ISI_target', 'I_ISI_target'
]


''' Main script '''
# Define the directory containing the JSON files
#set path of this script as base_dir
base_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(base_dir, 'old_data_to_extract_ftiness_data_from')
assert os.path.isdir(base_dir), f"Directory not found: {base_dir}"

# Initialize lists to hold the data
fitness_data_list = []
pops_data_list = []

# Traverse the directory tree, extract data, and create DataFrames
fitness_df, pops_df = traverse_dir_tree_and_make_dfs(base_dir)

# Post-process the dataframes
fitness_df, pops_df = post_process_dataframes(fitness_df, pops_df)

# Optionally, save the DataFrames to CSV files
fitness_output_file = os.path.join(base_dir, 'fitness_data.csv')
pops_output_file = os.path.join(base_dir, 'pops_data.csv')
fitness_df.to_csv(fitness_output_file, index=False)
pops_df.to_csv(pops_output_file, index=False)

'''Old Code, it still works, just not what Roy was looking for'''

# # Plot the histogram for burst_frequency_fitness values
# plot_burst_frequency_fitness(fitness_df, pops_df, exclude_no_value=True)
# plot_candidates_per_generation(fitness_df, fitness_type='burst_frequency_fitness', exclude_no_value=True)

# print(f"Fitness data extracted and saved to {fitness_output_file}")
# print(f"Pops data extracted and saved to {pops_output_file}")

'''New Code'''

# Define the fitness types, corresponding targets, and units
fitness_targets = [
    ('burst_frequency_fitness', 'burst_frequency_target', 'Hz (bursts/s)'),
    ('thresh_fit', 'threshold_target', 'spike count'),
    # ('slope_fit', 'slope_target', 'spikes/s'),
    # ('sustained_activity_fitness', 'sustained_activity_target', 'units'), # Uncomment if needed
    ('baseline_fit', 'baseline_target', 'spike count'),
    ('big_burst_fit', 'big_burst_target', 'Hz (bursts/s)'),
    ('small_burst_fit', 'small_burst_target', 'Hz (bursts/s)'),
    ('bimodal_burst_fit', 'bimodal_burst_target', 'ratio (big:small)'),
    ('IBI_fitness', 'IBI_target', 's'),
    ('E_rate_fit', 'E_rate_target', 'Hz (spikes/s)'),
    ('I_rate_fit', 'I_rate_target', 'Hz (spikes/s)'),
    ('E_ISI_fit', 'E_ISI_target', 's'),
    ('I_ISI_fit', 'I_ISI_target', 's'),
]

# Loop through each fitness type and target, and plot the histogram
for num_std in [2, 4]:
    for fitness_type, target, units in fitness_targets:
        plot_histogram_with_error_handling(fitness_type, target, units, fitness_df, pops_df, base_dir, num_std=num_std, mode='rms_diff')
        
rms_diff_values = extract_rms_diff_values(fitness_df, pops_df, fitness_types, target_types, num_std=2, base_dir=base_dir)
plot_rms_diff_over_generations(rms_diff_values, base_dir)


