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
        
def plot_fitness_histogram(fitness_df, fitness_type, pops_df, target_type, bin_width=None):
    """Plot histogram for a specific fitness type with modular bin width and target features."""
    # Filter the DataFrame for the specified fitness type
    fitness_type_df = fitness_df[fitness_df['Type'] == fitness_type]

    # Exclude candidates with no value
    fitness_type_df = fitness_type_df[fitness_type_df['Value'].notnull()]
    
    # Save the data to a CSV file
    fitness_type_df.to_csv(f'{fitness_type}_data.csv', index=False)
    
    # If bin width is not specified, calculate it based on the data
    if bin_width is None:
        data_range = fitness_type_df['Value'].max() - fitness_type_df['Value'].min()
        bin_width = data_range / 100  # 1% increments of the range

    # Calculate the number of bins
    num_bins = int(data_range / bin_width) + 1

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(fitness_type_df['Value'], bins=num_bins, color='blue', alpha=0.7)
    plt.title(f'Histogram of {fitness_type} Values')
    plt.xlabel('Value')
    plt.ylabel('Candidate Count')
    plt.grid(axis='y')

    # Plot target features as vertical lines
    for run in fitness_type_df['Simulation_Run'].unique():
        #target_info = pops_df[f'{fitness_type}_target'].values[0]
        #taget info = fitness_type without _fitness and + _target
        #replace _fitness with _target
        #current_fit_type = fitness_type
        #target_info = pops_df[f'{fitness_type.replace("_fitness", "_target")}'].values[0]
        target_info = pops_df[f'{target_type}'].values[0]
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

    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a file
    plot_file = f"{fitness_type}_histogram.png"
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

    # Show the plot in VS Code
    # plt.show()


''' Main script '''
# Define the directory containing the JSON files
base_dir = './development scripts/2024-10-18_histogram_plots/old_data_to_extract_ftiness_data_from'
base_dir = os.path.abspath(base_dir)

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

# plot histogram for all fitness types
try:
    plot_fitness_histogram(fitness_df, 'burst_frequency_fitness', pops_df, 'burst_frequency_target')
except Exception as e:
    print(f"Error plotting burst_frequency_fitness: {e}")

try:
    plot_fitness_histogram(fitness_df, 'thresh_fit', pops_df, 'threshold_target')
except Exception as e:
    print(f"Error plotting threshold_fitness: {e}")

try:
    plot_fitness_histogram(fitness_df, 'slope_fitness', pops_df)
except Exception as e:
    print(f"Error plotting slope_fitness: {e}")

try:
    plot_fitness_histogram(fitness_df, 'sustained_activity_fitness', pops_df)
except Exception as e:
    print(f"Error plotting sustained_activity_fitness: {e}")

try:
    plot_fitness_histogram(fitness_df, 'baseline_fitness', pops_df)
except Exception as e:
    print(f"Error plotting baseline_fitness: {e}")

try:
    plot_fitness_histogram(fitness_df, 'big_burst_fitness', pops_df)
except Exception as e:
    print(f"Error plotting big_burst_fitness: {e}")

try:
    plot_fitness_histogram(fitness_df, 'small_burst_fitness', pops_df)
except Exception as e:
    print(f"Error plotting small_burst_fitness: {e}")

try:
    plot_fitness_histogram(fitness_df, 'bimodal_burst_fitness', pops_df)
except Exception as e:
    print(f"Error plotting bimodal_burst_fitness: {e}")

try:
    plot_fitness_histogram(fitness_df, 'IBI_fitness', pops_df)
except Exception as e:
    print(f"Error plotting IBI_fitness: {e}")

try:
    plot_fitness_histogram(fitness_df, 'E_rate_fitness', pops_df)
except Exception as e:
    print(f"Error plotting E_rate_fitness: {e}")

try:
    plot_fitness_histogram(fitness_df, 'I_rate_fitness', pops_df)
except Exception as e:
    print(f"Error plotting I_rate_fitness: {e}")

try:
    plot_fitness_histogram(fitness_df, 'E_ISI_fitness', pops_df)
except Exception as e:
    print(f"Error plotting E_ISI_fitness: {e}")

try:
    plot_fitness_histogram(fitness_df, 'I_ISI_fitness', pops_df)
except Exception as e:
    print(f"Error plotting I_ISI_fitness: {e}")

try:
    plot_fitness_histogram(fitness_df, 'bimodality_fitness', pops_df)
except Exception as e:
    print(f"Error plotting bimodality_fitness: {e}")
