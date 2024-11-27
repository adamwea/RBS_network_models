import numpy as np
import matplotlib.pyplot as plt

def plot_fitness_trend(data, gen_col='#gen', fitness_col='fitness'):
    """
    Plots a scatter plot of fitness scores by generation with a 2nd-order trendline.
    Highlights minimum and 50th fitness values for each generation.
    Displays mean and standard deviation for each generation.
    
    Parameters:
    - data (pd.DataFrame): The dataset containing generation and fitness data.
    - gen_col (str): The name of the generation column.
    - fitness_col (str): The name of the fitness column.
    """
    # Ensure numeric data types
    data[gen_col] = pd.to_numeric(data[gen_col], errors='coerce')
    data[fitness_col] = pd.to_numeric(data[fitness_col], errors='coerce')
    data = data.dropna(subset=[gen_col, fitness_col])

    # Group by generation and calculate statistics
    grouped_stats = data.groupby(gen_col)[fitness_col].agg(['mean', 'std']).reset_index()
    grouped_stats[gen_col] = grouped_stats[gen_col].astype(int)  # Ensure integer generations

    # Highlight points for min and 50th fitness in each generation
    highlighted_points = []
    generations = data[gen_col].unique()
    for gen in generations:
        gen_data = data[data[gen_col] == gen].sort_values(by=fitness_col, ascending=True)
        min_point = gen_data.iloc[0]  # Lowest fitness
        if len(gen_data) >= 50:
            mid_point = gen_data.iloc[49]  # 50th fitness
        else:
            mid_point = gen_data.iloc[-1]  # Last available if fewer than 50
        highlighted_points.append((min_point, mid_point))

    # Calculate 2nd-order trendline
    x = data[gen_col]
    y = data[fitness_col]
    coefficients_2nd_order = np.polyfit(x, y, deg=2)  # 2nd-order regression
    trendline_2nd_order = np.poly1d(coefficients_2nd_order)

    # Create the scatter plot
    plt.figure(figsize=(12, 6))
    for gen in grouped_stats[gen_col]:
        gen_data = data[data[gen_col] == gen]
        plt.scatter(gen_data[gen_col], gen_data[fitness_col], label=f'Gen {gen}', alpha=0.6, s=10)

    # Highlight points for min and 50th fitness in each generation
    for min_point, mid_point in highlighted_points:
        plt.scatter(min_point[gen_col], min_point[fitness_col], color='green', label='Min Fitness' if 'Min Fitness' not in plt.gca().get_legend_handles_labels()[1] else "", edgecolors='black', s=50)
        plt.scatter(mid_point[gen_col], mid_point[fitness_col], color='blue', label='50th Fitness' if '50th Fitness' not in plt.gca().get_legend_handles_labels()[1] else "", edgecolors='black', s=50)

    # Plot the 2nd-order trendline
    x_range = np.linspace(grouped_stats[gen_col].min(), grouped_stats[gen_col].max(), 500)
    plt.plot(x_range, trendline_2nd_order(x_range), color='orange', label='2nd Order Trendline', linewidth=2)

    # Plot mean and standard deviation for each generation
    plt.errorbar(
        grouped_stats[gen_col], grouped_stats['mean'],
        yerr=grouped_stats['std'], fmt='o', color='purple', label='Mean ± Std Dev', capsize=5
    )

    # Customize plot
    plt.xticks(grouped_stats[gen_col])  # Ensure x-axis only shows whole-number generations
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Scatter Plot with Mean, Std Dev, and 2nd Order Trendline')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Return the trendline equation
    return f"Trendline Function: f(x) = {coefficients_2nd_order[0]:.3f}x^2 + {coefficients_2nd_order[1]:.3f}x + {coefficients_2nd_order[2]:.3f}"

def get_detailed_simulation_data(data_path):
    from netpyne import sim
    """
    Processes simulation data to extract relevant information for analysis.
    
    Parameters:
    - data_path (str): The path to the simulation data file.
    
    Returns:
    - extracted_data (dict): A dictionary containing the extracted data.
    """
    # Load the simulation data
    sim.loadSimCfg(data_path)
    sim.loadNet(data_path)
    sim.loadNetParams(data_path)
    sim.loadSimData(data_path)
    
    simData = sim.allSimData.copy()
    cellData = sim.net.allCells.copy()
    popData = sim.net.allPops.copy()
    simCfg = sim.cfg.__dict__.copy()
    netParams = sim.net.params.__dict__.copy()
    
    sim.clearAll()
    
    extracted_data = {
        'simData': simData,
        'cellData': cellData,
        'popData': popData,
        'simCfg': simCfg,
        'netParams': netParams,
    }
    
    return extracted_data

import matplotlib.pyplot as plt

def plot_raster_plot(spiking_data_by_unit, E_gids, I_gids):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot excitatory neurons in yellow
    for gid in E_gids:
        if gid in spiking_data_by_unit:
            spike_times = spiking_data_by_unit[gid]['spike_times']
            #make sure spike times are expressed as list even if only one spike
            spike_times = [spike_times] if isinstance(spike_times, (int, float)) else spike_times
            #if isinstance(spike_times, list) and all(isinstance(t, (int, float)) for t in spike_times):
            ax.plot(spike_times, [gid] * len(spike_times), 'y.', markersize=2)

    # Plot inhibitory neurons in blue
    for gid in I_gids:
        if gid in spiking_data_by_unit:
            spike_times = spiking_data_by_unit[gid]['spike_times']
            #make sure spike times are expressed as list even if only one spike
            spike_times = [spike_times] if isinstance(spike_times, (int, float)) else spike_times
            #if isinstance(spike_times, list) and all(isinstance(t, (int, float)) for t in spike_times):
            ax.plot(spike_times, [gid] * len(spike_times), 'b.', markersize=2)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title('Raster Plot')
    plt.tight_layout()
    return fig, ax

def analyze_simulation_data(data_path):
    import re
    from modules.analysis.analyze_network_activity import get_simulated_network_activity_metrics
    #get extracted data
    extracted_data = get_detailed_simulation_data(data_path)
    raster_plot_path = re.sub(r'_data.json', '_raster_plot.pdf', data_path)
    bursting_plot_path = re.sub(r'_data.json', '_bursting_plot.pdf', data_path)
    if os.path.exists(raster_plot_path) and os.path.exists(bursting_plot_path):
        print(f"Raster plot and bursting plot already exist for {data_path}. Skipping analysis.")
        return

    #get network activity metrics
    kwargs=extracted_data
    network_data = get_simulated_network_activity_metrics(**kwargs)
    
    #use spiking data to generate raster plot
    simulated_data = network_data['simulated_data']
    spiking_data_by_unit = simulated_data['spiking_data_by_unit']
    E_gids = simulated_data['E_Gids']
    I_gids = simulated_data['I_Gids']
    if not os.path.exists(raster_plot_path):
        fig, ax = plot_raster_plot(spiking_data_by_unit, E_gids, I_gids)
        fig.savefig(raster_plot_path)
        print(f"Raster plot saved to {raster_plot_path}")
    
    #use bursting data to generate bursting plot    
    #get ax and fig for network bursting plot out of network_data
    if not os.path.exists(bursting_plot_path):
        fig = network_data['bursting_data']['bursting_summary_data']['fig']
        ax = network_data['bursting_data']['bursting_summary_data']['ax']
        # replace _data.json with _bursting_plot.pdf
        fig.savefig(bursting_plot_path)
        print(f"Bursting plot saved to {bursting_plot_path}")   

''' 
reintegrating code below into analysis functions above 
'''

'''Setup Python environment for running the script'''
from pprint import pprint
import setup_environment
setup_environment.set_pythonpath()

'''Import necessary modules'''
from workspace.RBS_network_simulations.workspace.optimization_projects.CDKL5_DIV21.calculate_fitness import fitnessFunc
from netpyne import sim
import dill
import os

def extract_data_of_interest_from_sim(data_path, temp_dir="_temp-sim-files", load_extract=True):
    #init and decide to load or not
    sim.loadSimCfg(data_path)
    simLabel = sim.cfg.simLabel
    temp_sim_dir = os.path.join(temp_dir, simLabel)
    pkl_file = os.path.join(temp_sim_dir, f'{simLabel}_extracted_data.pkl')
    
    #load extracted data if it exists
    if os.path.exists(pkl_file) and load_extract:
        try:
            with open(pkl_file, 'rb') as f:
                extracted_data = dill.load(f)
            return extracted_data
        except Exception as e:
            print(f'Error loading extracted data: {e}')
            pass    
    
    #if it doesn't exist, load the sim data and extract the data of interest
    sim.loadNet(data_path)
    sim.loadNetParams(data_path)
    sim.loadSimData(data_path)
    
    simData = sim.allSimData.copy()
    cellData = sim.net.allCells.copy()
    popData = sim.net.allPops.copy()
    simCfg = sim.cfg.__dict__.copy()
    netParams = sim.net.params.__dict__.copy()
    
    sim.clearAll()
    
    extracted_data = {
        'simData': simData,
        'cellData': cellData,
        'popData': popData,
        'simCfg': simCfg,
        'netParams': netParams,
    }
    
    if not os.path.exists(temp_sim_dir):
        os.makedirs(temp_sim_dir)
    # save extracted data
    with open(pkl_file, 'wb') as f:
        dill.dump(extracted_data, f)
    
    return extracted_data

def find_simulation_files(target_dir):
    simulation_files = {}
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('_data.json'):
                data_path = os.path.join(root, file)
                cfg_path = os.path.join(root, file.replace('_data.json', '_cfg.json'))
                fitness_path = os.path.join(root, file.replace('_data.json', '_Fitness.json'))
                if not os.path.exists(fitness_path):
                    fitness_path = os.path.join(root, file.replace('_data.json', '_fitness.json'))
                
                if os.path.exists(cfg_path) and os.path.exists(data_path):
                    simulation_files[file] = {
                        'data_path': os.path.abspath(data_path),
                        'cfg_path': os.path.abspath(cfg_path),
                        'fitness_path': os.path.abspath(fitness_path) if os.path.exists(fitness_path) else None,
                    }
                else:
                    print(f'File {cfg_path} or {data_path} not found.')
    return simulation_files

def process_simulation_file(file_name, file_paths, selection=None):
    import re
    if selection and selection not in file_paths['data_path']:
        print(f'Skipping {file_name} because it does not match the selection criteria.')
        return None, None, None
    
    #generate output path. Replace 'Fitness' with 'fitness' in the file name. Also modify path to save in the same location as this script
    fitness_save_name = os.path.basename(file_paths['fitness_path'])
    fitness_save_name = re.sub(r'Fitness', 'fitness', fitness_save_name)
    fitness_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fitness_save_name)
    
    extracted_data = extract_data_of_interest_from_sim(file_paths['data_path'])
    kwargs = {
        'simData': extracted_data['simData'],
        'cellData': extracted_data['cellData'],
        'popData': extracted_data['popData'],
        'simCfg': extracted_data['simCfg'],
        'netParams': extracted_data['netParams'],
        'simLabel': extracted_data['simCfg']['simLabel'],
        'data_file_path': file_paths['data_path'],
        'cfg_file_path': file_paths['cfg_path'],
        'fitness_file_path': file_paths['fitness_path'], #input
        #replace 'Fitness' with 'fitness' in the file name
        'fitness_save_path': fitness_save_path, #output
        #'fitnessFuncArgs': fitnessFuncArgs_dep,
    }
    
    return fitnessFunc(**kwargs)

# def main():
#     # Set the target directory to the directory containing the simulation output
#     target_dir = os.path.abspath('simulation_output/240808_Run1_testing_data/gen_0')
#     simulation_files = find_simulation_files(target_dir)
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     bad_files_path = os.path.join(script_dir, 'bad_files.txt')
    
#     bad_files = []
#     selection = None  # Set to a specific selection if needed
    
#     for file_name, file_paths in simulation_files.items():
#         try:
#             average_fitness, avg_scaled_fitness, fitnessVals = process_simulation_file(file_name, file_paths, selection)
#             if average_fitness is not None:
#                 print(f'Average Fitness for {file_name}: {average_fitness}')
#                 print(f'Average Scaled Fitness for {file_name}: {avg_scaled_fitness}')
#                 print(f'Fitness Values for {file_name}: {fitnessVals}')
#         except Exception as e:
#             print(f'Error in processing {file_name}: {e}')
#             bad_files.append(file_name)
    

#     with open(bad_files_path, 'w') as f:
#         for file in bad_files:
#             f.write(f'{file}\n')

# if __name__ == "__main__":
#     main()