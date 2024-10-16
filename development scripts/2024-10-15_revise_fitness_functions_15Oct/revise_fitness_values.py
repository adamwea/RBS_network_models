from simulation_fitness_functions.calculate_fitness import fitnessFunc
from simulation_config_files import fitnessFuncArgs
import os
import sys
sys.path.append('submodules/netpyne')

from netpyne import sim
from copy import deepcopy
import dill
from netpyne import sim

def extract_data_of_interest_from_sim(data_path, temp_dir="_temp-sim-files"):
    
    sim.loadSimCfg(data_path)           # Load the simulation configuration
    sim.loadNet(data_path)              # Load the network
    sim.loadNetParams(data_path)        # Load the network parameters
    sim.loadSimData(data_path)          # Load the simulation data
    
    # Copy data of interest
    simData = sim.allSimData.copy()     # Copy the simulation data
    cellData = sim.net.allCells.copy()   # Copy the network data
    popData = sim.net.allPops.copy()    # Copy the population data

    #Copy relevant params
    simCfg = sim.cfg.__dict__.copy()
    netParams = sim.net.params.__dict__.copy()

    #clear the sim object
    sim.clearAll()

    extracted_data= {
        'simData': simData, # Simulated network activity data
        'cellData': cellData, # Network data: Individual neuron data, including connectivity, synapses, etc.
        'popData': popData, # Network data: Population data
        'simCfg': simCfg, # Simulation configuration
        'netParams': netParams, # Network parameters
    }

    return extracted_data

# Define the target directory
target_dir = 'simulation_output/240808_Run1_testing_data/gen_0'
target_dir = os.path.abspath(target_dir)

# Define the directory for temp files
temp_dir = "_temp-sim-files"

# Get all files in the target directory ending with _data.json. Walk through the directory tree and get all files.
simulation_files = {}
for root, dirs, files in os.walk(target_dir):
    for file in files:
        if file.endswith('_data.json'):
            data_path = os.path.join(root, file)
            data_path = os.path.abspath(data_path)
            
            #Find matching file with _cfg.json instead of _data.json
            cfg_path = file.replace('_data.json', '_cfg.json')
            cfg_path = os.path.join(root, cfg_path)

            #try to find match file with _Fitness.json instead of _data.json
            try:
                fitness_path = file.replace('_data.json', '_Fitness.json')
                fitness_path = os.path.join(root, fitness_path)
            except:
                try: 
                    fitness_path = file.replace('_data.json', '_fitness.json')
                    fitness_path = os.path.join(root, fitness_path)
                except:
                    fitness_path = None

            #Assert that both files exist
            try:
                assert os.path.exists(cfg_path), f'File {cfg_path} not found.'
                assert os.path.exists(data_path), f'File {data_path} not found.'
            except AssertionError as e:
                print(e)
                continue

            #Add the files to the dictionary
            simulation_files[file] = {
                'data_path': data_path,
                'cfg_path': cfg_path,
                'fitness_path': fitness_path,
            }

# Prepare the fitness function arguments
kwargs = {
    'simData' : None,
    'simLabel': None,
    'data_file_path': None,
    'batch_saveFolder': None,
    'fitness_save_path': None,
    'fitnessFuncArgs': fitnessFuncArgs,
}

# Iterate through the simulation data files
bad_files = []
selection = None
#selection = 'gen_0_cand_0_'
for f in simulation_files.items():
    try:
        #Get paths out
        file_name = f[0]
        data_path = f[1]['data_path']
        cfg_path = f[1]['cfg_path']
        fitness_path = f[1]['fitness_path']

        #Apply selection if specified
        if selection is not None and selection not in data_path:
            print(f'Skipping {f} because it does not match the selection criteria.')
            continue

        # Load the simulation objects
        extracted_data = extract_data_of_interest_from_sim(data_path)

        # init kwargs
        kwargs = {
            'simData': extracted_data['simData'],
            'cellData': extracted_data['cellData'],
            'popData': extracted_data['popData'],
            'simCfg': extracted_data['simCfg'],
            'netParams': extracted_data['netParams'],
            'simLabel': extracted_data['simCfg']['simLabel'],
            'data_file_path': data_path,
            'cfg_file_path': cfg_path,
            'fitness_file_path': fitness_path,
        }
        
        # Calculate the fitness
        average_fitness, avg_scaled_fitness, fitnessVals = fitnessFunc(**kwargs)
        
        print(f'Average Fitness for {f}: {average_fitness}')
        print(f'Average Scaled Fitness for {f}: {avg_scaled_fitness}')
        print(f'Fitness Values for {f}: {fitnessVals}')
    except Exception as e:
        print(f'Error in processing {f}: {e}')
        bad_files.append(f)
        continue

# Save the bad files as .txt
with open('bad_files.txt', 'w') as f:
    for file in bad_files:
        f.write(f'{file}\n')