'''Setup Python environment for running the script'''
from pprint import pprint
import setup_environment
setup_environment.set_pythonpath()

'''Import necessary modules'''
from workspace.RBS_network_simulations.workspace.optimization_projects.CDKL5_DIV21_dep.calculate_fitness import fitnessFunc
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

def main():
    # Set the target directory to the directory containing the simulation output
    target_dir = os.path.abspath('simulation_output/240808_Run1_testing_data/gen_0')
    simulation_files = find_simulation_files(target_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bad_files_path = os.path.join(script_dir, 'bad_files.txt')
    
    bad_files = []
    selection = None  # Set to a specific selection if needed
    
    for file_name, file_paths in simulation_files.items():
        try:
            average_fitness, avg_scaled_fitness, fitnessVals = process_simulation_file(file_name, file_paths, selection)
            if average_fitness is not None:
                print(f'Average Fitness for {file_name}: {average_fitness}')
                print(f'Average Scaled Fitness for {file_name}: {avg_scaled_fitness}')
                print(f'Fitness Values for {file_name}: {fitnessVals}')
        except Exception as e:
            print(f'Error in processing {file_name}: {e}')
            bad_files.append(file_name)
    

    with open(bad_files_path, 'w') as f:
        for file in bad_files:
            f.write(f'{file}\n')

if __name__ == "__main__":
    main()
