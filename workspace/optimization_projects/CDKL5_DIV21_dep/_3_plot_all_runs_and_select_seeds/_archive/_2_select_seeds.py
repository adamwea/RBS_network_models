import os
import sys
from pprint import pprint


'''functions'''
def main():
    #create a csv with a list of all simulations
    from _1_analyze_and_plot_network_summaries_v3 import collect_fitness_data
    # Collect fitness data
    fitness_data = collect_fitness_data(SIMULATION_RUN_PATHS, get_extra_data=True)
    print(f"Collected fitness data for {len(fitness_data)} simulations.")
    
    #fitness_data should be a dict with paths as keys and a dict of fitness data as values
    #excluding content expressed as a list of values
    #prepare a pandas dataframe, and save it as a csv
    import pandas as pd
    fitness_df = pd.DataFrame.from_dict(fitness_data, orient='index')
    
    #save the dataframe as npy
    import numpy as np
    np.save(os.path.join(SEED_SELECTION_OUTPUT_PATH, 'fitness_data.npy'), fitness_df.to_numpy())    
    fitness_df.to_pickle(os.path.join(SEED_SELECTION_OUTPUT_PATH, 'fitness_data.pkl'))
    
    #extract fitness score 
    
    #flatten the dataframe. deal with dicts within dicts recursively
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    fitness_df = pd.DataFrame.from_dict({k: flatten_dict(v) for k, v in fitness_data.items()}, orient='index')
    
    # #exclude keys with data as lists of values
    # # for key in fitness_df.keys():
    # #     if type(fitness_df[key][0]) == list:
    # #         fitness_df = fitness_df.drop(columns=[key])
    # def is_list_of_values(value):
    #     return type(value) == list and all([type(v) in [int, float] for v in value])
    
    # for key in fitness_df.keys():
    #     if is_list_of_values(fitness_df[key][0]):
    #         fitness_df = fitness_df.drop(columns=[key])
    
    #replace the left most column with the candidate name (i.e. gen_i_cand_j)
    fitness_df['candidate'] = fitness_df.index
    fitness_df.index = range(len(fitness_df))
    
    #save the dataframe as csv
    fitness_df.to_csv(os.path.join(SEED_SELECTION_OUTPUT_PATH, 'fitness_data.csv'))

'''notes and main script'''
# ===============================================================================================================================
# NOTE: This script is also meant to act as a very basic analysis log. Look for corresponding notes in aw obsidian notes.
# ===============================================================================================================================

global SIMULATION_RUN_PATHS, REFERENCE_DATA_NPY, SLIDES_OUTPUT_PATH, SEED_SELECTION_OUTPUT_PATH
SIMULATION_RUN_PATHS = []
# ok, this is working now - 2024-12-1

#run: 241126_Run2_improved_netparams
SIMULATION_RUN_PATHS.append("/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams")
REFERENCE_DATA_NPY = ("/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_1_derive_features_from_experimental_data/network_metrics/network_metrics_well000.npy")
#SLIDES_OUTPUT_PATH = "/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_3_analyze_plot_review/network_summary_slides"
SEED_SELECTION_OUTPUT_PATH = ("/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/_seed_selection_reference")

# Entry point check
if __name__ == "__main__":
    main()