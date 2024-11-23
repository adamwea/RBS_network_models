# New Fitness Targets

import pickle
import numpy

#load network metric pickle
pickle_path = '/pscratch/sd/a/adammwea/RBS_network_simulations/tunning_scripts/CDKL5-E6D_T2_C1_DIV21/network_metrics_well000.pkl'
with open(pickle_path, 'rb') as f:
    network_metrics = pickle.load(f)
    
def handle_numpy_float64(data):
    try: 
        if isinstance(data, numpy.ndarray):
            data = data.tolist()
    except: 
        pass
    return data

#get min, exclude None, inf, -inf, nan, 0
def get_min(data, key):
    data_min = min(unit[key] for unit in data.values() if unit[key] is not None and unit[key] > 0 and not (unit[key] != unit[key] or unit[key] == float('inf') or unit[key] == float('-inf')))
    #if data is an array with one value, return that value
    # if isinstance(data_min, list) and len(data_min) == 1:
    #     data_min = data_min[0]
    # try: 
    #     len(data_min)
    #     print('data_min is an array')
    # except TypeError: pass
    data_min = handle_numpy_float64(data_min)
    #print('data_min:', data_min)
    return data_min

#get max, exclude None, inf, -inf, nan
def get_max(data, key):
    data_max = max(unit[key] for unit in data.values() if not (unit[key] is None or unit[key] != unit[key] or unit[key] == float('inf') or unit[key] == float('-inf')))
    #if data is an array with one value, return that value
    # if type(data_max) == list:
    #     data_max = data_max[0]
    data_max = handle_numpy_float64(data_max)
    return data_max

#modeled after the network_metric_targets dictionary in modules/analysis/simulation_fitness_functions/calculate_fitness.py
network_metric_targets = {
    #General Data
    'source': network_metrics['source'], # 'simulated' or 'experimental'
    #'timeVector': network_metrics['timeVector'],
    
    # Spiking Data
    'spiking_data': {
        #'spike_times': network_metrics['spiking_data']['spike_times'],
        #'spiking_times_by_unit': network_metrics['spiking_data']['spiking_times_by_unit'],
        #'spiking_data_by_unit': network_metrics['spiking_data']['spiking_data_by_unit'],
        'spiking_summary_data': {
            'MeanFireRate': {
                'target': network_metrics['spiking_data']['spiking_summary_data']['MeanFireRate'],
                'min': get_min(network_metrics['spiking_data']['spiking_data_by_unit'], 'FireRate'),
                'max': get_max(network_metrics['spiking_data']['spiking_data_by_unit'], 'FireRate'),
                'weight': 1, # TODO: update these with Nfactors
            },
            'CoVFireRate': {
                'target': network_metrics['spiking_data']['spiking_summary_data']['CoVFireRate'],
                'min': get_min(network_metrics['spiking_data']['spiking_data_by_unit'], 'fr_CoV'),
                'max': get_max(network_metrics['spiking_data']['spiking_data_by_unit'], 'fr_CoV'),
                'weight': 1, # TODO: update these with Nfactors
            },
            'MeanISI': {
                'target': network_metrics['spiking_data']['spiking_summary_data']['MeanISI'],
                'min': get_min(network_metrics['spiking_data']['spiking_data_by_unit'], 'meanISI'),
                'max': get_max(network_metrics['spiking_data']['spiking_data_by_unit'], 'meanISI'),
                'weight': 1, # TODO: update these with Nfactors
            },
            'CoV_ISI': {
                'target': network_metrics['spiking_data']['spiking_summary_data']['CoV_ISI'],
                'min': get_min(network_metrics['spiking_data']['spiking_data_by_unit'], 'isi_CoV'),
                'max': get_max(network_metrics['spiking_data']['spiking_data_by_unit'], 'isi_CoV'),
                'weight': 1, # TODO: update these with Nfactors
            },
        },
    },
    
    #Bursting Data
    'bursting_data': {
        'bursting_summary_data': {
            'MeanWithinBurstISI': {
                'target': network_metrics['bursting_data']['bursting_summary_data'].get('MeanWithinBurstISI'),
                'min': get_min(network_metrics['bursting_data']['bursting_data_by_unit'], 'mean_isi_within'),
                'max': get_max(network_metrics['bursting_data']['bursting_data_by_unit'], 'mean_isi_within'),
                'weight': 1,
            },
            'CovWithinBurstISI': {
                'target': network_metrics['bursting_data']['bursting_summary_data'].get('CoVWithinBurstISI'),
                'min': get_min(network_metrics['bursting_data']['bursting_data_by_unit'], 'cov_isi_within'),
                'max': get_max(network_metrics['bursting_data']['bursting_data_by_unit'], 'cov_isi_within'),
                'weight': 1,
            },
            'MeanOutsideBurstISI': {
                'target': network_metrics['bursting_data']['bursting_summary_data'].get('MeanOutsideBurstISI'),
                'min': get_min(network_metrics['bursting_data']['bursting_data_by_unit'], 'mean_isi_outside'),
                'max': get_max(network_metrics['bursting_data']['bursting_data_by_unit'], 'mean_isi_outside'),
                'weight': 1,
            },
            'CoVOutsideBurstISI': {
                'target': network_metrics['bursting_data']['bursting_summary_data'].get('CoVOutsideBurstISI'),
                'min': get_min(network_metrics['bursting_data']['bursting_data_by_unit'], 'cov_isi_outside'),
                'max': get_max(network_metrics['bursting_data']['bursting_data_by_unit'], 'cov_isi_outside'),
                'weight': 1,
            },
            'MeanNetworkISI': {
                'target': network_metrics['bursting_data']['bursting_summary_data'].get('MeanNetworkISI', 0),
                'min': get_min(network_metrics['bursting_data']['bursting_data_by_unit'], 'mean_isi_all'),
                'max': get_max(network_metrics['bursting_data']['bursting_data_by_unit'], 'mean_isi_all'),
                'weight': 1,
            },
            'CoVNetworkISI': {
                'target': network_metrics['bursting_data']['bursting_summary_data'].get('CoVNetworkISI', 0),
                'min': get_min(network_metrics['bursting_data']['bursting_data_by_unit'], 'cov_isi_all'),
                'max': get_max(network_metrics['bursting_data']['bursting_data_by_unit'], 'cov_isi_all'),
                'weight': 1,
            },
             'Number_Bursts': {
                'target': network_metrics['bursting_data']['bursting_summary_data'].get('Number_Bursts'),
                'min': 0,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'mean_IBI': {
                'target': network_metrics['bursting_data']['bursting_summary_data'].get('mean_IBI'),
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'cov_IBI': {
                'target': network_metrics['bursting_data']['bursting_summary_data'].get('cov_IBI'),
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'mean_Burst_Peak': {
                'target': network_metrics['bursting_data']['bursting_summary_data'].get('mean_Burst_Peak'),
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'cov_Burst_Peak': {
                'target': network_metrics['bursting_data']['bursting_summary_data'].get('cov_Burst_Peak'),
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'fano_factor': {
                'target': network_metrics['bursting_data']['bursting_summary_data'].get('fano_factor'),
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
        },
        #'bursting_data_by_unit': None,
    }
}

output_dir = '/pscratch/sd/a/adammwea/RBS_network_simulations/tunning_scripts/CDKL5-E6D_T2_C1_DIV21/derived_fitness_args/'
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_path = output_dir + 'fitness_args_' + timestamp + '.py'

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def format_dict(d, indent=0):
    formatted_str = ''
    for key, value in d.items():
        if isinstance(value, dict):
            formatted_str += ' ' * indent + f"'{key}': {{\n" + format_dict(value, indent + 4) + ' ' * indent + '},\n'
        elif isinstance(value, list):
            formatted_str += ' ' * indent + f"'{key}': [\n"
            for item in value:
                formatted_str += ' ' * (indent + 4) + f"{item},\n"
            formatted_str += ' ' * indent + '],\n'
        else:
            formatted_str += ' ' * indent + f"'{key}': {value},\n"
    return formatted_str

# Convert network_metric_targets to a formatted string
formatted_fitness_args = 'fitness_args = {\n' + format_dict(network_metric_targets, 4) + '}'

with open(output_path, 'w') as f:
    f.write(formatted_fitness_args)

print('Updated fitness args saved to:', output_path)

  