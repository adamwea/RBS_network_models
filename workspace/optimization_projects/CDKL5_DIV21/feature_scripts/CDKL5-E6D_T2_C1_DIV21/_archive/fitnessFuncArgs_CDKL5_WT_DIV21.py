# New Fitness Targets

#modeled after the network_metric_targets dictionary in modules/analysis/simulation_fitness_functions/calculate_fitness.py
network_metric_targets = {
    #General Data
    #'source': None, # 'simulated' or 'experimental'
    #'timeVector': None,
    
    # #Simulated Data
    # 'simulated_data': {
    #     'soma_voltage': None,
    #     'E_Gids': None,
    #     'I_Gids': None,
    #     'MeanFireRate_E': None,
    #     'CoVFireRate_E': None,
    #     'MeanFireRate_I': None,
    #     'CoVFireRate_I': None,
    #     'MeanISI_E': None,
    #     'MeanISI_I': None,
    #     'CoV_ISI_E': None,
    #     'CoV_ISI_I': None,
    #     'spiking_data_by_unit': None, 
    # },
    
    #Spiking Data
    'spiking_data': {
        'spike_times': None,
        'spiking_summary_data': {
            #'spike_times': None,
            'MeanFireRate': {
                'target': 2, #TODO: these are placeholder values
                'min': 0.00,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'CoVFireRate': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
                },
            'MeanISI': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
                },
            'CoV_ISI': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
                },         
        },
        #'spiking_times_by_unit': None,
        #'spiking_data_by_unit': None,
    },
    
    #Bursting Data
    'bursting_data': {
        'bursting_summary_data': {
            'baseline': {
                'target': 0, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'MeanWithinBurstISI': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'CovWithinBurstISI': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'MeanOutsideBurstISI': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'CoVOutsideBurstISI': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'MeanNetworkISI': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'CoVNetworkISI': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
             'Number_Bursts': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'mean_IBI': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'cov_IBI': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'mean_Burst_Peak': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'cov_Burst_Peak': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
            'fano_factor': {
                'target': 1000, #TODO: these are placeholder values
                'min': None,
                'max': None,
                'weight': 1, #TODO: update these with Nfactors
            },
        },
        #'bursting_data_by_unit': None,
    }
}
    
    
'''FITNESS FUNCTION ARGUMENTS'''
# fitness function
fitnessFuncArgs = {}
fitnessFuncArgs['targets'] = network_metric_targets
fitnessFuncArgs['features'] = {
    'num_excite': 100,
    'num_inhib': 46,
}
fitnessFuncArgs['maxFitness'] = 1000