fitness_args = {
    'source': 'experimental',
    'spiking_data': {
        'spiking_summary_data': {
            'MeanFireRate': {
                'target': 0.9033360510215639,
                'min': 0.0066657778962804955,
                'max': 42.45100653246234,
                'weight': 1,
            },
            'CoVFireRate': {
                'target': 3.1662656967538165,
                'min': 0.09795245112054385,
                'max': 5.309807350265894,
                'weight': 1,
            },
            'MeanISI': {
                'target': 11.619033431530301,
                'min': 0.023541959798994976,
                'max': 222.0746,
                'weight': 1,
            },
            'CoV_ISI': {
                'target': 1.5407573082169885,
                'min': 0.09795245112054385,
                'max': 5.309807350265894,
                'weight': 1,
            },
        },
    },
    'bursting_data': {
        'bursting_summary_data': {
            'MeanWithinBurstISI': {
                'target': 0.02922164743789626,
                'min': 0.005933333333332828,
                'max': 0.09940000000000282,
                'weight': 1,
            },
            'CovWithinBurstISI': {
                'target': 0.0005404776859221188,
                'min': 8.000000000531145e-08,
                'max': 0.00227137999999947,
                'weight': 1,
            },
            'MeanOutsideBurstISI': {
                'target': 2.4194958016336567,
                'min': 0.15281871968962205,
                'max': 222.0746,
                'weight': 1,
            },
            'CoVOutsideBurstISI': {
                'target': 44.37059120795159,
                'min': 0.005583332765737877,
                'max': 21836.865414843334,
                'weight': 1,
            },
            'MeanNetworkISI': {
                'target': 1.009971435507435,
                'min': 0.023541959798994976,
                'max': 222.0746,
                'weight': 1,
            },
            'CoVNetworkISI': {
                'target': 19.552489465742415,
                'min': 0.001046061965974361,
                'max': 21836.865414843334,
                'weight': 1,
            },
            'NumUnits': {
                'target': 852,
            },
            'Number_Bursts': {
                'target': 81,
                'min': 1,
                'max': None,
                'weight': 1,
            },
            'mean_IBI': {
                'target': 3.6625,
                'min': None,
                'max': None,
                'weight': 1,
            },
            'cov_IBI': {
                'target': 2.649462025316461,
                'min': None,
                'max': None,
                'weight': 1,
            },
            'mean_Burst_Peak': {
                'target': 2.963130862629896,
                'min': None,
                'max': None,
                'weight': 1,
            },
            'cov_Burst_Peak': {
                'target': 0.482038017037109,
                'min': None,
                'max': None,
                'weight': 1,
            },
            'fano_factor': {
                'target': 1340.9514103561949,
                'min': None,
                'max': None,
                'weight': 1,
            },
            'baseline': {
                'target': 756.6503175331021,
                'min': None,
                'max': None,
                'weight': 1,
            },
        },
    },
}

'''FITNESS FUNCTION ARGUMENTS'''
# fitness function
fitnessFuncArgs = {}
fitnessFuncArgs['targets'] = fitness_args
number_of_units = fitness_args['bursting_data']['bursting_summary_data']['NumUnits']['target']
number_of_units = 30
fitnessFuncArgs['features'] = {
    'num_excite': int(number_of_units * 0.25),
    'num_inhib': int(number_of_units * 0.75),
}
fitnessFuncArgs['maxFitness'] = 1000