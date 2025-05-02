fitness_args = {
    'source': 'experimental',
    'spiking_data': {
        'spiking_summary_data': {
            'MeanFireRate': {
                'target': 1.3493960522578143,
                'min': 0.0066657778962804955,
                'max': 42.5643247566991,
                'weight': 1,
            },
            'CoVFireRate': {
                'target': 2.591323362187853,
                'min': 0.2103127524797561,
                'max': 5.905062197129402,
                'weight': 1,
            },
            'MeanISI': {
                'target': 3.7804899153453517,
                'min': 0.02347847298355521,
                'max': 69.02876666666667,
                'weight': 1,
            },
            'CoV_ISI': {
                'target': 1.7804453437461392,
                'min': 0.2103127524797561,
                'max': 5.905062197129402,
                'weight': 1,
            },
        },
    },
    'bursting_data': {
        'bursting_summary_data': {
            'MeanWithinBurstISI': {
                'target': 0.029192188540145263,
                'min': 0.0014999999999645297,
                'max': 0.09329999999999927,
                'weight': 1,
            },
            'CovWithinBurstISI': {
                'target': 0.000548383905158979,
                'min': 4.499999999984354e-06,
                'max': 0.00227137999999947,
                'weight': 1,
            },
            'MeanOutsideBurstISI': {
                'target': 1.7670706555370062,
                'min': 0.15285004840271066,
                'max': 118.51629999999997,
                'weight': 1,
            },
            'CoVOutsideBurstISI': {
                'target': 14.860041027096026,
                'min': 0.0058125521342031194,
                'max': 3593.9447805433338,
                'weight': 1,
            },
            'MeanNetworkISI': {
                'target': 0.7139108533981939,
                'min': 0.02347847298355521,
                'max': 69.02876666666667,
                'weight': 1,
            },
            'CoVNetworkISI': {
                'target': 6.559491165054411,
                'min': 0.0009960420340060488,
                'max': 7015.532913919993,
                'weight': 1,
            },
            'NumUnits': {
                'target': 572,
            },
            'Number_Bursts': {
                'target': 329,
                'min': 1,
                'max': None,
                'weight': 1,
            },
            'mean_IBI': {
                'target': 0.9056707317073169,
                'min': None,
                'max': None,
                'weight': 1,
            },
            'cov_IBI': {
                'target': 1.7055029126575674,
                'min': None,
                'max': None,
                'weight': 1,
            },
            'mean_Burst_Peak': {
                'target': 6.610908472008412,
                'min': None,
                'max': None,
                'weight': 1,
            },
            'cov_Burst_Peak': {
                'target': 1.9829197478053402,
                'min': None,
                'max': None,
                'weight': 1,
            },
            'fano_factor': {
                'target': 1323.038087848163,
                'min': None,
                'max': None,
                'weight': 1,
            },
            'baseline': {
                'target': 1.33509444210873,
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

#override
#number_of_units_override = 400
#number_of_units = number_of_units_override

fitnessFuncArgs['features'] = {
    # 'num_excite': int(number_of_units * 0.80),
    # 'num_inhib': int(number_of_units * 0.20),
    
    #tweaking the ratio a bit based on Roy's suggestion
    'num_excite': int(number_of_units * 0.70),
    'num_inhib': int(number_of_units * 0.30),
    
}
fitnessFuncArgs['maxFitness'] = 1000