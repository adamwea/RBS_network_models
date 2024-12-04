import os
import numpy as np
from netpyne import specs
from scipy.stats import truncnorm

# Initialize NetParams object
netParams = specs.NetParams()

# Default ranges for random neuron placement
DEFAULT_Y_RANGE = [6.72, 2065.09]
DEFAULT_X_RANGE = [173.66, 3549.91]
DEFAULT_Z_RANGE = [1, 11.61]

# Function for generating random positive values based on Gaussian distributions - this is a bad idea
def positive_normal(mean, std, size):
    tries = 0
    # Ensure mean is sufficiently larger than std to increase likelihood of positive values
    if mean <= 0:
        raise ValueError("Mean must be positive")
    if std >= mean:
        raise ValueError("Standard deviation must be smaller than the mean")

    values = np.random.normal(mean, std, size)
    while np.any(values <= 0):
        values = np.random.normal(mean, std, size)
        tries += 1
        number_of_negative_values = np.sum(values <= 0)
        if number_of_negative_values > 0:
            print(f"Found {number_of_negative_values} negative values")
            print(f"Retrying {size} times to generate positive values")
    return values

# Function to implement dynamic truncation for symmetry
def dynamic_positive_normal(mean, std, size):    
    # Initial truncation only on the lower side
    a, b = (0 - mean) / std, float("inf")
    initial_values = truncnorm.rvs(a, b, loc=mean, scale=std, size=size * 2)  # Oversample
    
    # Count values truncated on the left
    left_truncated_count = np.sum(initial_values < 0)
    
    # Define the upper truncation limit to balance the truncation
    upper_truncation_value = np.percentile(initial_values, 100 - (left_truncated_count / len(initial_values)) * 100)
    b_dynamic = (upper_truncation_value - mean) / std  # Convert to standard normal scale
    
    # Resample with dynamic truncation limits
    values = truncnorm.rvs(a, b_dynamic, loc=mean, scale=std, size=size)
    return values


# Create cell rules and Gaussian-distributed properties for populations
def create_cell_rules(cfg, numExcitatory, numInhibitory):
    # Precompute properties for excitatory neurons
    excitatory_props = {
        'L': dynamic_positive_normal(cfg.E_L_mean, cfg.E_L_stdev, numExcitatory),
        'diam': dynamic_positive_normal(cfg.E_diam_mean, cfg.E_diam_stdev, numExcitatory),
        'Ra': dynamic_positive_normal(cfg.E_Ra_mean, cfg.E_Ra_stdev, numExcitatory),
        'gnabar': dynamic_positive_normal(cfg.gnabar_E, cfg.gnabar_E_std, numExcitatory),
        'gkbar': dynamic_positive_normal(cfg.gkbar_E, cfg.gkbar_E_std, numExcitatory),
    }

    # Define excitatory cell rule
    cellRuleE = {
        'conds': {'cellType': 'E'},
        'secs': {
            'soma': {
                'geom': {
                    'L': excitatory_props['L'].tolist(),
                    'diam': excitatory_props['diam'].tolist(),
                    'Ra': excitatory_props['Ra'].tolist(),
                },
                'mechs': {
                    'hh': {
                        'gnabar': excitatory_props['gnabar'].tolist(),
                        'gkbar': excitatory_props['gkbar'].tolist(),
                        'gl': 0.003, 'el': -70,
                    }
                }
            }
        }
    }
    netParams.cellParams['E'] = cellRuleE

    # Precompute properties for inhibitory neurons
    inhibitory_props = {
        'L': dynamic_positive_normal(cfg.I_L_mean, cfg.I_L_stdev, numInhibitory),
        'diam': dynamic_positive_normal(cfg.I_diam_mean, cfg.I_diam_stdev, numInhibitory),
        'Ra': dynamic_positive_normal(cfg.I_Ra_mean, cfg.I_Ra_stdev, numInhibitory),
        'gnabar': dynamic_positive_normal(cfg.gnabar_I, cfg.gnabar_I_std, numInhibitory),
        'gkbar': dynamic_positive_normal(cfg.gkbar_I, cfg.gkbar_I_std, numInhibitory),
    }

    # Define inhibitory cell rule
    cellRuleI = {
        'conds': {'cellType': 'I'},
        'secs': {
            'soma': {
                'geom': {
                    'L': inhibitory_props['L'].tolist(),
                    'diam': inhibitory_props['diam'].tolist(),
                    'Ra': inhibitory_props['Ra'].tolist(),
                },
                'mechs': {
                    'hh': {
                        'gnabar': inhibitory_props['gnabar'].tolist(),
                        'gkbar': inhibitory_props['gkbar'].tolist(),
                        'gl': 0.003, 'el': -70,
                    }
                }
            }
        }
    }
    netParams.cellParams['I'] = cellRuleI

# Define populations
def define_populations(cfg, numExcitatory, numInhibitory):
    # Add populations with explicit parameters
    netParams.popParams['E'] = {
        'cellType': 'E',
        'numCells': numExcitatory,
        'xRange': DEFAULT_X_RANGE, 'yRange': DEFAULT_Y_RANGE, 'zRange': DEFAULT_Z_RANGE
    }
    netParams.popParams['I'] = {
        'cellType': 'I',
        'numCells': numInhibitory,
        'xRange': DEFAULT_X_RANGE, 'yRange': DEFAULT_Y_RANGE, 'zRange': DEFAULT_Z_RANGE
    }

# Define connectivity rules
def define_connectivity(cfg, numExcitatory, numInhibitory):
    EI_props = {
        'probability': dynamic_positive_normal(cfg.probEI, cfg.probEI_std, numExcitatory),
        'weight': dynamic_positive_normal(cfg.weightEI, cfg.weightEI_std, numExcitatory),
    }
    IE_props = {
        'probability': dynamic_positive_normal(cfg.probIE, cfg.probIE_std, numInhibitory),
        'weight': dynamic_positive_normal(cfg.weightIE, cfg.weightIE_std, numInhibitory),
    }
    EE_props = {
        'probability': dynamic_positive_normal(cfg.probEE, cfg.probEE_std, numExcitatory),
        'weight': dynamic_positive_normal(cfg.weightEE, cfg.weightEE_std, numExcitatory),
    }
    II_props = {
        'probability': dynamic_positive_normal(cfg.probII, cfg.probII_std, numInhibitory),
        'weight': dynamic_positive_normal(cfg.weightII, cfg.weightII_std, numInhibitory),
    }    
    
    # Define E -> I connection
    netParams.connParams['E->I'] = {
        'preConds': {'cellType': 'E'}, 'postConds': {'cellType': 'I'},
        'probability': EI_props['probability'],
        'weight': EI_props['weight'],
        'delay': 'dist_3D / {}'.format(cfg.propVelocity),
        'synMech': 'exc'
    }
    
    # Define I -> E connection
    netParams.connParams['I->E'] = {
        'preConds': {'cellType': 'I'}, 'postConds': {'cellType': 'E'},
        'probability': IE_props['probability'],
        'weight': IE_props['weight'],
        'delay': 'dist_3D / {}'.format(cfg.propVelocity),
        'synMech': 'inh'
    }
    
    # Define E -> E connection
    netParams.connParams['E->E'] = {
        'preConds': {'cellType': 'E'}, 'postConds': {'cellType': 'E'},
        'probability': EE_props['probability'],
        'weight': EE_props['weight'],
        'delay': 'dist_3D / {}'.format(cfg.propVelocity),
        'synMech': 'exc'
    }
    
    # Define I -> I connection
    netParams.connParams['I->I'] = {
        'preConds': {'cellType': 'I'}, 'postConds': {'cellType': 'I'},
        'probability': II_props['probability'],
        'weight': II_props['weight'],
        'delay': 'dist_3D / {}'.format(cfg.propVelocity),
        'synMech': 'inh'
    }

    netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': cfg.tau1_exc, 'tau2': cfg.tau2_exc, 'e': 0}
    netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': cfg.tau1_inh, 'tau2': cfg.tau2_inh, 'e': -75}

# Save to JSON
def save_network(cfg, filename):
    netParams.save(filename)
    print(f'Network parameters saved to {filename}')

# Main execution
def main(cfg):
    batch_run_path = os.path.dirname(cfg.saveFolder)
    simLabel = cfg.simLabel
    #sim label example: gen_0_cand_0
    #gen_dir would be gen_0
    #get gen_dir from simLabel
    gen_dir = simLabel.split('_')[0] + '_' + simLabel.split('_')[1]
    filename = os.path.join(batch_run_path, gen_dir, f'{simLabel}_netParams.json')
    print('filename:', filename)

    numExcitatory = cfg.num_excite
    numInhibitory = cfg.num_inhib

    create_cell_rules(cfg, numExcitatory, numInhibitory)
    define_populations(cfg, numExcitatory, numInhibitory)
    define_connectivity(cfg, numExcitatory, numInhibitory)
    
    # Save the generated network
    print('Saving network parameters to', filename)
    save_network(cfg, filename)
    # import sys
    # sys.exit(0)

'''main script'''
#if __name__ == '__main__':
try:
    from __main__ import cfg
    print('cfg found in __main__')
except:
    print('cfg not found')
    raise Exception('cfg not found')   
#print(cfg.simLabel)
#import sys
#sys.exit(0)

main(cfg)
    
    



# #def main(cell_positions = None):
# try:
#     '''
#     Define the parameter space for the evolutionary search
#     '''
#     from __main__ import cfg
#     batch_run_path = os.path.dirname(cfg.saveFolder)
#     filename = 'netParams.json'
# except:
#     print('cfg not found')
#     raise Exception('cfg not found')
#     # # Placeholder example of cfg in case it isn't available (for debug)
#     # class Cfg:
#     #     E_L_mean, E_L_stdev = 100, 20
#     #     E_diam_mean, E_diam_stdev = 10, 2
#     #     E_Ra_mean, E_Ra_stdev = 150, 30
#     #     I_L_mean, I_L_stdev = 90, 15
#     #     I_diam_mean, I_diam_stdev = 8, 1.5
#     #     I_Ra_mean, I_Ra_stdev = 140, 20
#     #     probEI, probEI_std = 0.3, 0.05
#     #     weightEI, weightEI_std = 0.01, 0.002
#     #     probEE, probEE_std = 0.4, 0.05
#     #     weightEE, weightEE_std = 0.01, 0.002
#     #     probIE, probIE_std = 0.2, 0.03
#     #     weightIE, weightIE_std = 0.01, 0.002
#     #     probII, probII_std = 0.2, 0.03
#     #     weightII, weightII_std = 0.01, 0.002
#     #     propVelocity = 500
#     #     probLengthConst = 200

#     # cfg = Cfg()
#     # batch_run_path = './simulate_config_files/'
#     # filename = 'netParams_debug.json'
#     # if not os.path.exists(batch_run_path):
#     #         os.makedirs(batch_run_path)

# # Number of neurons
# #from simulate._temp_files.temp_user_args import USER_num_excite, USER_num_inhib
# #numExcitatory, numInhibitory = USER_num_excite, USER_num_inhib
# numExcitatory, numInhibitory = cfg.num_excite, cfg.num_inhib

# #Cell positions (optional)
# cell_positions = {
#     'E': [random_position(DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_Z_RANGE) for _ in range(numExcitatory)],
#     'I': [random_position(DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_Z_RANGE) for _ in range(numInhibitory)]
# }

# # Create the network with excitatory and inhibitory neurons
# create_network(cfg, numExcitatory, numInhibitory, cell_positions=cell_positions)


# # Define the connection parameters for the 4 types (EE, EI, II, IE)
# define_conn_params(cfg)

# # Save the generated network
# #save_network_to_json(os.path.join(batch_run_path, filename))
# netParams.save(os.path.join(batch_run_path, filename))
# print('Network parameters saved to', os.path.join(batch_run_path, filename))
# # import sys
# # sys.exit(0)

# # Main execution block
# # if __name__ == '__main__':
# # 	main()
