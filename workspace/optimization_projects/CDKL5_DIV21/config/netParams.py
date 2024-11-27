import os
import numpy as np
from netpyne import specs
from scipy.stats import truncnorm

# Initialize NetParams object
netParams = specs.NetParams()

# Function for dynamic truncation to ensure symmetry
def dynamic_positive_normal(mean, std, size):    
    a, b = (0 - mean) / std, float("inf")
    initial_values = truncnorm.rvs(a, b, loc=mean, scale=std, size=size * 2)
    left_truncated_count = np.sum(initial_values < 0)
    upper_truncation_value = np.percentile(initial_values, 100 - (left_truncated_count / len(initial_values)) * 100)
    b_dynamic = (upper_truncation_value - mean) / std
    values = truncnorm.rvs(a, b_dynamic, loc=mean, scale=std, size=size)
    return values

# Main functions
def create_cell_params(cell_type, props, num_cells):
    cellRule = {
        'conds': {'cellType': cell_type},
        'secs': {
            'soma': {
                'geom': {
                    'L': props['L'].tolist(),
                    'diam': props['diam'].tolist(),
                    'Ra': props['Ra'].tolist(),
                },
                'mechs': {
                    'hh': {
                        'gnabar': props['gnabar'].tolist(),
                        'gkbar': props['gkbar'].tolist(),
                        'gl': [0.003] * num_cells, 'el': [-70] * num_cells,
                    }
                }
            }
        }
    }
    netParams.cellParams[cell_type] = cellRule

def define_populations(cfg):
    netParams.popParams['E'] = {
        'cellType': 'E',
        'numCells': cfg.num_excite,
        'cellModel': 'HH'
    }
    netParams.popParams['I'] = {
        'cellType': 'I',
        'numCells': cfg.num_inhib,
        'cellModel': 'HH'
    }

def define_connectivity(cfg):
    from time import time
    start = time()
    
    dynamic_normal_applied = False
    #dynamic_normal_applied = True - too complex, not implemented, not needed for now
    
    if not dynamic_normal_applied:
        print('Static normal applied')
        print('Generating static normal values for connectivity properties...')
        conn_specs = {
            'E->I': {'pre': 'E', 'post': 'I', 'synMech': 'exc', 
                     'prob': cfg.probEI, 
                     'weight': cfg.weightEI},
            'I->E': {'pre': 'I', 'post': 'E', 'synMech': 'inh', 
                     'prob': cfg.probIE, 
                     'weight': cfg.weightIE},
            'E->E': {'pre': 'E', 'post': 'E', 'synMech': 'exc', 
                     'prob': cfg.probEE, 
                     'weight': cfg.weightEE},
            'I->I': {'pre': 'I', 'post': 'I', 'synMech': 'inh', 
                     'prob': cfg.probII, 
                     'weight': cfg.weightII},
        }
        for key, spec in conn_specs.items():
            netParams.connParams[key] = {
                'preConds': {'cellType': spec['pre']},
                'postConds': {'cellType': spec['post']},
                'probability': spec['prob'],
                'weight': spec['weight'],
                'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                'synMech': spec['synMech']
            }

        netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': cfg.tau1_exc, 'tau2': cfg.tau2_exc, 'e': 0}
        netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': cfg.tau1_inh, 'tau2': cfg.tau2_inh, 'e': -75}
        
        print('Connectivity properties generated')
        time_elapsed = time() - start
        print(f'Connectivity properties generated in {time_elapsed:.2f} seconds')
    else:
        print('Dynamic normal applied')
        print('Generating dynamic normal values for connectivity properties...')
        
        EI_props = {
            'prob': dynamic_positive_normal(cfg.probEI, cfg.probEI_std, cfg.num_excite * cfg.num_inhib),
            'weight': dynamic_positive_normal(cfg.weightEI, cfg.weightEI_std, cfg.num_excite * cfg.num_inhib),
        }
        IE_props = {
            'prob': dynamic_positive_normal(cfg.probIE, cfg.probIE_std, cfg.num_inhib * cfg.num_excite),
            'weight': dynamic_positive_normal(cfg.weightIE, cfg.weightIE_std, cfg.num_inhib * cfg.num_excite),
        }
        EE_props = {
            'prob': dynamic_positive_normal(cfg.probEE, cfg.probEE_std, cfg.num_excite * cfg.num_excite),
            'weight': dynamic_positive_normal(cfg.weightEE, cfg.weightEE_std, cfg.num_excite * cfg.num_excite),
        }
        II_props = {
            'prob': dynamic_positive_normal(cfg.probII, cfg.probII_std, cfg.num_inhib * cfg.num_inhib),
            'weight': dynamic_positive_normal(cfg.weightII, cfg.weightII_std, cfg.num_inhib * cfg.num_inhib),
        }
        
        for i in range(cfg.num_excite):
            for j in range(cfg.num_inhib):
                netParams.connParams[f'E->I_{i}_{j}'] = {
                    'preConds': {'cellType': 'E', 'cellIndex': i},
                    'postConds': {'cellType': 'I', 'cellIndex': j},
                    'probability': EI_props['prob'][i * cfg.num_inhib + j],
                    'weight': EI_props['weight'][i * cfg.num_inhib + j],
                    'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                    'synMech': 'exc'
                }
                
        for i in range(cfg.num_inhib):
            for j in range(cfg.num_excite):
                netParams.connParams[f'I->E_{i}_{j}'] = {
                    'preConds': {'cellType': 'I', 'cellIndex': i},
                    'postConds': {'cellType': 'E', 'cellIndex': j},
                    'probability': IE_props['prob'][i * cfg.num_excite + j],
                    'weight': IE_props['weight'][i * cfg.num_excite + j],
                    'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                    'synMech': 'inh'
                }

        for i in range(cfg.num_excite):
            for j in range(cfg.num_excite):
                netParams.connParams[f'E->E_{i}_{j}'] = {
                    'preConds': {'cellType': 'E', 'cellIndex': i},
                    'postConds': {'cellType': 'E', 'cellIndex': j},
                    'probability': EE_props['prob'][i * cfg.num_excite + j],
                    'weight': EE_props['weight'][i * cfg.num_excite + j],
                    'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                    'synMech': 'exc'
                }
                
        for i in range(cfg.num_inhib):
            for j in range(cfg.num_inhib):
                netParams.connParams[f'I->I_{i}_{j}'] = {
                    'preConds': {'cellType': 'I', 'cellIndex': i},
                    'postConds': {'cellType': 'I', 'cellIndex': j},
                    'probability': II_props['prob'][i * cfg.num_inhib + j],
                    'weight': II_props['weight'][i * cfg.num_inhib + j],
                    'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                    'synMech': 'inh'
                }
        
        netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': cfg.tau1_exc, 'tau2': cfg.tau2_exc, 'e': 0}
        netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': cfg.tau1_inh, 'tau2': cfg.tau2_inh, 'e': -75}
        
        print('Connectivity properties generated')
        time_elapsed = time() - start
        print(f'Connectivity properties generated in {time_elapsed:.2f} seconds')
        
def create_network(cfg):
    # Generate properties for all neurons
    num_excite = cfg.num_excite
    num_inhib = cfg.num_inhib

    excitatory_props = {
        'L': dynamic_positive_normal(cfg.E_L_mean, cfg.E_L_stdev, num_excite),
        'diam': dynamic_positive_normal(cfg.E_diam_mean, cfg.E_diam_stdev, num_excite),
        'Ra': dynamic_positive_normal(cfg.E_Ra_mean, cfg.E_Ra_stdev, num_excite),
        'gnabar': dynamic_positive_normal(cfg.gnabar_E, cfg.gnabar_E_std, num_excite),
        'gkbar': dynamic_positive_normal(cfg.gkbar_E, cfg.gkbar_E_std, num_excite),
    }
    inhibitory_props = {
        'L': dynamic_positive_normal(cfg.I_L_mean, cfg.I_L_stdev, num_inhib),
        'diam': dynamic_positive_normal(cfg.I_diam_mean, cfg.I_diam_stdev, num_inhib),
        'Ra': dynamic_positive_normal(cfg.I_Ra_mean, cfg.I_Ra_stdev, num_inhib),
        'gnabar': dynamic_positive_normal(cfg.gnabar_I, cfg.gnabar_I_std, num_inhib),
        'gkbar': dynamic_positive_normal(cfg.gkbar_I, cfg.gkbar_I_std, num_inhib),
    }

    # Create cell rules
    create_cell_params('E', excitatory_props, num_excite)
    create_cell_params('I', inhibitory_props, num_inhib)

    # Define populations and connectivity
    define_populations(cfg)
    define_connectivity(cfg)

def save_network(cfg, filename):
    netParams.save(filename)
    print(f'Network parameters saved to {filename}')

# Main execution
def main(cfg):
    output_path = os.path.join(cfg.saveFolder, f"{cfg.simLabel}_netParams.json")
    create_network(cfg)
    save_network(cfg, output_path)

# Import cfg and run
try:
    from __main__ import cfg
    main(cfg)
except ImportError:
    raise Exception("cfg not found")
