import os
import json
import numpy as np
from netpyne import specs

# Initialize NetParams object to store network parameters
netParams = specs.NetParams()

# Default 3D space ranges for random placement
DEFAULT_Y_RANGE = [6.72, 2065.09]
DEFAULT_X_RANGE = [173.66, 3549.91]
DEFAULT_Z_RANGE = [1, 11.61]

# Helper function to generate positive values based on a normal distribution
def positive_normal(mean, std):
    return abs(np.random.normal(mean, std))

# Function to randomly place neurons in 3D space if no exact positions are provided
def random_position(xRange, yRange, zRange):
    x = np.random.uniform(xRange[0], xRange[1])
    y = np.random.uniform(yRange[0], yRange[1])
    z = np.random.uniform(zRange[0], zRange[1])
    return x, y, z

# Generate cell parameters including exact positions (for both excitatory and inhibitory neurons)
def generate_neuron_params(cfg, neuron_id, cellType, x=None, y=None, z=None, xRange=None, yRange=None, zRange=None):
    cellRule = {'conds': {'cellType': cellType}, 'secs': {}}
    cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}
    cellRule['secs']['soma']['geom'] = {    
            'L': positive_normal(cfg.E_L_mean if cellType == 'E' else cfg.I_L_mean, 
                                 cfg.E_L_stdev if cellType == 'E' else cfg.I_L_stdev),  # Length
            'diam': positive_normal(cfg.E_diam_mean if cellType == 'E' else cfg.I_diam_mean, 
                                    cfg.E_diam_stdev if cellType == 'E' else cfg.I_diam_stdev),  # Diameter
            'Ra': positive_normal(cfg.E_Ra_mean if cellType == 'E' else cfg.I_Ra_mean, 
                                  cfg.E_Ra_stdev if cellType == 'E' else cfg.I_Ra_stdev),  # Axial resistance
        }
    cellRule['secs']['soma']['mechs']['hh'] = {
            'gnabar': positive_normal(cfg.gnabar_E if cellType == 'E' else cfg.gnabar_I, cfg.gnabar_E_std if cellType == 'E' else cfg.gnabar_I_std),  # Sodium conductance
            'gkbar': positive_normal(cfg.gkbar_I if cellType == 'I' else cfg.gkbar_E, cfg.gkbar_I_std if cellType == 'I' else cfg.gkbar_E_std),  # Potassium conductance
            'gl': 0.003,  # Leak conductance
            'el': -70,  # Leak reversal potential
            }

    # # Use predefined positions if provided, else generate randomly
    # xRange = xRange if xRange else DEFAULT_X_RANGE
    # yRange = yRange if yRange else DEFAULT_Y_RANGE
    # zRange = zRange if zRange else DEFAULT_Z_RANGE

    # cellRule['pointps'] = [{'loc': [x if x is not None else np.random.uniform(xRange[0], xRange[1]),
    #                                    y if y is not None else np.random.uniform(yRange[0], yRange[1]),
    #                                    z if z is not None else np.random.uniform(zRange[0], zRange[1])]}]
    
    return cellRule

# Define connections between neuron types (EE, EI, II, IE)
def define_conn_params(cfg):
    # Pull connection parameters from the cfg object
    probLengthConst = cfg.probLengthConst
    propVelocity = cfg.propVelocity

    # Define E -> I connection
    netParams.connParams['E->I'] = {
        'preConds': {'cellType': 'E'}, 'postConds': {'cellType': 'I'},
        'probability': 'abs(normal({}, {})) * exp(-dist_3D/{})'.format(cfg.probEI, cfg.probEI_std, probLengthConst),
        'weight': 'abs(normal({}, {}))'.format(cfg.weightEI, cfg.weightEI_std),
        'delay': 'dist_3D/{}'.format(propVelocity),
        'synMech': 'exc'
    }

    # Define I -> E connection
    netParams.connParams['I->E'] = {
        'preConds': {'cellType': 'I'}, 'postConds': {'cellType': 'E'},
        'probability': 'abs(normal({}, {})) * exp(-dist_3D/{})'.format(cfg.probIE, cfg.probIE_std, probLengthConst),
        'weight': 'abs(normal({}, {}))'.format(cfg.weightIE, cfg.weightIE_std),
        'delay': 'dist_3D/{}'.format(propVelocity),
        'synMech': 'inh'
    }

    # Define E -> E connection
    netParams.connParams['E->E'] = {
        'preConds': {'cellType': 'E'}, 'postConds': {'cellType': 'E'},
        'probability': 'abs(normal({}, {})) * exp(-dist_3D/{})'.format(cfg.probEE, cfg.probEE_std, probLengthConst),
        'weight': 'abs(normal({}, {}))'.format(cfg.weightEE, cfg.weightEE_std),
        'delay': 'dist_3D/{}'.format(propVelocity),
        'synMech': 'exc'
    }

    # Define I -> I connection
    netParams.connParams['I->I'] = {
        'preConds': {'cellType': 'I'}, 'postConds': {'cellType': 'I'},
        'probability': 'abs(normal({}, {})) * exp(-dist_3D/{})'.format(cfg.probII, cfg.probII_std, probLengthConst),
        'weight': 'abs(normal({}, {}))'.format(cfg.weightII, cfg.weightII_std),
        'delay': 'dist_3D/{}'.format(propVelocity),
        'synMech': 'inh'
    }

# Create the network with cells and connections
def create_network(cfg, numExcitatory, numInhibitory, cell_positions=None):
    #neurons = {'E': [], 'I': []}

    #Generate excitatory neurons
    for i in range(numExcitatory):
        position = cell_positions['E'][i] if cell_positions and 'E' in cell_positions else None
        netParams.cellParams[f'Exc_{i}rule'] = generate_neuron_params(cfg, i, 'E', *position)
    # cellRule = {'conds': {'cellType': 'E'},  'secs': {}}  # cell rule dict
    # cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}                              # soma params dict
    # cellRule['secs']['soma']['geom'] = {
	# 	'diam': 18.8, 
	# 	'L': 18.8, 
	# 	'Ra': 123.0}                   # soma geometry
    # cellRule['secs']['soma']['mechs']['hh'] = {
	# 	'gnabar': cfg.gnabar_E, 
	# 	'gkbar': cfg.gkbar_E, 
	# 	'gl': 0.003, 
	# 	'el': -70}      # soma hh mechanism
    # netParams.cellParams['Erule'] = cellRule                          # add dict to list of cell params



    
    netParams.synMechParams[f'exc'] = {'mod': 'Exp2Syn', 
                                        'tau1': cfg.tau1_exc,
                                        'tau2': cfg.tau2_exc,
                                        'e': 0} # AMPA synaptic mechanism / NMDA synaptic mechanism
    
    # Generate inhibitory neurons
    for i in range(numInhibitory):
        position = cell_positions['I'][i] if cell_positions and 'I' in cell_positions else None
        netParams.cellParams[f'Inh_{i}rule'] = generate_neuron_params(cfg, i + numExcitatory, 'I', *position)
    # cellRule = {'conds': {'cellType': 'I'},  'secs': {}}  # cell rule dic
    # cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}                              # soma params dict
    # cellRule['secs']['soma']['geom'] = {
	# 	'diam': 10.0, 
	# 	'L': 9.0, 
	# 	'Ra': 110.0}                  # soma geometry
    # cellRule['secs']['soma']['mechs']['hh'] = {
	# 	'gnabar': cfg.gnabar_I, 
	# 	'gkbar': cfg.gkbar_I, 
	# 	'gl': 0.003, 
	# 	'el': -70}      # soma hh mechanism
    # netParams.cellParams['Irule'] = cellRule                          # add dict to list of cell params
    
   
    netParams.synMechParams[f'inh'] = {'mod': 'Exp2Syn', 
                                        'tau1': cfg.tau1_inh,
                                        'tau2': cfg.tau2_inh,
                                        'e': -75} # GABA synaptic mechanism
    
    # Define popParams
    netParams.popParams['E'] = {'cellType': 'E', 'numCells': numExcitatory}
    netParams.popParams['I'] = {'cellType': 'I', 'numCells': numInhibitory}

# Save the network configuration to a JSON file
def save_network_to_json(filename):
    # net_params_json = {
    #     'cellParams': netParams.cellParams,
    #     'connParams': netParams.connParams
    # }
    net_params_json = netParams.__dict__

    with open(filename, 'w') as f:
        json.dump(net_params_json, f, indent=4)

#def main(cell_positions = None):
try:
    '''
    Define the parameter space for the evolutionary search
    '''
    from __main__ import cfg
    batch_run_path = os.path.dirname(cfg.saveFolder)
    filename = 'netParams.json'
except:
    print('cfg not found')
    raise Exception('cfg not found')
    # # Placeholder example of cfg in case it isn't available (for debug)
    # class Cfg:
    #     E_L_mean, E_L_stdev = 100, 20
    #     E_diam_mean, E_diam_stdev = 10, 2
    #     E_Ra_mean, E_Ra_stdev = 150, 30
    #     I_L_mean, I_L_stdev = 90, 15
    #     I_diam_mean, I_diam_stdev = 8, 1.5
    #     I_Ra_mean, I_Ra_stdev = 140, 20
    #     probEI, probEI_std = 0.3, 0.05
    #     weightEI, weightEI_std = 0.01, 0.002
    #     probEE, probEE_std = 0.4, 0.05
    #     weightEE, weightEE_std = 0.01, 0.002
    #     probIE, probIE_std = 0.2, 0.03
    #     weightIE, weightIE_std = 0.01, 0.002
    #     probII, probII_std = 0.2, 0.03
    #     weightII, weightII_std = 0.01, 0.002
    #     propVelocity = 500
    #     probLengthConst = 200

    # cfg = Cfg()
    # batch_run_path = './simulate_config_files/'
    # filename = 'netParams_debug.json'
    # if not os.path.exists(batch_run_path):
    #         os.makedirs(batch_run_path)

# Number of neurons
#from simulate._temp_files.temp_user_args import USER_num_excite, USER_num_inhib
#numExcitatory, numInhibitory = USER_num_excite, USER_num_inhib
numExcitatory, numInhibitory = cfg.num_excite, cfg.num_inhib

#Cell positions (optional)
cell_positions = {
    'E': [random_position(DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_Z_RANGE) for _ in range(numExcitatory)],
    'I': [random_position(DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_Z_RANGE) for _ in range(numInhibitory)]
}

# Create the network with excitatory and inhibitory neurons
create_network(cfg, numExcitatory, numInhibitory, cell_positions=cell_positions)


# Define the connection parameters for the 4 types (EE, EI, II, IE)
define_conn_params(cfg)

# Save the generated network
#save_network_to_json(os.path.join(batch_run_path, filename))
netParams.save(os.path.join(batch_run_path, filename))
print('Network parameters saved to', os.path.join(batch_run_path, filename))
# import sys
# sys.exit(0)

# Main execution block
# if __name__ == '__main__':
# 	main()
