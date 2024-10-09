#from netpyne import specs, sim
import json
import os
import sys
import os
import sys

try:
	from __main__ import cfg
	batch_run_path = os.path.dirname(cfg.saveFolder)
	output_path = os.path.dirname(batch_run_path)
	working_dir = os.path.dirname(output_path)
	with open(f'{batch_run_path}/batch_config.json', 'r') as f:
		batch_config = json.load(f)
	batch_method = batch_config['method']
except Exception as e:
	print(f"{e}")

## Generate or Load Constant Network Parameters
if 'evol' in batch_method:
	sys.path.append(working_dir)
	#from evol_param_setup import define_population_params
	import simulate_config_files.deprecated.evol_param_setup as evol_param_setup
	netParams = evol_param_setup.define_population_params(batch_run_path = batch_run_path)
elif 'grid' in batch_method:
	print('Error, grid method not yet implemented')
else:
	print('Error, invalid method specified in batch_config')

# --------------------------------------------------------
# network param types
# --------------------------------------------------------
if cfg.networkType == '07Oct24':  # Network used for grant proposal in October 2024
	import numpy as np
	import math

	# Default ranges for 3D space if none are provided
	DEFAULT_Y_RANGE = [6.72, 2065.09]
	DEFAULT_X_RANGE = [173.66, 3549.91]
	DEFAULT_Z_RANGE = [1, 11.61]

	# Helper function to generate positive values based on a normal distribution
	def positive_normal(mean, std):
		return abs(np.random.normal(mean, std))

	# Function to randomly place neurons in 3D space
	def random_position(xRange, yRange, zRange):
		x = np.random.uniform(xRange[0], xRange[1])
		y = np.random.uniform(yRange[0], yRange[1])
		z = np.random.uniform(zRange[0], zRange[1])
		return x, y, z

	# Generate excitatory neuron parameters, including 3D position or using experimental data
	def generate_exc_neuron_params(cfg, neuron_id, xRange=None, yRange=None, zRange=None, x=None, y=None, z=None):
		params = {}
		params['id'] = neuron_id  # Assign a unique ID
		params['diam'] = positive_normal(cfg.E_diam_mean, cfg.E_diam_stdev)
		params['L'] = positive_normal(cfg.E_L_mean, cfg.E_L_stdev)
		params['Ra'] = positive_normal(cfg.E_Ra_mean, cfg.E_Ra_stdev)
		params['gnabar'] = positive_normal(cfg.gnabar_E, cfg.gnabar_E_std)
		params['gkbar'] = positive_normal(cfg.gkbar_E, cfg.gkbar_E_std)
		
		# Use default ranges if not provided
		xRange = xRange if xRange is not None else DEFAULT_X_RANGE
		yRange = yRange if yRange is not None else DEFAULT_Y_RANGE
		zRange = zRange if zRange is not None else DEFAULT_Z_RANGE
		
		# Assign predefined positions if provided, otherwise randomly generate positions
		params['x'] = x if x is not None else np.random.uniform(xRange[0], xRange[1])
		params['y'] = y if y is not None else np.random.uniform(yRange[0], yRange[1])
		params['z'] = z if z is not None else np.random.uniform(zRange[0], zRange[1])
		
		return params

	# Generate inhibitory neuron parameters, including 3D position or using experimental data
	def generate_inh_neuron_params(cfg, neuron_id, xRange=None, yRange=None, zRange=None, x=None, y=None, z=None):
		params = {}
		params['id'] = neuron_id  # Assign a unique ID
		params['diam'] = positive_normal(cfg.I_diam_mean, cfg.I_diam_stdev)
		params['L'] = positive_normal(cfg.I_L_mean, cfg.I_L_stdev)
		params['Ra'] = positive_normal(cfg.I_Ra_mean, cfg.I_Ra_stdev)
		params['gnabar'] = positive_normal(cfg.gnabar_I, cfg.gnabar_I_std)
		params['gkbar'] = positive_normal(cfg.gkbar_I, cfg.gkbar_I_std)

		# Use default ranges if not provided
		xRange = xRange if xRange is not None else DEFAULT_X_RANGE
		yRange = yRange if yRange is not None else DEFAULT_Y_RANGE
		zRange = zRange if zRange is not None else DEFAULT_Z_RANGE
		
		# Assign predefined positions if provided, otherwise randomly generate positions
		params['x'] = x if x is not None else np.random.uniform(xRange[0], xRange[1])
		params['y'] = y if y is not None else np.random.uniform(yRange[0], yRange[1])
		params['z'] = z if z is not None else np.random.uniform(zRange[0], zRange[1])

		return params

	# Calculate probability of connection based on distance and length constant
	def connection_probability(prob_mean, prob_std, dist_3D, probLengthConst):
		prob = positive_normal(prob_mean, prob_std) * math.exp(-dist_3D / probLengthConst)
		return min(max(prob, 0), 1)  # Ensure probability is between 0 and 1

	# Generate synaptic weights and delays based on distance
	def generate_synaptic_params(weight_mean, weight_std, dist_3D, post_ynorm, propVelocity):
		weight = positive_normal(weight_mean, weight_std) * post_ynorm
		delay = dist_3D / propVelocity
		return weight, delay

	# Generate connection between two neurons based on probability
	def generate_connection(pre_neuron, post_neuron, prob, weight_mean, weight_std, dist_3D, propVelocity, synMech, post_ynorm=1):
		if np.random.rand() < prob:
			weight, delay = generate_synaptic_params(weight_mean, weight_std, dist_3D, post_ynorm, propVelocity)
			connection = {
				'pre': pre_neuron,
				'post': post_neuron,
				'weight': weight,
				'delay': delay,
				'synMech': synMech
			}
			return connection
		return None

	import numpy as np

	def compute_distance_matrix(positions):
		num_neurons = len(positions)
		dist_matrix = np.zeros((num_neurons, num_neurons))

		for i in range(num_neurons):
			for j in range(num_neurons):
				dist_matrix[i, j] = np.linalg.norm(positions[i] - positions[j])

		return dist_matrix
		
	# Function to create the network with excitatory and inhibitory neurons and their connections
	def create_network(cfg, numExcitatory, numInhibitory, dist_matrix):
		# Store neurons
		neurons = {'E': [], 'I': []}
		
		# Generate excitatory and inhibitory neuron properties
		for i in range(numExcitatory):
			neurons['E'].append(generate_exc_neuron_params(cfg, neuron_id=i))
		for i in range(numInhibitory):
			neurons['I'].append(generate_inh_neuron_params(cfg, neuron_id=i + numExcitatory))
		
		# Compute distance matrix if not provided
		if not dist_matrix:
			#print(neurons['E'][0])
			#init np array
			positions = np.zeros((numExcitatory + numInhibitory, 3))
			for neuron in neurons['E'] + neurons['I']:
				#print(neuron)
				x = neuron['x']
				y = neuron['y']
				z = neuron['z']
				#print(x, y, z)
				positions[neuron['id']] = [x, y, z]
				#print(neuron['x', 'y', 'z'])
				#
			#positions = n
			#sys.exit()
			#positions = np.array([neuron['x', 'y', 'z'] for neuron in neurons['E'] + neurons['I']])
			dist_matrix = compute_distance_matrix(positions)
			#print(dist_matrix)
			#sys.exit()
		
		# Create connections
		connections = []

		# E->I connections
		for pre in neurons['E']:
			for post in neurons['I']:
				#print(post)
				#print(dist_matrix)
				#sys.exit()
				dist_3D = dist_matrix[pre['id'], post['id']]  # Placeholder for 3D distance
				prob = connection_probability(cfg.probEI, cfg.probEI_std, dist_3D, cfg.probLengthConst)
				conn = generate_connection(pre, post, prob, cfg.weightEI, cfg.weightEI_std, dist_3D, cfg.propVelocity, 'exc')
				if conn: connections.append(conn)
		
		# I->E connections
		for pre in neurons['I']:
			for post in neurons['E']:
				dist_3D = dist_matrix[pre['id'], post['id']]  # Placeholder for 3D distance
				prob = connection_probability(cfg.probIE, cfg.probIE_std, dist_3D, cfg.probLengthConst)
				conn = generate_connection(pre, post, prob, cfg.weightIE, cfg.weightIE_std, dist_3D, cfg.propVelocity, 'inh')
				if conn: connections.append(conn)

		# E->E connections
		for pre in neurons['E']:
			for post in neurons['E']:
				dist_3D = dist_matrix[pre['id'], post['id']]  # Placeholder for 3D distance
				prob = connection_probability(cfg.probEE, cfg.probEE_std, dist_3D, cfg.probLengthConst)
				conn = generate_connection(pre, post, prob, cfg.weightEE, cfg.weightEE_std, dist_3D, cfg.propVelocity, 'exc')
				if conn: connections.append(conn)

		# I->I connections
		for pre in neurons['I']:
			for post in neurons['I']:
				dist_3D = dist_matrix[pre['id'], post['id']]  # Placeholder for 3D distance
				prob = connection_probability(cfg.probII, cfg.probII_std, dist_3D, cfg.probLengthConst)
				conn = generate_connection(pre, post, prob, cfg.weightII, cfg.weightII_std, dist_3D, cfg.propVelocity, 'inh')
				if conn: connections.append(conn)

		return neurons, connections

	# Example usage: Saving network to JSON
	def save_network_to_json(neurons, connections, filename='netParams_direct.json'):
		net_params = {
			'cells': [],
			'conns': connections
		}

		for i, neuron in enumerate(neurons['E'] + neurons['I']):
			net_params['cells'].append({
				'gid': i,
				'secs': {
					'soma': {
						'geom': {
							'L': neuron['L'],
							'Ra': neuron['Ra'],
							'diam': neuron['diam']
						},
						'mechs': {
							'hh': {
								'gnabar': neuron['gnabar'],
								'gkbar': neuron['gkbar'],
								'gl': 0.003,  # fixed parameter
								'el': -70  # fixed parameter
							}
						}
					}
				}
			})

		with open(filename, 'w') as f:
			json.dump(net_params, f, indent=4)

	# Assuming you have numExcitatory, numInhibitory, dist_matrix, and cfg set up
	neurons, connections = create_network(cfg, numExcitatory=100, numInhibitory=43, dist_matrix={})  # Replace dist_matrix with actual data
	save_network_to_json(neurons, connections)
	sys.exit()
