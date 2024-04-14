from netpyne import specs, sim
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
	#load batch_config from batch_config.json
	with open(f'{batch_run_path}/batch_config.json', 'r') as f:
		batch_config = json.load(f)
	batch_method = batch_config['method']
except Exception as e:
	print(f"{e}")
	#from simConfig import cfg
	#from simConfig import batch

## Generate or Load Constant Network Parameters
if 'evol' in batch_method:
	sys.path.append(working_dir)
	from evol_param_config import define_population_params
	netParams = define_population_params(batch_run_path = batch_run_path)
elif 'grid' in batch_method:
	print('Error, grid method not yet implemented')
else:
	print('Error, invalid method specified in batch_config')

# --------------------------------------------------------
# network param types
# --------------------------------------------------------
if cfg.networkType == 'pre13Apr24': #Network used for grant proposal in 01Apr24

	## Simulation Duration
 	#update last cfg params
	# try: 
	# 	sys.argv[2]
	# 	seconds = int(sys.argv[2])
	# except: 
	# 	#sys.argv.append('100')
	# 	seconds = cfg.duration_seconds
	# print(f"Duration: {seconds} seconds")
	seconds = cfg.duration_seconds
	cfg.duration = seconds*1e3           # Duration of the simulation, in ms

	##General network parameters
	netParams.propVelocity = cfg.propVelocity # propagation velocity (um/ms)
	netParams.probLengthConst = cfg.probLengthConst # length constant for conn probability (um)

	## Cell property rules
	cellRule = {'conds': {'cellType': 'E'},  'secs': {}}  # cell rule dict
	
	cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}                              # soma params dict
	cellRule['secs']['soma']['geom'] = {
		'diam': 18.8, 
		'L': 18.8, 
		'Ra': 123.0}                   # soma geometry
	cellRule['secs']['soma']['mechs']['hh'] = {
		'gnabar': cfg.gnabar_E, 
		'gkbar': cfg.gkbar_E, 
		'gl': 0.003, 
		'el': -70}      # soma hh mechanism
	netParams.cellParams['Erule'] = cellRule                          # add dict to list of cell params

	cellRule = {'conds': {'cellType': 'I'},  'secs': {}}  # cell rule dict
	cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}                              # soma params dict
	cellRule['secs']['soma']['geom'] = {
		'diam': 10.0, 
		'L': 9.0, 
		'Ra': 110.0}                  # soma geometry
	cellRule['secs']['soma']['mechs']['hh'] = {
		'gnabar': cfg.gnabar_I, 
		'gkbar': cfg.gkbar_I, 
		'gl': 0.003, 
		'el': -70}      # soma hh mechanism
	netParams.cellParams['Irule'] = cellRule                          # add dict to list of cell params

	## Synaptic mechanism parameters
	netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 
								#    'tau1': 0.8, 
								#    'tau2': 5.3, 
								   'tau1': cfg.tau1_exc, # Rise time constant (ms)
									'tau2': cfg.tau2_exc, # Decay time constant (ms)
									#'weight': cfg.exc_synaptic_weight, # Excitatory synaptic weight (mS/cm^2)
								   'e': 0}  # NMDA synaptic mechanism
	netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 
								   'tau1': cfg.tau1_inh, # Rise time constant (ms)
									'tau2': cfg.tau2_inh, # Decay time constant (ms)
									#'weight': cfg.inh_synaptic_weight, # Inhibitory synaptic weight (mS/cm^2)
									'e': -75}  # GABA synaptic mechanism


	# Stimulation parameters
	netParams.stimSourceParams['bkg'] = {'type': 
									  'NetStim', 
									  'rate': 20, 
									  'noise': 0.3}
	netParams.stimTargetParams['bkg->all'] = {'source': 
										   'bkg', 
										   'conds': {'cellType': ['E','I']}, 
										   'weight': cfg.stimWeight, 
										   'delay': 'max(1, normal(5,2))', 
										   'synMech': 'exc'}
 
	# Define a new stimulation source that provides rhythmic stimulation
	netParams.stimSourceParams['rhythmic'] = {
		'type': 'NetStim', 
		'interval': 1000.0/cfg.stim_rate,  # Interval in ms (1 Hz = 1000 ms)
		'number': 1000,  # Number of spikes (use a large number)
		'start': 1,  # Start time of the first spike
		'noise': cfg.stim_noise  # Noise percentage (0 for no noise)
	}

	# Connect the rhythmic stimulation source to the 'E' population
	netParams.stimTargetParams['rhythmic->E'] = {
		'source': 'rhythmic', 
		'conds': {'cellType': ['E']}, 
		'weight': cfg.stimWeight, 
		'sec': 'soma', 
		'delay': 'max(1, normal(5,2))', 
		'synMech': 'exc'
	}

	# Connect the rhythmic stimulation source to the 'I' population
	netParams.stimTargetParams['rhythmic->I'] = {
		'source': 'rhythmic', 
		'conds': {'cellType': ['I']}, 
		'weight': cfg.stimWeight*2/10, 
		'sec': 'soma', 
		'delay': 'max(1, normal(5,2))', 
		'synMech': 'exc'
	}

	# ## Cell connectivity rules
	netParams.connParams['E->I'] = {
	  'preConds': {'cellType': 'E'}, 'postConds': {'pop': 'I'},  #  E -> all (100-2000 um)
	  'probability': str(cfg.probEI)+'*exp(-dist_3D/probLengthConst)',  # adjust mu and sigma as needed
	  'weight': str(cfg.weightEI)+'*post_ynorm',         # synaptic weight
	  'delay': 'dist_3D/propVelocity',      # transmission delay (ms)	
	  'synMech': 'exc'}                     # synaptic mechanism

	netParams.connParams['I->E'] = {
	  'preConds': {'cellType': 'I'}, 'postConds': {'pop': 'E'},       #  I -> E
	  'probability': str(cfg.probIE)+'*exp(-dist_3D/probLengthConst)',   # probability of connection
	  'weight': str(cfg.weightIE)+'*post_ynorm',                                       # synaptic weight
	  'delay': 'dist_3D/propVelocity',                      # transmission delay (ms)
	  'synMech': 'inh'}                                     # synaptic mechanism	

	## E -> E recurrent connectivity
	netParams.connParams['E->E'] = {
		'preConds': {'pop': 'E'}, 'postConds': {'pop': 'E'},
		'probability': str(cfg.probEE)+'*exp(-dist_3D/probLengthConst)',  # adjust mu and sigma as needed
		'weight': str(cfg.weightEE)+'*post_ynorm',  
		'delay': 'dist_3D/propVelocity', 
		'synMech': 'exc'
	}

	## I -> I recurrent connectivity
	netParams.connParams['I->I'] = {
		'preConds': {'pop': 'I'}, 'postConds': {'pop': 'I'},
		'probability': str(cfg.probII)+'*exp(-dist_3D/probLengthConst)',  # adjust as needed
		'weight':  str(cfg.weightII)+'*post_ynorm',   
		'delay': 'dist_3D/propVelocity', 
		'synMech': 'inh'
    }
