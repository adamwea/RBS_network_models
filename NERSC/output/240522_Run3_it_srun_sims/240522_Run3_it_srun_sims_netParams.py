#from netpyne import specs, sim
import json
import os
import sys
import os
import sys
print('im here second')

try:
	from __main__ import cfg
	batch_run_path = os.path.dirname(cfg.saveFolder)
	output_path = os.path.dirname(batch_run_path)
	working_dir = os.path.dirname(output_path)
	#load batch_config from batch_config.json
	#try:
	with open(f'{batch_run_path}/batch_config.json', 'r') as f:
		batch_config = json.load(f)
	batch_method = batch_config['method']
	# except: 
	# 	#only good for debug:
	# 	batch_method = 'evol'
	try:
		from simConfig import cfg
		for key, value in vars(cfg).items():
			print(f"{key}: {value}")
	except: print('no simConfig found')
except Exception as e:
	print(f"{e}")
	#from simConfig import cfg
	#from simConfig import batch

## Generate or Load Constant Network Parameters
if 'evol' in batch_method:
	sys.path.append(working_dir)
	from evol_param_setup import define_population_params
	netParams = define_population_params(batch_run_path = batch_run_path)
elif 'grid' in batch_method:
	print('Error, grid method not yet implemented')
else:
	print('Error, invalid method specified in batch_config')

# --------------------------------------------------------
# network param types
# --------------------------------------------------------

#cfg.networkType = 'pre13Apr24'  # Network used for grant proposal in 01Apr24	
# cfg.networkType = 'pre13Apr24'  # Network used for grant proposal in 01Apr24	
if cfg.networkType == '22May24':  # Network used for grant proposal in June 2024

	seconds = cfg.duration_seconds
	cfg.duration = seconds * 1e3  # Duration of the simulation, in ms

	## General network parameters
	netParams.propVelocity = cfg.propVelocity  # propagation velocity (um/ms)
	netParams.probLengthConst = cfg.probLengthConst  # length constant for conn probability (um)

	## Cell property rules

	# Excitatory cell rule
	diam = 18.8
	L = 18.8
	Ra = 123.0
	gnabar_E = cfg.gnabar_E
	gkbar_E = cfg.gkbar_E

	cellRule = {'conds': {'cellType': 'E'}, 'secs': {}}  # Initialize cell rule dict
	cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}  # Initialize soma params dict

	cellRule['secs']['soma']['geom'] = {
		'diam': 'abs(normal({}, {}))'.format(diam, diam/3),  # Ensure positive diameter
		'L': 'abs(normal({}, {}))'.format(L, L/3),  # Ensure positive length
		'Ra': 'abs(normal({}, {}))'.format(Ra, Ra/3)  # Ensure positive axial resistance
	}

	cellRule['secs']['soma']['mechs']['hh'] = {
		'gnabar': 'abs(normal({}, {}))'.format(gnabar_E, gnabar_E/3),  # Ensure positive gnabar
		'gkbar': 'abs(normal({}, {}))'.format(gkbar_E, gkbar_E/3),  # Ensure positive gkbar
		'gl': 0.003,
		'el': -70
	}

	netParams.cellParams['Erule'] = cellRule  # Add dict to list of cell params

	# Inhibitory cell rule
	diam = 10.0
	L = 9.0
	Ra = 110.0
	gnabar_I = cfg.gnabar_I
	gkbar_I = cfg.gkbar_I

	cellRule = {'conds': {'cellType': 'I'}, 'secs': {}}  # Initialize cell rule dict
	cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}  # Initialize soma params dict

	cellRule['secs']['soma']['geom'] = {
		'diam': 'abs(normal({}, {}))'.format(diam, diam/3),  # Ensure positive diameter
		'L': 'abs(normal({}, {}))'.format(L, L/3),  # Ensure positive length
		'Ra': 'abs(normal({}, {}))'.format(Ra, Ra/3)  # Ensure positive axial resistance
	}

	cellRule['secs']['soma']['mechs']['hh'] = {
		'gnabar': 'abs(normal({}, {}))'.format(gnabar_I, gnabar_I/3),  # Ensure positive gnabar
		'gkbar': 'abs(normal({}, {}))'.format(gkbar_I, gkbar_I/3),  # Ensure positive gkbar
		'gl': 0.003,
		'el': -70
	}

	netParams.cellParams['Irule'] = cellRule  # Add dict to list of cell params

	## Synaptic mechanism parameters

	# Excitatory synaptic mechanism with Gaussian distribution for tau1 and tau2
	tau1_exc = cfg.tau1_exc
	tau2_exc = cfg.tau2_exc

	netParams.synMechParams['exc'] = {
		'mod': 'Exp2Syn',
		'tau1': 'abs(normal({}, {}))'.format(tau1_exc, tau1_exc/3),  # Ensure positive tau1
		'tau2': 'abs(normal({}, {}))'.format(tau2_exc, tau2_exc/3),  # Ensure positive tau2
		'e': 0  # NMDA synaptic mechanism
	}

	# Inhibitory synaptic mechanism with Gaussian distribution for tau1 and tau2
	tau1_inh = cfg.tau1_inh
	tau2_inh = cfg.tau2_inh

	netParams.synMechParams['inh'] = {
		'mod': 'Exp2Syn',
		'tau1': 'abs(normal({}, {}))'.format(tau1_inh, tau1_inh/3),  # Ensure positive tau1
		'tau2': 'abs(normal({}, {}))'.format(tau2_inh, tau2_inh/3),  # Ensure positive tau2
		'e': -75  # GABA synaptic mechanism
	}

	## Cell connectivity rules
	probEI = cfg.probEI
	weightEI = cfg.weightEI

	netParams.connParams['E->I'] = {
		'preConds': {'cellType': 'E'}, 'postConds': {'pop': 'I'},  # E -> all (100-2000 um)
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probEI, probEI/3),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightEI, weightEI/3),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'exc'  # Synaptic mechanism
	}

	probIE = cfg.probIE
	weightIE = cfg.weightIE

	netParams.connParams['I->E'] = {
		'preConds': {'cellType': 'I'}, 'postConds': {'pop': 'E'},  # I -> E
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probIE, probIE/3),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightIE, weightIE/3),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'inh'  # Synaptic mechanism
	}

	probEE = cfg.probEE
	weightEE = cfg.weightEE

	netParams.connParams['E->E'] = {
		'preConds': {'pop': 'E'}, 'postConds': {'pop': 'E'},
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probEE, probEE/3),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightEE, weightEE/3),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'exc'  # Synaptic mechanism
	}

	probII = cfg.probII
	weightII = cfg.weightII

	netParams.connParams['I->I'] = {
		'preConds': {'pop': 'I'}, 'postConds': {'pop': 'I'},
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probII, probII/3),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightII, weightII/3),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'inh'  # Synaptic mechanism
	}

if cfg.networkType == 'pre13Apr24': #Network used for grant proposal in 01Apr24

	#print('ADLFKJAS;DLFKJ')
	#print to error
	#print(cfg)

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
