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

if cfg.networkType == '22May24': #Network used for grant proposal in 01Apr24

    seconds = cfg.duration_seconds
    cfg.duration = seconds * 1e3  # Duration of the simulation, in ms

    ## General network parameters
    netParams.propVelocity = cfg.propVelocity  # propagation velocity (um/ms)
    netParams.probLengthConst = cfg.probLengthConst  # length constant for conn probability (um)

    ## Cell property rules

    # Excitatory cell rule
    cellRule = {'conds': {'cellType': 'E'}, 'secs': {}}  # cell rule dict
    cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}  # soma params dict

    # Soma geometry with Gaussian distribution
    cellRule['secs']['soma']['geom'] = {
        'diam': 'h.normal(18.8, 1.0)',  # Diameter of the soma (mean=18.8, std=1.0)
        'L': 'h.normal(18.8, 1.0)',  # Length of the soma (mean=18.8, std=1.0)
        'Ra': 'h.normal(123.0, 1.0)'  # Axial resistance (mean=123.0, std=1.0)
    }

    # Soma hh mechanism with Gaussian distribution for gnabar and gkbar
    cellRule['secs']['soma']['mechs']['hh'] = {
        'gnabar': 'h.normal({}, 1.0)'.format(cfg.gnabar_E),
        'gkbar': 'h.normal({}, 1.0)'.format(cfg.gkbar_E),
        'gl': 0.003,
        'el': -70
    }

    netParams.cellParams['Erule'] = cellRule  # add dict to list of cell params

    # Inhibitory cell rule
    cellRule = {'conds': {'cellType': 'I'}, 'secs': {}}  # cell rule dict
    cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}  # soma params dict

    # Soma geometry with Gaussian distribution
    cellRule['secs']['soma']['geom'] = {
        'diam': 'h.normal(10.0, 1.0)',  # Diameter of the soma (mean=10.0, std=1.0)
        'L': 'h.normal(9.0, 1.0)',  # Length of the soma (mean=9.0, std=1.0)
        'Ra': 'h.normal(110.0, 1.0)'  # Axial resistance (mean=110.0, std=1.0)
    }

    # Soma hh mechanism with Gaussian distribution for gnabar and gkbar
    cellRule['secs']['soma']['mechs']['hh'] = {
        'gnabar': 'h.normal({}, 1.0)'.format(cfg.gnabar_I),
        'gkbar': 'h.normal({}, 1.0)'.format(cfg.gkbar_I),
        'gl': 0.003,
        'el': -70
    }

    netParams.cellParams['Irule'] = cellRule  # add dict to list of cell params

    ## Synaptic mechanism parameters

    # Excitatory synaptic mechanism with Gaussian distribution for tau1 and tau2
    netParams.synMechParams['exc'] = {
        'mod': 'Exp2Syn',
        'tau1': 'h.normal({}, 1.0)'.format(cfg.tau1_exc),  # Rise time constant (ms)
        'tau2': 'h.normal({}, 1.0)'.format(cfg.tau2_exc),  # Decay time constant (ms)
        'e': 0  # NMDA synaptic mechanism
    }

    # Inhibitory synaptic mechanism with Gaussian distribution for tau1 and tau2
    netParams.synMechParams['inh'] = {
        'mod': 'Exp2Syn',
        'tau1': 'h.normal({}, 1.0)'.format(cfg.tau1_inh),  # Rise time constant (ms)
        'tau2': 'h.normal({}, 1.0)'.format(cfg.tau2_inh),  # Decay time constant (ms)
        'e': -75  # GABA synaptic mechanism
    }

    ## Cell connectivity rules

    netParams.connParams['E->I'] = {
        'preConds': {'cellType': 'E'}, 'postConds': {'pop': 'I'},  # E -> all (100-2000 um)
        'probability': 'h.normal({}, 1.0) * exp(-dist_3D/probLengthConst)'.format(cfg.probEI),  # Connection probability
        'weight': 'h.normal({}, 1.0) * post_ynorm'.format(cfg.weightEI),  # Synaptic weight
        'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
        'synMech': 'exc'  # Synaptic mechanism
    }

    netParams.connParams['I->E'] = {
        'preConds': {'cellType': 'I'}, 'postConds': {'pop': 'E'},  # I -> E
        'probability': 'h.normal({}, 1.0) * exp(-dist_3D/probLengthConst)'.format(cfg.probIE),  # Connection probability
        'weight': 'h.normal({}, 1.0) * post_ynorm'.format(cfg.weightIE),  # Synaptic weight
        'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
        'synMech': 'inh'  # Synaptic mechanism
    }

    ## E -> E recurrent connectivity
    netParams.connParams['E->E'] = {
        'preConds': {'pop': 'E'}, 'postConds': {'pop': 'E'},
        'probability': 'h.normal({}, 1.0) * exp(-dist_3D/probLengthConst)'.format(cfg.probEE),  # Connection probability
        'weight': 'h.normal({}, 1.0) * post_ynorm'.format(cfg.weightEE),  # Synaptic weight
        'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
        'synMech': 'exc'  # Synaptic mechanism
    }

    ## I -> I recurrent connectivity
    netParams.connParams['I->I'] = {
        'preConds': {'pop': 'I'}, 'postConds': {'pop': 'I'},
        'probability': 'h.normal({}, 1.0) * exp(-dist_3D/probLengthConst)'.format(cfg.probII),  # Connection probability
        'weight': 'h.normal({}, 1.0) * post_ynorm'.format(cfg.weightII),  # Synaptic weight
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
