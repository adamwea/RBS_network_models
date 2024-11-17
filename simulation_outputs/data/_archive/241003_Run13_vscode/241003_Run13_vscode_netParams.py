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
	with open(f'{batch_run_path}/batch_config.json', 'r') as f:
		batch_config = json.load(f)
	batch_method = batch_config['method']
except Exception as e:
	print(f"{e}")

## Generate or Load Constant Network Parameters
if 'evol' in batch_method:
	sys.path.append(working_dir)
	#from evol_param_setup import define_population_params
	import evol_param_setup
	netParams = evol_param_setup.define_population_params(batch_run_path = batch_run_path)
elif 'grid' in batch_method:
	print('Error, grid method not yet implemented')
else:
	print('Error, invalid method specified in batch_config')

# --------------------------------------------------------
# network param types
# --------------------------------------------------------
if cfg.networkType == '27Sep24':  # Network used for grant proposal in June 2024

	#This fix is kinda hacky, but it works for now
	try: seconds = cfg.duration_seconds
	except: from evol_param_space import params; seconds = params['duration_seconds'][0]
	 
	cfg.duration = seconds * 1e3  # Duration of the simulation, in ms

	## General network parameters
	netParams.propVelocity = cfg.propVelocity  # propagation velocity (um/ms)
	netParams.probLengthConst = cfg.probLengthConst  # length constant for conn probability (um)

	## Cell property rules
	# Excitatory cell rule
	diam = cfg.E_diam_mean
	try: assert cfg.E_diam_stdev <= diam/3, 'E_diam_stdev must be less than 1/3 of E_diam_mean'
	except: cfg.E_diam_stdev = diam/3
	diam_stdev = cfg.E_diam_stdev

	L = cfg.E_L_mean
	try: assert cfg.E_L_stdev <= L/3, 'E_L_stdev must be less than 1/3 of E_L_mean'
	except: cfg.E_L_stdev = L/3
	L_stdev = cfg.E_L_stdev

	Ra = cfg.E_Ra_mean
	try: assert cfg.E_Ra_stdev <= Ra/3, 'E_Ra_stdev must be less than 1/3 of E_Ra_mean'
	except: cfg.E_Ra_stdev = Ra/3
	Ra_stdev = cfg.E_Ra_stdev

	gnabar_E = cfg.gnabar_E
	try: assert cfg.gnabar_E_std <= gnabar_E/3, 'gnabar_E_std must be less than 1/3 of gnabar_E'
	except: cfg.gnabar_E_std = gnabar_E/3
	gnabar_E_std = cfg.gnabar_E_std

	gkbar_E = cfg.gkbar_E
	try: assert cfg.gkbar_E_std <= gkbar_E/3, 'gkbar_E_std must be less than 1/3 of gkbar_E'
	except: cfg.gkbar_E_std = gkbar_E/3
	gkbar_E_std = cfg.gkbar_E_std

	cellRule = {'conds': {'cellType': 'E'}, 'secs': {}}  # Initialize cell rule dict
	cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}  # Initialize soma params dict

	cellRule['secs']['soma']['geom'] = {
		'diam': 'abs(normal({}, {}))'.format(diam, diam_stdev),  # Ensure positive diameter
		'L': 'abs(normal({}, {}))'.format(L, L_stdev),  # Ensure positive length
		'Ra': 'abs(normal({}, {}))'.format(Ra, Ra_stdev)  # Ensure positive axial resistance
	}

	cellRule['secs']['soma']['mechs']['hh'] = {
		'gnabar': 'abs(normal({}, {}))'.format(gnabar_E, gnabar_E_std),  # Ensure positive gnabar
		'gkbar': 'abs(normal({}, {}))'.format(gkbar_E, gkbar_E_std),  # Ensure positive gkbar
		'gl': 0.003,
		'el': -70
	}

	netParams.cellParams['Erule'] = cellRule  # Add dict to list of cell params

	# Inhibitory cell rule
	diam = cfg.I_diam_mean
	try: assert cfg.I_diam_stdev <= diam/3, 'I_diam_stdev must be less than 1/3 of I_diam_mean'
	except: cfg.I_diam_stdev = diam/3
	diam_stdev = cfg.I_diam_stdev

	L = cfg.I_L_mean
	try: assert cfg.I_L_stdev <= L/3, 'I_L_stdev must be less than 1/3 of I_L_mean'
	except: cfg.I_L_stdev = L/3
	L_stdev = cfg.I_L_stdev

	Ra = cfg.I_Ra_mean
	try: assert cfg.I_Ra_stdev <= Ra/3, 'I_Ra_stdev must be less than 1/3 of I_Ra_mean'
	except: cfg.I_Ra_stdev = Ra/3
	Ra_stdev = cfg.I_Ra_stdev

	gnabar_I = cfg.gnabar_I
	try: assert cfg.gnabar_I_std <= gnabar_I/3, 'gnabar_I_std must be less than 1/3 of gnabar_I'
	except: cfg.gnabar_I_std = gnabar_I/3
	gnabar_I_std = cfg.gnabar_I_std

	gkbar_I = cfg.gkbar_I
	try: assert cfg.gkbar_I_std <= gkbar_I/3, 'gkbar_I_std must be less than 1/3 of gkbar_I'
	except: cfg.gkbar_I_std = gkbar_I/3
	gkbar_I_std = cfg.gkbar_I_std

	cellRule = {'conds': {'cellType': 'I'}, 'secs': {}}  # Initialize cell rule dict
	cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}  # Initialize soma params dict

	cellRule['secs']['soma']['geom'] = {
		'diam': 'abs(normal({}, {}))'.format(diam, diam_stdev),  # Ensure positive diameter
		'L': 'abs(normal({}, {}))'.format(L, L_stdev),  # Ensure positive length
		'Ra': 'abs(normal({}, {}))'.format(Ra, Ra_stdev)  # Ensure positive axial resistance
	}

	cellRule['secs']['soma']['mechs']['hh'] = {
		'gnabar': 'abs(normal({}, {}))'.format(gnabar_I, gnabar_I_std),  # Ensure positive gnabar
		'gkbar': 'abs(normal({}, {}))'.format(gkbar_I, gkbar_I_std),  # Ensure positive gkbar
		'gl': 0.003,
		'el': -70
	}

	netParams.cellParams['Irule'] = cellRule  # Add dict to list of cell params

	## Synaptic mechanism parameters

	# Excitatory synaptic mechanism with Gaussian distribution for tau1 and tau2
	tau1_exc = cfg.tau1_exc
	try: assert cfg.tau1_exc_std <= tau1_exc/3, 'tau1_exc_std must be less than 1/3 of tau1_exc'
	except: cfg.tau1_exc_std = tau1_exc/3
	tau1_exc_std = cfg.tau1_exc_std

	tau2_exc = cfg.tau2_exc
	try: assert cfg.tau2_exc_std <= tau2_exc/3, 'tau2_exc_std must be less than 1/3 of tau2_exc'
	except: cfg.tau2_exc_std = tau2_exc/3
	tau2_exc_std = cfg.tau2_exc_std

	netParams.synMechParams['exc'] = {
		'mod': 'Exp2Syn',
		'tau1': 'abs(normal({}, {}))'.format(tau1_exc, 	tau1_exc_std),  # Ensure positive tau1
		'tau2': 'abs(normal({}, {}))'.format(tau2_exc, tau2_exc_std),  # Ensure positive tau2
		'e': 0  # NMDA synaptic mechanism
	}
	# Inhibitory synaptic mechanism with Gaussian distribution for tau1 and tau2
	tau1_inh = cfg.tau1_inh
	try: assert cfg.tau1_inh_std <= tau1_inh/3, 'tau1_inh_std must be less than 1/3 of tau1_inh'
	except: cfg.tau1_inh_std = tau1_inh/3
	tau1_inh_std = cfg.tau1_inh_std

	tau2_inh = cfg.tau2_inh
	try: assert cfg.tau2_inh_std <= tau2_inh/3, 'tau2_inh_std must be less than 1/3 of tau2_inh'
	except: cfg.tau2_inh_std = tau2_inh/3
	tau2_inh_std = cfg.tau2_inh_std

	netParams.synMechParams['inh'] = {
		'mod': 'Exp2Syn',
		'tau1': 'abs(normal({}, {}))'.format(tau1_inh, tau1_inh_std),  # Ensure positive tau1
		'tau2': 'abs(normal({}, {}))'.format(tau2_inh, tau2_inh_std),  # Ensure positive tau2
		'e': -75  # GABA synaptic mechanism
	}

	# Cell connectivity rules
	probEI = cfg.probEI
	try: assert cfg.probEI_std <= probEI/3, 'probEI_std must be less than 1/3 of probEI'
	except: cfg.probEI_std = probEI/3
	probEI_std = cfg.probEI_std

	weightEI = cfg.weightEI
	try: assert cfg.weightEI_std <= weightEI/3, 'weightEI_std must be less than 1/3 of weightEI'
	except: cfg.weightEI_std = weightEI/3
	weightEI_std = cfg.weightEI_std

	netParams.connParams['E->I'] = {
		'preConds': {'cellType': 'E'}, 'postConds': {'pop': 'I'},  # E -> all (100-2000 um)
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probEI, probEI_std),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightEI, weightEI_std),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'exc'  # Synaptic mechanism
	}

	# Cell connectivity rules
	probIE = cfg.probIE
	try: assert cfg.probIE_std <= probIE/3, 'probIE_std must be less than 1/3 of probIE'
	except: cfg.probIE_std = probIE/3
	probIE_std = cfg.probIE_std

	weightIE = cfg.weightIE
	try: assert cfg.weightIE_std <= weightIE/3, 'weightIE_std must be less than 1/3 of weightIE'
	except: cfg.weightIE_std = weightIE/3
	weightIE_std = cfg.weightIE_std

	netParams.connParams['I->E'] = {
		'preConds': {'cellType': 'I'}, 'postConds': {'pop': 'E'},  # I -> E
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probIE, probIE_std),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightIE, weightIE_std),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'inh'  # Synaptic mechanism
	}

	# Cell connectivity rules
	probEE = cfg.probEE
	try: assert cfg.probEE_std < probEE/3, 'probEE_std must be less than 1/3 of probEE'
	except: cfg.probEE_std = probEE/3
	probEE_std = cfg.probEE_std

	weightEE = cfg.weightEE
	try: assert cfg.weightEE_std < weightEE/3, 'weightEE_std must be less than 1/3 of weightEE'
	except: cfg.weightEE_std = weightEE/3
	weightEE_std = cfg.weightEE_std

	netParams.connParams['E->E'] = {
		'preConds': {'pop': 'E'}, 'postConds': {'pop': 'E'},
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probEE, probEE_std),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightEE, weightEE_std),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'exc'  # Synaptic mechanism
	}

	# Cell connectivity rules
	probII = cfg.probII
	try: assert cfg.probII_std < probII/3, 'probII_std must be less than 1/3 of probII'
	except: cfg.probII_std = probII/3
	probII_std = cfg.probII_std

	weightII = cfg.weightII
	try: assert cfg.weightII_std < weightII/3, 'weightII_std must be less than 1/3 of weightII'
	except: cfg.weightII_std = weightII/3
	weightII_std = cfg.weightII_std

	netParams.connParams['I->I'] = {
		'preConds': {'pop': 'I'}, 'postConds': {'pop': 'I'},
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probII, probII_std),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightII, weightII_std),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'inh'  # Synaptic mechanism
	}
elif cfg.networkType == '05June24':  # Network used for grant proposal in June 2024

	#This fix is kinda hacky, but it works for now
	try: seconds = cfg.duration_seconds
	except: from evol_param_space import params; seconds = params['duration_seconds']

	sys.exit()
	cfg.duration = seconds * 1e3  # Duration of the simulation, in ms

	## General network parameters
	netParams.propVelocity = cfg.propVelocity  # propagation velocity (um/ms)
	netParams.probLengthConst = cfg.probLengthConst  # length constant for conn probability (um)

	## Cell property rules
	# Excitatory cell rule
	diam = cfg.E_diam_mean
	try: assert cfg.E_diam_stdev <= diam/3, 'E_diam_stdev must be less than 1/3 of E_diam_mean'
	except: cfg.E_diam_stdev = diam/3
	diam_stdev = cfg.E_diam_stdev

	L = cfg.E_L_mean
	try: assert cfg.E_L_stdev <= L/3, 'E_L_stdev must be less than 1/3 of E_L_mean'
	except: cfg.E_L_stdev = L/3
	L_stdev = cfg.E_L_stdev

	Ra = cfg.E_Ra_mean
	try: assert cfg.E_Ra_stdev <= Ra/3, 'E_Ra_stdev must be less than 1/3 of E_Ra_mean'
	except: cfg.E_Ra_stdev = Ra/3
	Ra_stdev = cfg.E_Ra_stdev

	gnabar_E = cfg.gnabar_E
	try: assert cfg.gnabar_E_std <= gnabar_E/3, 'gnabar_E_std must be less than 1/3 of gnabar_E'
	except: cfg.gnabar_E_std = gnabar_E/3
	gnabar_E_std = cfg.gnabar_E_std

	gkbar_E = cfg.gkbar_E
	try: assert cfg.gkbar_E_std <= gkbar_E/3, 'gkbar_E_std must be less than 1/3 of gkbar_E'
	except: cfg.gkbar_E_std = gkbar_E/3
	gkbar_E_std = cfg.gkbar_E_std

	cellRule = {'conds': {'cellType': 'E'}, 'secs': {}}  # Initialize cell rule dict
	cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}  # Initialize soma params dict

	cellRule['secs']['soma']['geom'] = {
		'diam': 'abs(normal({}, {}))'.format(diam, diam_stdev),  # Ensure positive diameter
		'L': 'abs(normal({}, {}))'.format(L, L_stdev),  # Ensure positive length
		'Ra': 'abs(normal({}, {}))'.format(Ra, Ra_stdev)  # Ensure positive axial resistance
	}

	cellRule['secs']['soma']['mechs']['hh'] = {
		'gnabar': 'abs(normal({}, {}))'.format(gnabar_E, gnabar_E_std),  # Ensure positive gnabar
		'gkbar': 'abs(normal({}, {}))'.format(gkbar_E, gkbar_E_std),  # Ensure positive gkbar
		'gl': 0.003,
		'el': -70
	}

	netParams.cellParams['Erule'] = cellRule  # Add dict to list of cell params

	# Inhibitory cell rule
	diam = cfg.I_diam_mean
	try: assert cfg.I_diam_stdev <= diam/3, 'I_diam_stdev must be less than 1/3 of I_diam_mean'
	except: cfg.I_diam_stdev = diam/3
	diam_stdev = cfg.I_diam_stdev

	L = cfg.I_L_mean
	try: assert cfg.I_L_stdev <= L/3, 'I_L_stdev must be less than 1/3 of I_L_mean'
	except: cfg.I_L_stdev = L/3
	L_stdev = cfg.I_L_stdev

	Ra = cfg.I_Ra_mean
	try: assert cfg.I_Ra_stdev <= Ra/3, 'I_Ra_stdev must be less than 1/3 of I_Ra_mean'
	except: cfg.I_Ra_stdev = Ra/3
	Ra_stdev = cfg.I_Ra_stdev

	gnabar_I = cfg.gnabar_I
	try: assert cfg.gnabar_I_std <= gnabar_I/3, 'gnabar_I_std must be less than 1/3 of gnabar_I'
	except: cfg.gnabar_I_std = gnabar_I/3
	gnabar_I_std = cfg.gnabar_I_std

	gkbar_I = cfg.gkbar_I
	try: assert cfg.gkbar_I_std <= gkbar_I/3, 'gkbar_I_std must be less than 1/3 of gkbar_I'
	except: cfg.gkbar_I_std = gkbar_I/3
	gkbar_I_std = cfg.gkbar_I_std

	cellRule = {'conds': {'cellType': 'I'}, 'secs': {}}  # Initialize cell rule dict
	cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}  # Initialize soma params dict

	cellRule['secs']['soma']['geom'] = {
		'diam': 'abs(normal({}, {}))'.format(diam, diam_stdev),  # Ensure positive diameter
		'L': 'abs(normal({}, {}))'.format(L, L_stdev),  # Ensure positive length
		'Ra': 'abs(normal({}, {}))'.format(Ra, Ra_stdev)  # Ensure positive axial resistance
	}

	cellRule['secs']['soma']['mechs']['hh'] = {
		'gnabar': 'abs(normal({}, {}))'.format(gnabar_I, gnabar_I_std),  # Ensure positive gnabar
		'gkbar': 'abs(normal({}, {}))'.format(gkbar_I, gkbar_I_std),  # Ensure positive gkbar
		'gl': 0.003,
		'el': -70
	}

	netParams.cellParams['Irule'] = cellRule  # Add dict to list of cell params

	## Synaptic mechanism parameters

	# Excitatory synaptic mechanism with Gaussian distribution for tau1 and tau2
	tau1_exc = cfg.tau1_exc
	try: assert cfg.tau1_exc_std <= tau1_exc/3, 'tau1_exc_std must be less than 1/3 of tau1_exc'
	except: cfg.tau1_exc_std = tau1_exc/3
	tau1_exc_std = cfg.tau1_exc_std

	tau2_exc = cfg.tau2_exc
	try: assert cfg.tau2_exc_std <= tau2_exc/3, 'tau2_exc_std must be less than 1/3 of tau2_exc'
	except: cfg.tau2_exc_std = tau2_exc/3
	tau2_exc_std = cfg.tau2_exc_std

	netParams.synMechParams['exc'] = {
		'mod': 'Exp2Syn',
		'tau1': 'abs(normal({}, {}))'.format(tau1_exc, 	tau1_exc_std),  # Ensure positive tau1
		'tau2': 'abs(normal({}, {}))'.format(tau2_exc, tau2_exc_std),  # Ensure positive tau2
		'e': 0  # NMDA synaptic mechanism
	}
	# Inhibitory synaptic mechanism with Gaussian distribution for tau1 and tau2
	tau1_inh = cfg.tau1_inh
	try: assert cfg.tau1_inh_std <= tau1_inh/3, 'tau1_inh_std must be less than 1/3 of tau1_inh'
	except: cfg.tau1_inh_std = tau1_inh/3
	tau1_inh_std = cfg.tau1_inh_std

	tau2_inh = cfg.tau2_inh
	try: assert cfg.tau2_inh_std <= tau2_inh/3, 'tau2_inh_std must be less than 1/3 of tau2_inh'
	except: cfg.tau2_inh_std = tau2_inh/3
	tau2_inh_std = cfg.tau2_inh_std

	netParams.synMechParams['inh'] = {
		'mod': 'Exp2Syn',
		'tau1': 'abs(normal({}, {}))'.format(tau1_inh, tau1_inh_std),  # Ensure positive tau1
		'tau2': 'abs(normal({}, {}))'.format(tau2_inh, tau2_inh_std),  # Ensure positive tau2
		'e': -75  # GABA synaptic mechanism
	}

	# Cell connectivity rules
	probEI = cfg.probEI
	try: assert cfg.probEI_std <= probEI/3, 'probEI_std must be less than 1/3 of probEI'
	except: cfg.probEI_std = probEI/3
	probEI_std = cfg.probEI_std

	weightEI = cfg.weightEI
	try: assert cfg.weightEI_std <= weightEI/3, 'weightEI_std must be less than 1/3 of weightEI'
	except: cfg.weightEI_std = weightEI/3
	weightEI_std = cfg.weightEI_std

	netParams.connParams['E->I'] = {
		'preConds': {'cellType': 'E'}, 'postConds': {'pop': 'I'},  # E -> all (100-2000 um)
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probEI, probEI_std),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightEI, weightEI_std),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'exc'  # Synaptic mechanism
	}

	# Cell connectivity rules
	probIE = cfg.probIE
	try: assert cfg.probIE_std <= probIE/3, 'probIE_std must be less than 1/3 of probIE'
	except: cfg.probIE_std = probIE/3
	probIE_std = cfg.probIE_std

	weightIE = cfg.weightIE
	try: assert cfg.weightIE_std <= weightIE/3, 'weightIE_std must be less than 1/3 of weightIE'
	except: cfg.weightIE_std = weightIE/3
	weightIE_std = cfg.weightIE_std

	netParams.connParams['I->E'] = {
		'preConds': {'cellType': 'I'}, 'postConds': {'pop': 'E'},  # I -> E
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probIE, probIE_std),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightIE, weightIE_std),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'inh'  # Synaptic mechanism
	}

	# Cell connectivity rules
	probEE = cfg.probEE
	try: assert cfg.probEE_std < probEE/3, 'probEE_std must be less than 1/3 of probEE'
	except: cfg.probEE_std = probEE/3
	probEE_std = cfg.probEE_std

	weightEE = cfg.weightEE
	try: assert cfg.weightEE_std < weightEE/3, 'weightEE_std must be less than 1/3 of weightEE'
	except: cfg.weightEE_std = weightEE/3
	weightEE_std = cfg.weightEE_std

	netParams.connParams['E->E'] = {
		'preConds': {'pop': 'E'}, 'postConds': {'pop': 'E'},
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probEE, probEE_std),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightEE, weightEE_std),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'exc'  # Synaptic mechanism
	}

	# Cell connectivity rules
	probII = cfg.probII
	try: assert cfg.probII_std < probII/3, 'probII_std must be less than 1/3 of probII'
	except: cfg.probII_std = probII/3
	probII_std = cfg.probII_std

	weightII = cfg.weightII
	try: assert cfg.weightII_std < weightII/3, 'weightII_std must be less than 1/3 of weightII'
	except: cfg.weightII_std = weightII/3
	weightII_std = cfg.weightII_std

	netParams.connParams['I->I'] = {
		'preConds': {'pop': 'I'}, 'postConds': {'pop': 'I'},
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probII, probII_std),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightII, weightII_std),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'inh'  # Synaptic mechanism
	}
elif cfg.networkType == '22May24':  # Network used for grant proposal in June 2024

	seconds = cfg.duration_seconds
	cfg.duration = seconds * 1e3  # Duration of the simulation, in ms

	## General network parameters
	netParams.propVelocity = cfg.propVelocity  # propagation velocity (um/ms)
	netParams.probLengthConst = cfg.probLengthConst  # length constant for conn probability (um)

	## Cell property rules
	# Excitatory cell rule
	diam = cfg.E_diam_mean
	try: assert cfg.E_diam_stdev <= diam/3, 'E_diam_stdev must be less than 1/3 of E_diam_mean'
	except: cfg.E_diam_stdev = diam/3
	diam_stdev = cfg.E_diam_stdev

	L = cfg.E_L_mean
	try: assert cfg.E_L_stdev <= L/3, 'E_L_stdev must be less than 1/3 of E_L_mean'
	except: cfg.E_L_stdev = L/3
	L_stdev = cfg.E_L_stdev

	Ra = cfg.E_Ra_mean
	try: assert cfg.E_Ra_stdev <= Ra/3, 'E_Ra_stdev must be less than 1/3 of E_Ra_mean'
	except: cfg.E_Ra_stdev = Ra/3
	Ra_stdev = cfg.E_Ra_stdev

	gnabar_E = cfg.gnabar_E
	try: assert cfg.gnabar_E_std <= gnabar_E/3, 'gnabar_E_std must be less than 1/3 of gnabar_E'
	except: cfg.gnabar_E_std = gnabar_E/3
	gnabar_E_std = cfg.gnabar_E_std

	gkbar_E = cfg.gkbar_E
	try: assert cfg.gkbar_E_std <= gkbar_E/3, 'gkbar_E_std must be less than 1/3 of gkbar_E'
	except: cfg.gkbar_E_std = gkbar_E/3
	gkbar_E_std = cfg.gkbar_E_std

	cellRule = {'conds': {'cellType': 'E'}, 'secs': {}}  # Initialize cell rule dict
	cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}  # Initialize soma params dict

	cellRule['secs']['soma']['geom'] = {
		'diam': 'abs(normal({}, {}))'.format(diam, diam_stdev),  # Ensure positive diameter
		'L': 'abs(normal({}, {}))'.format(L, L_stdev),  # Ensure positive length
		'Ra': 'abs(normal({}, {}))'.format(Ra, Ra_stdev)  # Ensure positive axial resistance
	}

	cellRule['secs']['soma']['mechs']['hh'] = {
		'gnabar': 'abs(normal({}, {}))'.format(gnabar_E, gnabar_E_std),  # Ensure positive gnabar
		'gkbar': 'abs(normal({}, {}))'.format(gkbar_E, gkbar_E_std),  # Ensure positive gkbar
		'gl': 0.003,
		'el': -70
	}

	netParams.cellParams['Erule'] = cellRule  # Add dict to list of cell params

	# Inhibitory cell rule
	diam = cfg.I_diam_mean
	try: assert cfg.I_diam_stdev <= diam/3, 'I_diam_stdev must be less than 1/3 of I_diam_mean'
	except: cfg.I_diam_stdev = diam/3
	diam_stdev = cfg.I_diam_stdev

	L = cfg.I_L_mean
	try: assert cfg.I_L_stdev <= L/3, 'I_L_stdev must be less than 1/3 of I_L_mean'
	except: cfg.I_L_stdev = L/3
	L_stdev = cfg.I_L_stdev

	Ra = cfg.I_Ra_mean
	try: assert cfg.I_Ra_stdev <= Ra/3, 'I_Ra_stdev must be less than 1/3 of I_Ra_mean'
	except: cfg.I_Ra_stdev = Ra/3
	Ra_stdev = cfg.I_Ra_stdev

	gnabar_I = cfg.gnabar_I
	try: assert cfg.gnabar_I_std <= gnabar_I/3, 'gnabar_I_std must be less than 1/3 of gnabar_I'
	except: cfg.gnabar_I_std = gnabar_I/3
	gnabar_I_std = cfg.gnabar_I_std

	gkbar_I = cfg.gkbar_I
	try: assert cfg.gkbar_I_std <= gkbar_I/3, 'gkbar_I_std must be less than 1/3 of gkbar_I'
	except: cfg.gkbar_I_std = gkbar_I/3
	gkbar_I_std = cfg.gkbar_I_std

	cellRule = {'conds': {'cellType': 'I'}, 'secs': {}}  # Initialize cell rule dict
	cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}  # Initialize soma params dict

	cellRule['secs']['soma']['geom'] = {
		'diam': 'abs(normal({}, {}))'.format(diam, diam_stdev),  # Ensure positive diameter
		'L': 'abs(normal({}, {}))'.format(L, L_stdev),  # Ensure positive length
		'Ra': 'abs(normal({}, {}))'.format(Ra, Ra_stdev)  # Ensure positive axial resistance
	}

	cellRule['secs']['soma']['mechs']['hh'] = {
		'gnabar': 'abs(normal({}, {}))'.format(gnabar_I, gnabar_I_std),  # Ensure positive gnabar
		'gkbar': 'abs(normal({}, {}))'.format(gkbar_I, gkbar_I_std),  # Ensure positive gkbar
		'gl': 0.003,
		'el': -70
	}

	netParams.cellParams['Irule'] = cellRule  # Add dict to list of cell params

	## Synaptic mechanism parameters

	# Excitatory synaptic mechanism with Gaussian distribution for tau1 and tau2
	tau1_exc = cfg.tau1_exc
	try: assert cfg.tau1_exc_std <= tau1_exc/3, 'tau1_exc_std must be less than 1/3 of tau1_exc'
	except: cfg.tau1_exc_std = tau1_exc/3
	tau1_exc_std = cfg.tau1_exc_std

	tau2_exc = cfg.tau2_exc
	try: assert cfg.tau2_exc_std <= tau2_exc/3, 'tau2_exc_std must be less than 1/3 of tau2_exc'
	except: cfg.tau2_exc_std = tau2_exc/3
	tau2_exc_std = cfg.tau2_exc_std

	netParams.synMechParams['exc'] = {
		'mod': 'Exp2Syn',
		'tau1': 'abs(normal({}, {}))'.format(tau1_exc, 	tau1_exc_std),  # Ensure positive tau1
		'tau2': 'abs(normal({}, {}))'.format(tau2_exc, tau2_exc_std),  # Ensure positive tau2
		'e': 0  # NMDA synaptic mechanism
	}
	# Inhibitory synaptic mechanism with Gaussian distribution for tau1 and tau2
	tau1_inh = cfg.tau1_inh
	try: assert cfg.tau1_inh_std <= tau1_inh/3, 'tau1_inh_std must be less than 1/3 of tau1_inh'
	except: cfg.tau1_inh_std = tau1_inh/3
	tau1_inh_std = cfg.tau1_inh_std

	tau2_inh = cfg.tau2_inh
	try: assert cfg.tau2_inh_std <= tau2_inh/3, 'tau2_inh_std must be less than 1/3 of tau2_inh'
	except: cfg.tau2_inh_std = tau2_inh/3
	tau2_inh_std = cfg.tau2_inh_std

	netParams.synMechParams['inh'] = {
		'mod': 'Exp2Syn',
		'tau1': 'abs(normal({}, {}))'.format(tau1_inh, tau1_inh_std),  # Ensure positive tau1
		'tau2': 'abs(normal({}, {}))'.format(tau2_inh, tau2_inh_std),  # Ensure positive tau2
		'e': -75  # GABA synaptic mechanism
	}

	# Cell connectivity rules
	probEI = cfg.probEI
	try: assert cfg.probEI_std <= probEI/3, 'probEI_std must be less than 1/3 of probEI'
	except: cfg.probEI_std = probEI/3
	probEI_std = cfg.probEI_std

	weightEI = cfg.weightEI
	try: assert cfg.weightEI_std <= weightEI/3, 'weightEI_std must be less than 1/3 of weightEI'
	except: cfg.weightEI_std = weightEI/3
	weightEI_std = cfg.weightEI_std

	netParams.connParams['E->I'] = {
		'preConds': {'cellType': 'E'}, 'postConds': {'pop': 'I'},  # E -> all (100-2000 um)
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probEI, probEI_std),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightEI, weightEI_std),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'exc'  # Synaptic mechanism
	}

	# Cell connectivity rules
	probIE = cfg.probIE
	try: assert cfg.probIE_std <= probIE/3, 'probIE_std must be less than 1/3 of probIE'
	except: cfg.probIE_std = probIE/3
	probIE_std = cfg.probIE_std

	weightIE = cfg.weightIE
	try: assert cfg.weightIE_std <= weightIE/3, 'weightIE_std must be less than 1/3 of weightIE'
	except: cfg.weightIE_std = weightIE/3
	weightIE_std = cfg.weightIE_std

	netParams.connParams['I->E'] = {
		'preConds': {'cellType': 'I'}, 'postConds': {'pop': 'E'},  # I -> E
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probIE, probIE_std),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightIE, weightIE_std),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'inh'  # Synaptic mechanism
	}

	# Cell connectivity rules
	probEE = cfg.probEE
	try: assert cfg.probEE_std < probEE/3, 'probEE_std must be less than 1/3 of probEE'
	except: cfg.probEE_std = probEE/3
	probEE_std = cfg.probEE_std

	weightEE = cfg.weightEE
	try: assert cfg.weightEE_std < weightEE/3, 'weightEE_std must be less than 1/3 of weightEE'
	except: cfg.weightEE_std = weightEE/3
	weightEE_std = cfg.weightEE_std

	netParams.connParams['E->E'] = {
		'preConds': {'pop': 'E'}, 'postConds': {'pop': 'E'},
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probEE, probEE_std),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightEE, weightEE_std),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'exc'  # Synaptic mechanism
	}

	# Cell connectivity rules
	probII = cfg.probII
	try: assert cfg.probII_std < probII/3, 'probII_std must be less than 1/3 of probII'
	except: cfg.probII_std = probII/3
	probII_std = cfg.probII_std

	weightII = cfg.weightII
	try: assert cfg.weightII_std < weightII/3, 'weightII_std must be less than 1/3 of weightII'
	except: cfg.weightII_std = weightII/3
	weightII_std = cfg.weightII_std

	netParams.connParams['I->I'] = {
		'preConds': {'pop': 'I'}, 'postConds': {'pop': 'I'},
		'probability': 'abs(normal({}, {})) * exp(-dist_3D/probLengthConst)'.format(probII, probII_std),  # Connection probability
		'weight': 'abs(normal({}, {})) * post_ynorm'.format(weightII, weightII_std),  # Synaptic weight
		'delay': 'dist_3D/propVelocity',  # Transmission delay (ms)
		'synMech': 'inh'  # Synaptic mechanism
	}
elif cfg.networkType == 'pre13Apr24': #Network used for grant proposal in 01Apr24

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
