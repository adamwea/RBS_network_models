sys.path.insert(0, './submodules/netpyne')
from netpyne import specs
cfg = specs.SimConfig()
# --------------------------------------------------------
# network iteration notes
# --------------------------------------------------------
cfg.networkType = '07Oct24' # Updating everything while working odxckl4
# --------------------------------------------------------
# network cfg iterations
# --------------------------------------------------------
if cfg.networkType == '07Oct24':
	import sys
	sys.path.append('simulate_config_files')
	from simulation_config_files.evolutionary_parameter_space import params
	for key, value in params.items():
		if isinstance(value, (int, float)):
			params[key] = [value, value]
		elif isinstance(value, list) and len(value) == 1:
			params[key] = [value[0], value[0]]
		#append params to cfg
		setattr(cfg, key, params[key])
	#evol_param_space = params
	#from simulate_config_files.deprecated.netParams_constant import netParams
	from simulate._temp_files.temp_user_args import USER_seconds

	cfg.duration_seconds = USER_seconds
	cfg.duration = cfg.duration_seconds*1e3           # Duration of the simulation, in ms
	cfg.cache_efficient = True 	#cache_efficient - Use CVode cache_efficient option to optimize load when running on many cores (default: False)
	print('im here first')
	cfg.dt = 0.025                # Internal integration timestep to use
	cfg.verbose = False            # Show detailed messages
	cfg.recordStep = 0.1             # Step size in ms to save data (eg. V traces, LFP, etc)
	cfg.saveDataInclude = [
		'simData', 
		'simConfig', 
		'netParams', 
		'net'
  		]
	cfg.saveJson = True
	cfg.printPopAvgRates = [100, cfg.duration]
	cfg.recordTraces['soma_voltage'] = { "sec": "soma", "loc": 0.5, "var": "v"}	 	#http://doc.netpyne.org/user_documentation.html#simconfig-recordtraces
	# Add netParam dependent cfg params
	import random
	# Choose two random cells from each population
	from simulate._temp_files.temp_user_args import USER_num_excite, USER_num_inhib
	numExcitatory, numInhibitory = USER_num_excite, USER_num_inhib
	E_cells = random.sample(range(numExcitatory), min(2, numExcitatory))
	I_cells = random.sample(range(numInhibitory), min(2, numInhibitory))

cfg.recordCells = [('E', E_cells), ('I', I_cells)]
print('Cells selected for recording:', cfg.recordCells)