from netpyne import specs
cfg = specs.SimConfig()

# --------------------------------------------------------
# network iterations
# --------------------------------------------------------
cfg.networkType = '27Sep24' 
#cfg.networkType = '05June24' #Updating complexity of network, grant due in June 2024 was extended
#cfg.networkType = '22May24' #Network used for grant proposal in June 2024
#cfg.networkType = 'pre13Apr24' #Network used for grant proposal in 01Apr24
if cfg.networkType == '27Sep24':
	import random
	from netParams_constant import netParams

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
	num_Ecells = netParams.popParams['E']['numCells']
	num_Icells = netParams.popParams['I']['numCells']

	# Choose two random cells from each population
	E_cells = random.sample(range(num_Ecells), min(2, num_Ecells))
	I_cells = random.sample(range(num_Icells), min(2, num_Icells))

	cfg.recordCells = [('E', E_cells), ('I', I_cells)]
	print('Cells selected for recording:', cfg.recordCells)
if cfg.networkType == '22May24':
	##
	#cache_efficient - Use CVode cache_efficient option to optimize load when running on many cores (default: False)
	cfg.cache_efficient = True

	#seconds = cfg.duration_seconds
	#cfg.duration = seconds*1e3           # Duration of the simulation, in ms
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

	#http://doc.netpyne.org/user_documentation.html#simconfig-recordtraces
	cfg.recordTraces['soma_voltage'] = { "sec": "soma", "loc": 0.5, "var": "v"}	
	from netParams_constant import netParams
	num_Ecells = netParams.popParams['E']['numCells']
	num_Icells = netParams.popParams['I']['numCells']
	import random

	# Choose two random cells from each population
	E_cells = random.sample(range(num_Ecells), min(2, num_Ecells))
	I_cells = random.sample(range(num_Icells), min(2, num_Icells))

	cfg.recordCells = [('E', E_cells), ('I', I_cells)]
	#cfg.recordCells = [('E', 0), ('I', 0)]
	print(cfg.recordCells)
elif cfg.networkType == '05June24':
	import random
	from netParams_constant import netParams

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
	num_Ecells = netParams.popParams['E']['numCells']
	num_Icells = netParams.popParams['I']['numCells']

	# Choose two random cells from each population
	E_cells = random.sample(range(num_Ecells), min(2, num_Ecells))
	I_cells = random.sample(range(num_Icells), min(2, num_Icells))

	cfg.recordCells = [('E', E_cells), ('I', I_cells)]
	print('Cells selected for recording:', cfg.recordCells)
if cfg.networkType == '22May24':
	##
	#cache_efficient - Use CVode cache_efficient option to optimize load when running on many cores (default: False)
	cfg.cache_efficient = True

	#seconds = cfg.duration_seconds
	#cfg.duration = seconds*1e3           # Duration of the simulation, in ms
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

	#http://doc.netpyne.org/user_documentation.html#simconfig-recordtraces
	cfg.recordTraces['soma_voltage'] = { "sec": "soma", "loc": 0.5, "var": "v"}	
	from netParams_constant import netParams
	num_Ecells = netParams.popParams['E']['numCells']
	num_Icells = netParams.popParams['I']['numCells']
	import random

	# Choose two random cells from each population
	E_cells = random.sample(range(num_Ecells), min(2, num_Ecells))
	I_cells = random.sample(range(num_Icells), min(2, num_Icells))

	cfg.recordCells = [('E', E_cells), ('I', I_cells)]
	#cfg.recordCells = [('E', 0), ('I', 0)]
	print(cfg.recordCells)
if cfg.networkType == 'pre13Apr24':
	##
	#cache_efficient - Use CVode cache_efficient option to optimize load when running on many cores (default: False)
	cfg.cache_efficient = True

	#seconds = cfg.duration_seconds
	#cfg.duration = seconds*1e3           # Duration of the simulation, in ms
	print('im here first')
	cfg.dt = 0.025                # Internal integration timestep to use
	cfg.verbose = False            # Show detailed messages
	cfg.recordStep = 0.1             # Step size in ms to save data (eg. V traces, LFP, etc)
	#cfg.filename = 'aw_net'  # Set file output name
	cfg.saveDataInclude = [
		'simData', 
		'simConfig', 
		'netParams', 
		'net'
  		]
	cfg.saveJson = True
	#cfg.savePickle = True
	cfg.printPopAvgRates = [100, cfg.duration]

	#http://doc.netpyne.org/user_documentation.html#simconfig-recordtraces
	cfg.recordTraces['soma_voltage'] = { "sec": "soma", "loc": 0.5, "var": "v"}	
	from netParams_constant import netParams
	num_Ecells = netParams.popParams['E']['numCells']
	num_Icells = netParams.popParams['I']['numCells']
	import random

	# Choose two random cells from each population
	E_cells = random.sample(range(num_Ecells), min(2, num_Ecells))
	I_cells = random.sample(range(num_Icells), min(2, num_Icells))

	cfg.recordCells = [('E', E_cells), ('I', I_cells)]
	print(cfg.recordCells)


	
	#Investigate Oscillatory behavior
	# cfg.recordLFP = [[50, 50, 50]]
	# cfg.recordDipole = True

	#cfg.filename = 'aw_grid'
	# cfg.analysis['plotRaster'] = {'showFig': True, 'saveFig' : True}                  # Plot a raster
	# cfg.analysis['plot2Dnet'] = {'showConns': False, 'saveFig': True}                  # plot 2D cell positions and connections
	# cfg.analysis['plotConn'] = {'saveFig': True, 'showFig': True}                  # plot connectivity matrix
