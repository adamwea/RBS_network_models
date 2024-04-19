from netpyne import specs

cfg = specs.SimConfig()

cfg.networkType = 'pre13Apr24' #Network used for grant proposal in 01Apr24

# --------------------------------------------------------
# awa network
# --------------------------------------------------------
if cfg.networkType == 'pre13Apr24':
	##
	#cache_efficient - Use CVode cache_efficient option to optimize load when running on many cores (default: False)
	cfg.cache_efficient = True

	#seconds = cfg.duration_seconds
	#cfg.duration = seconds*1e3           # Duration of the simulation, in ms
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
	# record voltage at the center of the 'soma' section
	cfg.recordTraces['soma_voltage'] = { "sec": "soma", "loc": 0.5, "var": "v"}
	# only record this trace from populations 'M' and 'S'
	# record from the first cell in populations 'M' and 'S'
	#cfg.recordCells = [('E', [0]), ('I', [0])]
	
	#Investigate Oscillatory behavior
	# cfg.recordLFP = [[50, 50, 50]]
	# cfg.recordDipole = True

	#cfg.filename = 'aw_grid'
	# cfg.analysis['plotRaster'] = {'showFig': True, 'saveFig' : True}                  # Plot a raster
	# cfg.analysis['plot2Dnet'] = {'showConns': False, 'saveFig': True}                  # plot 2D cell positions and connections
	# cfg.analysis['plotConn'] = {'saveFig': True, 'showFig': True}                  # plot connectivity matrix
