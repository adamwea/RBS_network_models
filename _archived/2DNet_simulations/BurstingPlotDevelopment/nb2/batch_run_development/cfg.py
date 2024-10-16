from netpyne import specs

cfg = specs.SimConfig()

cfg.networkType = 'aw'#'simple' # 'complex'

# --------------------------------------------------------
# Simple network
# --------------------------------------------------------
if cfg.networkType == 'simple':
	# Simulation options
	cfg.dt = 0.025
	cfg.duration = 2*1e3

	cfg.verbose = False
	cfg.saveJson = True
	cfg.filename = 'simple_net'
	cfg.saveDataInclude = ['simData']
	cfg.recordStep = 0.1
	cfg.printPopAvgRates = [500, cfg.duration]

	# cfg.recordCells = [1]
	# cfg.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}

	# Variable parameters (used in netParams)
	cfg.prob = 0.2
	cfg.weight = 0.025
	cfg.delay = 2

# --------------------------------------------------------
# awa network
# --------------------------------------------------------
elif cfg.networkType == 'aw':
	seconds = 2
	cfg.duration = seconds*1e3           # Duration of the simulation, in ms
	cfg.dt = 0.025                # Internal integration timestep to use
	cfg.verbose = False            # Show detailed messages
	cfg.recordStep = 0.1             # Step size in ms to save data (eg. V traces, LFP, etc)
	cfg.filename = 'aw_net'  # Set file output name
	cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net']
	cfg.saveJson = True
	#cfg.savePickle = True
	cfg.printPopAvgRates = [100, cfg.duration]

	#Include Oscillatory behavior
	cfg.recordLFP = [[50, 50, 50]]
	cfg.recordDipole = True

	# Variable parameters (used in netParams)
	# cfg.probEall = 0.1
	# cfg.weightEall = 0.005
	# cfg.probIE = 0.4
	# cfg.weightIE = 0.001
	# cfg.probLengthConst = 150
	# cfg.stimWeight = 0.1

	cfg.filename = 'aw_grid'
	# cfg.analysis['plotRaster'] = {'showFig': True, 'saveFig' : True}                  # Plot a raster
	# cfg.analysis['plot2Dnet'] = {'showConns': False, 'saveFig': True}                  # plot 2D cell positions and connections
	# cfg.analysis['plotConn'] = {'saveFig': True, 'showFig': True}                  # plot connectivity matrix

	

# --------------------------------------------------------
# Complex network
# --------------------------------------------------------
elif cfg.networkType == 'complex':
	cfg.duration = 1*1e3           # Duration of the simulation, in ms
	cfg.dt = 0.1                # Internal integration timestep to use
	cfg.verbose = False            # Show detailed messages
	cfg.recordStep = 1             # Step size in ms to save data (eg. V traces, LFP, etc)
	cfg.filename = 'simple_net'   # Set file output name
	cfg.saveDataInclude = ['simData']
	cfg.saveJson = True
	cfg.printPopAvgRates = [100, cfg.duration]

	# Variable parameters (used in netParams)
	cfg.probEall = 0.1
	cfg.weightEall = 0.005
	cfg.probIE = 0.4
	cfg.weightIE = 0.001
	cfg.probLengthConst = 150
	cfg.stimWeight = 0.1
