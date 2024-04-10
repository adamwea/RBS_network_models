from netpyne import specs
import json
cfg = specs.SimConfig()

cfg.networkType = 'aw'#'simple' # 'complex'

# --------------------------------------------------------
# awa network
# --------------------------------------------------------
if cfg.networkType == 'aw':
	#seconds = cfg.duration_seconds
	cfg.duration = 60 *1e3           # Duration of the simulation, in ms
	cfg.dt = 0.025                # Internal integration timestep to use
	cfg.verbose = False            # Show detailed messages
	cfg.recordStep = 0.1             # Step size in ms to save data (eg. V traces, LFP, etc)
	cfg.filename = 'aw_net'  # Set file output name
	cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net']
	cfg.saveJson = True
	#cfg.savePickle = True
	cfg.printPopAvgRates = [100, cfg.duration]

	#http://doc.netpyne.org/user_documentation.html#simconfig-recordtraces
	# record voltage at the center of the 'soma' section
	#cfg.recordTraces['soma_voltage'] = { "sec": "soma", "loc": 0.5, "var": "v"}
	cfg.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
	cfg.recordStep = 0.1 
	cfg.analysis['plotRaster'] = {'saveFig': True}                   # Plot a raster
	cfg.analysis['plotTraces'] = {'include': [0], 'saveFig': True}  # Plot recorded traces for this lis
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

	#variable parameters
	# Cell Params
	cfg.propVelocity = 1

	with open("./const_netParams.json", 'r') as file:
		data = json.load(file)

	# Access the probLengthConst value
	cfg.prob_length_const = data['net']['params']['probLengthConst']

	cfg.probLengthConst = 0.1

	cfg.probEall = 0.2
	
	cfg.weightEall = 0.0025
	#cfg.prop_weightEall = 0.0025/10
	
	cfg.probIE = 0.4
	
	cfg.weightIE = 0.005*10

	cfg.probEI = 0.2
	
	cfg.weightEI = 0.002*10

	cfg.probEE = 0.2
	
	cfg.weightEE = 0.0025
	#cfg.prop_weightEE = 0.0025*10
	
	cfg.probII = 0.4
	
	cfg.weightII = 0.005*10
	
	cfg.gnabar_E = 0.2
	
	cfg.gkbar_E = 0.05
	#cfg.prop_gkbar_E = 0.05*10
	
	#cfg.prop_gnabar_I = 0.15

	cfg.gnabar_I = 1.5
	
	cfg.gkbar_I = 0.05
	
	cfg.tau1_exc = 0.8
	
	cfg.tau2_exc = 6.0
	
	cfg.tau1_inh = 0.8
	
	cfg.tau2_inh = 9.0
	
	# Stimulation Params
	cfg.stimWeight = 0.02
	#cfg.prop_stimWeight = 0.02/10
	
	cfg.stim_rate = 15
	
	cfg.stim_noise = 0.4
