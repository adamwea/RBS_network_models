from netpyne import specs    

## Hold Neuron Locations Constant Across Simulations
netParams = specs.NetParams()   # object of class NetParams to store the network parameters

## Population parameters
totalPop = 400
netParams.sizeX = 4000 # x-dimension (horizontal length) size in um
netParams.sizeY = 2000 # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = 0 # z-dimension (horizontal length) size in um
#netParams.probLengthConst = 500 # length constant for conn probability (um)    
netParams.popParams['E'] = {
    'cellType': 'E', 
    'numCells': int(totalPop*0.70), 
    'yRange': [100,1900], 
    'xRange': [100,3900]}
netParams.popParams['I'] = {
    'cellType': 'I', 
    'numCells': int(totalPop*0.30), 
    'yRange': [100,1900], 
    'xRange': [100,3900]}

	# cfg.recordTraces['soma_voltage'] = { "sec": "soma", "loc": 0.5, "var": "v"}
	# # only record this trace from populations 'M' and 'S'
	# # record from the first cell in populations 'M' and 'S'
	# cfg.recordCells = [('E', [0]), ('I', [0])]

