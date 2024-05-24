from netpyne import specs  
'''network parameters held constant across simulations'''
version = '23May2024' # version of network

'''23May2024'''
if version == '23May2024':
    ## Hold Neuron Locations Constant Across Simulations
    netParams = specs.NetParams()   # object of class NetParams to store the network parameters

    ## Population parameters
    totalPop = 144
    netParams.sizeX = 4000 # x-dimension (horizontal length) size in um
    netParams.sizeY = 2000 # y-dimension (vertical height or cortical depth) size in um
    netParams.sizeZ = 12 # z-dimension (horizontal length) size in um
    #netParams.probLengthConst = 500 # length constant for conn probability (um)    
    netParams.popParams['E'] = {
        'cellType': 'E', 
        'numCells': int(totalPop*0.70), 
        'yRange': [6.72,2065.09], 
        'xRange': [173.66,3549.91],
        'zRange': [1,11.61]}
    netParams.popParams['I'] = {
        'cellType': 'I', 
        'numCells': int(totalPop*0.30), 
        'yRange': [6.72,2065.09], 
        'xRange': [173.66,3549.91],
        'zRange': [1,11.61]}
    
    #specify specific exact xyz coordinates for each cell
    #TODO: implement later
    # netParams.popParams['E'] = {
    #     'cellType': 'E',
    #     'numCells': 2,
    #     'cellModel': 'HH',
    #     'cellpos': dict([(i, pos) for i, pos in enumerate([(1.5, 345.3, 1.2), (444, 343.2, 1)])])
    # }


        # cfg.recordTraces['soma_voltage'] = { "sec": "soma", "loc": 0.5, "var": "v"}
        # # only record this trace from populations 'M' and 'S'
        # # record from the first cell in populations 'M' and 'S'
        # cfg.recordCells = [('E', [0]), ('I', [0])]

'''pre-23May2024'''
if version == 'pre-23May2024':
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

