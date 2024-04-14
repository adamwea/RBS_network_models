'''
These parameters are run seperately so that neuron locs are consistent across simulations.
'''

## NB3 Functions
from netpyne import specs

## Hold Neuron Locations Constant Across Simulations
#def get_const_net_params():
netParams = specs.NetParams()   # object of class NetParams to store the network parameters

## Population parameters
totalPop = 400
netParams.sizeX = 4000 # x-dimension (horizontal length) size in um
netParams.sizeY = 2000 # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = 0 # z-dimension (horizontal length) size in um
netParams.probLengthConst = 500 # length constant for conn probability (um)    
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
    
##Save network params to file
filename = 'const_netParams'
netParams.save(filename+'.json')