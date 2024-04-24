# import debugpy

# # Allow other computers to attach to debugpy at this IP address and port.
# debugpy.listen(('0.0.0.0', 5678))

# # Pause the program until a remote debugger is attached
# debugpy.wait_for_client()

from netpyne import specs
from netpyne.batch import Batch

import os
import shutil

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Change the working directory to the script directory
os.chdir(script_dir) #+ '/batch_evol')

#print(os.getcwd())

''' Example of evolutionary algorithm optimization of a network using NetPyNE
To run use: mpiexec -np [num_cores] nrniv -mpi batchRun.py
'''

# if folder "aw_grid" exists, delete it
# if os.path.exists('aw_grid'):
#     shutil.rmtree('aw_grid')

def const_net_params():
	## Hold Neuron Locations Constant Across Simulations

    # %matplotlib inline
    # from netpyne import specs
    # from netpyne.batch import Batch

    netParams = specs.NetParams()   # object of class NetParams to store the network parameters

    netParams.sizeX = 4000 # x-dimension (horizontal length) size in um
    netParams.sizeY = 2000 # y-dimension (vertical height or cortical depth) size in um
    netParams.sizeZ = 0 # z-dimension (horizontal length) size in um
    # I think I will modulate these
    #netParams.propVelocity = 100.0     # propagation velocity (um/ms)
    netParams.probLengthConst = 500 # length constant for conn probability (um)

    ## Population parameters
    netParams.popParams['E'] = {
        'cellType': 'E', 
        'numCells': 300, 
        'yRange': [100,1900], 
        'xRange': [100,3900]}
    netParams.popParams['I'] = {
        'cellType': 'I', 
        'numCells': 100, 
        'yRange': [100,1900], 
        'xRange': [100,3900]}

    filename = 'constant_neuron_locs'
    netParams.save(filename+'_netParams.json')

def batchEvol(networkType, params=None):
    # parameters space to explore

    if networkType == 'simple':
        ## simple net
        params = specs.ODict()
        params['prob'] = [0.01, 0.5]
        params['weight'] = [0.001, 0.1]
        params['delay'] = [1, 20]

        pops = {}
        pops['S'] = {'target': 5, 'width': 2, 'min': 2}
        pops['M'] = {'target': 15, 'width': 2, 'min': 0.2}

    elif networkType == 'aw':
        const_net_params()

        if params is None:
            ## Thoughtful Params
            params = specs.ODict()

            # params['probEall'] = [0.2]  # Probability of excitatory-excitatory connections
            # params['weightEall'] = [0.0025]  # Weight of excitatory-excitatory connections
            # params['probIE'] = [0.4]  # Probability of inhibitory-excitatory connections
            # params['weightIE'] = [0.005*10]  # Weight of inhibitory-excitatory connections
            # params['stimWeight'] = [0.02]  # Weight of external stimulation

            # params['propVelocity'] = [100.0]  # Propagation velocity (μm/ms)

            # params['gnabar_E'] = [0.2]  # Maximum conductance of Na+ channels in excitatory neurons (S/cm^2)
            # params['gkbar_E'] = [0.05]  # Maximum conductance of K+ channels in excitatory neurons (S/cm^2)
            # params['gnabar_I'] = [0.15]  # Maximum conductance of Na+ channels in inhibitory neurons (S/cm^2)
            # params['gkbar_I'] = [0.05]  # Maximum conductance of K+ channels in inhibitory neurons (S/cm^2)        

            # params['tau1_exc'] = [0.8]  # Rise time constant of excitatory synaptic conductance (ms)
            # params['tau2_exc'] = [6.0]  # Decay time constant of excitatory synaptic conductance (ms)
            # params['tau1_inh'] = [0.8]  # Rise time constant of inhibitory synaptic conductance (ms)
            # params['tau2_inh'] = [9.0]  # Decay time constant of inhibitory synaptic conductance (ms)

            # params['stim_rate'] = [30*0.5]  # Stimulation rate (Hz)
            # params['stim_noise'] = [0.4]  # Stimulation noise

            ##
            
            # params['probEall'] = [0.2/100, 0.2/10, 0.2, 0.2*10, 0.2*100]  # Probability of excitatory-excitatory connections
            # params['weightEall'] = [0.0025/100, 0.0025/10, 0.0025, 0.0025*10, 0.0025*100]  # Weight of excitatory-excitatory connections
            # params['probIE'] = [0.4/100, 0.4/10, 0.4, 0.4*10, 0.4*100]  # Probability of inhibitory-excitatory connections
            # params['weightIE'] = [(0.005*10)/100, (0.005*10)/10, 0.005*10, (0.005*10)*10, (0.005*10)*100]  # Weight of inhibitory-excitatory connections
            # params['stimWeight'] = [0.02/100, 0.02/10, 0.02, 0.02*10, 0.02*100]  # Weight of external stimulation

            # params['propVelocity'] = [100.0/100, 100.0/10, 100.0, 100.0*10, 100.0*100]  # Propagation velocity (μm/ms)

            # params['gnabar_E'] = [0.2/100, 0.2/10, 0.2, 0.2*10, 0.2*100]  # Maximum conductance of Na+ channels in excitatory neurons (S/cm^2)
            # params['gkbar_E'] = [0.05/100, 0.05/10, 0.05, 0.05*10, 0.05*100]  # Maximum conductance of K+ channels in excitatory neurons (S/cm^2)
            # params['gnabar_I'] = [0.15/100, 0.15/10, 0.15, 0.15*10, 0.15*100]  # Maximum conductance of Na+ channels in inhibitory neurons (S/cm^2)
            # params['gkbar_I'] = [0.05/100, 0.05/10, 0.05, 0.05*10, 0.05*100]  # Maximum conductance of K+ channels in inhibitory neurons (S/cm^2)        

            # params['tau1_exc'] = [0.8/100, 0.8/10, 0.8, 0.8*10, 0.8*100]  # Rise time constant of excitatory synaptic conductance (ms)
            # params['tau2_exc'] = [6.0/100, 6.0/10, 6.0, 6.0*10, 6.0*100]  # Decay time constant of excitatory synaptic conductance (ms)
            # params['tau1_inh'] = [0.8/100, 0.8/10, 0.8, 0.8*10, 0.8*100]  # Rise time constant of inhibitory synaptic conductance (ms)
            # params['tau2_inh'] = [9.0/100, 9.0/10, 9.0, 9.0*10, 9.0*100]  # Decay time constant of inhibitory synaptic conductance (ms)

            # params['stim_rate'] = [(30*0.5)/100, (30*0.5)/10, 30*0.5, (30*0.5)*10, (30*0.5)*100]  # Stimulation rate (Hz)
            # params['stim_noise'] = [0.4/100, 0.4/10, 0.4, 0.4*10, 0.4*100]  # Stimulation noise

            ##

            # params['probEall'] = [0.2/100, 0.2, 0.2*100]  # Probability of excitatory-excitatory connections
            # params['weightEall'] = [0.0025/100, 0.0025, 0.0025*100]  # Weight of excitatory-excitatory connections
            # params['probIE'] = [0.4/100, 0.4, 0.4*100]  # Probability of inhibitory-excitatory connections
            # params['weightIE'] = [(0.005*10)/100, 0.005*10, (0.005*10)*100]  # Weight of inhibitory-excitatory connections
            # params['stimWeight'] = [0.02/100, 0.02, 0.02*100]  # Weight of external stimulation

            # params['propVelocity'] = [100.0/100, 100.0, 100.0*100]  # Propagation velocity (μm/ms)

            # params['gnabar_E'] = [0.2/100, 0.2, 0.2*100]  # Maximum conductance of Na+ channels in excitatory neurons (S/cm^2)
            # params['gkbar_E'] = [0.05/100, 0.05, 0.05*100]  # Maximum conductance of K+ channels in excitatory neurons (S/cm^2)
            # params['gnabar_I'] = [0.15/100, 0.15, 0.15*100]  # Maximum conductance of Na+ channels in inhibitory neurons (S/cm^2)
            # params['gkbar_I'] = [0.05/100, 0.05, 0.05*100]  # Maximum conductance of K+ channels in inhibitory neurons (S/cm^2)        

            # params['tau1_exc'] = [0.8/100, 0.8, 0.8*100]  # Rise time constant of excitatory synaptic conductance (ms)
            # params['tau2_exc'] = [6.0/100, 6.0, 6.0*100]  # Decay time constant of excitatory synaptic conductance (ms)
            # params['tau1_inh'] = [0.8/100, 0.8, 0.8*100]  # Rise time constant of inhibitory synaptic conductance (ms)
            # params['tau2_inh'] = [9.0/100, 9.0, 9.0*100]  # Decay time constant of inhibitory synaptic conductance (ms)

            # params['stim_rate'] = [(30*0.5)/100, 30*0.5, (30*0.5)*100]  # Stimulation rate (Hz)
            # params['stim_noise'] = [0.4/100, 0.4, 0.4*100]  # Stimulation noise

            ##

            # params['probEall'] = [0.2/100, 0.2, 0.2*100]  # Probability of excitatory-excitatory connections
            # params['weightEall'] = [0.0025/100, 0.0025, 0.0025*100]  # Weight of excitatory-excitatory connections
            # params['probIE'] = [0.4/100, 0.4, 0.4*100]  # Probability of inhibitory-excitatory connections
            # params['weightIE'] = [(0.005*10)/100, 0.005*10, (0.005*10)*100]  # Weight of inhibitory-excitatory connections
            # params['stimWeight'] = [0.02/100, 0.02, 0.02*100]  # Weight of external stimulation

            # params['propVelocity'] = [100.0/100, 100.0, 100.0*100]  # Propagation velocity (μm/ms)

            # params['gnabar_E'] = [0.5]  # Maximum conductance of Na+ channels in excitatory neurons (S/cm^2)
            # params['gkbar_E'] = [0.05]  # Maximum conductance of K+ channels in excitatory neurons (S/cm^2)
            # params['gnabar_I'] = [0.15]  # Maximum conductance of Na+ channels in inhibitory neurons (S/cm^2)
            # params['gkbar_I'] = [0.05]  # Maximum conductance of K+ channels in inhibitory neurons (S/cm^2)        

            # params['tau1_exc'] = [0.8/100, 0.8, 0.8*100]  # Rise time constant of excitatory synaptic conductance (ms)
            # params['tau2_exc'] = [6.0/100, 6.0, 6.0*100]  # Decay time constant of excitatory synaptic conductance (ms)
            # params['tau1_inh'] = [0.8/100, 0.8, 0.8*100]  # Rise time constant of inhibitory synaptic conductance (ms)
            # params['tau2_inh'] = [9.0/100, 9.0, 9.0*100]  # Decay time constant of inhibitory synaptic conductance (ms)

            # params['stim_rate'] = [(30*0.5)/100, 30*0.5, (30*0.5)*100]  # Stimulation rate (Hz)
            # params['stim_noise'] = [0.4/100, 0.4, 0.4*100]  # Stimulation noise

            ##

            # Probability of excitatory-excitatory connections
            #params['probEall'] = [0.2/100, 0.2/10, 0.2, 0.2*10, 0.2*100]  
            params['probEall'] = [0.2*100]  

            # Weight of excitatory-excitatory connections
            #params['weightEall'] = [0.0025/100, 0.0025/10, 0.0025, 0.0025*10, 0.0025*100]  
            params['weightEall'] = [0.0025]  

            # Probability of inhibitory-excitatory connections
            #params['probIE'] = [0.4/100, 0.4/10, 0.4, 0.4*10, 0.4*100]  
            params['probIE'] = [0.4*10]  

            # Weight of inhibitory-excitatory connections
            #params['weightIE'] = [(0.005*10)/100, (0.005*10)/10, 0.005*10, (0.005*10)*10, (0.005*10)*100]  
            params['weightIE'] = [0.005*10]  

            # Propagation velocity (μm/ms)
            #params['propVelocity'] = [100.0/100, 100.0/10, 100.0, 100.0*10, 100.0*100]  
            params['propVelocity'] = [100.0]
            
            # Maximum conductance of Na+ channels in excitatory neurons (S/cm^2)
            params['gnabar_E'] = [0.2/100, 0.2/10, 0.2, 0.2*10, 0.2*100]  
            #params['gnabar_E'] = [0.2]  

            # Maximum conductance of K+ channels in excitatory neurons (S/cm^2)
            #params['gkbar_E'] = [0.05/100, 0.05/10, 0.05, 0.05*10, 0.05*100]  
            params['gkbar_E'] = [0.05]  

            # Maximum conductance of Na+ channels in inhibitory neurons (S/cm^2)
            #params['gnabar_I'] = [0.15/100, 0.15/10, 0.15, 0.15*10, 0.15*100]  
            params['gnabar_I'] = [0.15]  

            # Maximum conductance of K+ channels in inhibitory neurons (S/cm^2)
            #params['gkbar_I'] = [0.05/100, 0.05/10, 0.05, 0.05*10, 0.05*100]  
            params['gkbar_I'] = [0.05]  

            # Rise time constant of excitatory synaptic conductance (ms)
            #params['tau1_exc'] = [0.8/100, 0.8/10, 0.8, 0.8*10, 0.8*100]  
            params['tau1_exc'] = [0.8]  

            # Decay time constant of excitatory synaptic conductance (ms)
            #params['tau2_exc'] = [6.0/100, 6.0/10, 6.0, 6.0*10, 6.0*100]  
            params['tau2_exc'] = [6.0]  

            # Rise time constant of inhibitory synaptic conductance (ms)
            #params['tau1_inh'] = [0.8/100, 0.8/10, 0.8, 0.8*10, 0.8*100]  
            params['tau1_inh'] = [0.8]  

            # Decay time constant of inhibitory synaptic conductance (ms)
            #params['tau2_inh'] = [9.0/100, 9.0/10, 9.0, 9.0*10, 9.0*100]  
            params['tau2_inh'] = [9.0]  
            
            # Weight of external stimulation
            #params['stimWeight'] = [0.02/100, 0.02/10, 0.02, 0.02*10, 0.02*100]  
            params['stimWeight'] = [0.02]             

            

            # Stimulation rate (Hz)
            #params['stim_rate'] = [(30*0.5)/100, (30*0.5)/10, 30*0.5, (30*0.5)*10, (30*0.5)*100]  
            params['stim_rate'] = [30*0.5]  

            # Stimulation noise
            #params['stim_noise'] = [0.4/100, 0.4/10, 0.4, 0.4*10, 0.4*100]  
            params['stim_noise'] = [0.4]  


    elif networkType == 'complex':
        # complex net
        params = specs.ODict()
        params['probEall'] = [0.05, 0.2] # 0.1
        params['weightEall'] = [0.0025, 0.0075] #5.0
        params['probIE'] = [0.2, 0.6] #0.4
        params['weightIE'] = [0.0005, 0.002]
        params['probLengthConst'] = [100,200]
        params['stimWeight'] = [0.05, 0.2]

        pops = {}
        pops['E2'] = {'target': 5, 'width': 2, 'min': 1}
        pops['I2'] = {'target': 10, 'width': 5, 'min': 2}
        pops['E4'] = {'target': 30, 'width': 10, 'min': 1}
        pops['I4'] = {'target': 10, 'width': 3, 'min': 2}
        pops['E5'] = {'target': 40, 'width': 4, 'min': 1}
        pops['I5'] = {'target': 25, 'width': 5, 'min': 2}

	# # fitness function
	# fitnessFuncArgs = {}
	# fitnessFuncArgs['pops'] = pops
	# fitnessFuncArgs['maxFitness'] = 1000

	# def fitnessFunc(simData, **kwargs):
	# 	import numpy as np
	# 	pops = kwargs['pops']
	# 	maxFitness = kwargs['maxFitness']
	# 	popFitness = [None for i in pops.items()]
	# 	popFitness = [min(np.exp(  abs(v['target'] - simData['popRates'][k])  /  v['width']), maxFitness)
	# 			if simData["popRates"][k]>v['min'] else maxFitness for k,v in pops.items()]
	# 	fitness = np.mean(popFitness)
	# 	popInfo = '; '.join(['%s rate=%.1f fit=%1.f'%(p,r,f) for p,r,f in zip(list(simData['popRates'].keys()), list(simData['popRates'].values()), popFitness)])
	# 	print('  '+popInfo)
	# 	return fitness

	# create Batch object with paramaters to modify, and specifying files to use
    batch = Batch(params=params)

	# Set output folder, grid method (all param combinations), and run configuration
    #batch.batchLabel = 'simple'
    batch.batchLabel = 'aw_grid'	
    batch.saveFolder = './' + batch.batchLabel
    #batch.method = 'evol'
    #batch.netParams = netParams
    #batch.netParamsFile = 'netParams.py'
    batch.method = 'grid'
    batch.runCfg = {
        'type': 'mpi_bulletin',#'hpc_slurm',
        'script': 'init.py',
        # options required only for hpc
        'mpiCommand': 'mpirun',
        'nodes': 4,
        'coresPerNode': 2,
        'allocation': 'default',
        #'email': 'salvadordura@gmail.com',
        'reservation': None,
        'skip': True,
        #'folder': '/home/salvadord/evol'
        #'custom': 'export LD_LIBRARY_PATH="$HOME/.openmpi/lib"' # only for conda users
    }
    # batch.evolCfg = {
    # 	'evolAlgorithm': 'custom',
    # 	'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
    # 	'fitnessFuncArgs': fitnessFuncArgs,
    # 	'pop_size': 6,
    # 	'num_elites': 1, # keep this number of parents for next generation if they are fitter than children
    # 	'mutation_rate': 0.4,
    # 	'crossover': 0.5,
    # 	'maximize': False, # maximize fitness function?
    # 	'max_generations': 4,
    # 	'time_sleep': 5, # wait this time before checking again if sim is completed (for each generation)
    # 	'maxiter_wait': 40, # max number of times to check if sim is completed (for each generation)
    # 	'defaultFitness': 1000 # set fitness value in case simulation time is over
    # }
    batch.run()

# Main code
if __name__ == '__main__':
	#batchEvol('simple')  # 'simple' or 'complex'
    batchEvol('aw')  # 'simple' or 'complex'    
