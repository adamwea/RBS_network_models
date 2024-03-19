# This file is a modified version of the file batchRun.py from the NetPyNE tutorial 9.

# General Imports
import os
import shutil
import json
import pickle
import glob

# NetPyne Imports
from netpyne import specs
from netpyne.batch import Batch

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
print(f'Script directory: {script_dir}')

# Change the working directory to the script directory
os.chdir(script_dir)

print(f'Changed working directory to: {script_dir}')
 #get current working directory
output_path = os.path.dirname(script_dir)
output_path = f'{output_path}/output' 
print(f'Output path: {output_path}')

''' Example of evolutionary algorithm optimization of a network using NetPyNE
To run use: mpiexec -np [num_cores] nrniv -mpi batchRun.py
'''

# if folder "aw_grid" exists, delete it
# if os.path.exists('aw_grid'):
#     shutil.rmtree('aw_grid')

def batchRun(batchLabel = 'batchRun', method = 'grid', params=None, skip = False):
    # parameters space to explore
    # const_net_params()         
         
    if params is None:
        # Load params from file, if it exists
        if os.path.exists(f'{output_path}/params.json'):
            try:
                # # Load params from file, json
                # with open('params.json', 'r') as f:
                #     params = json.load(f)

                # To load the OrderedDict
                  
                # To save the OrderedDict                
                #output_path = ('/home/adam/workspace/git_workspace/netpyne/hdmea_simulations/BurstingPlotDevelopment/nb3/output')
                
                ## backout of current directory
              

                # if os.getcwd() != output_path:
                #      os.chdir(output_path)
                #param_dir = f'./output/'
                # if os.path.exists(param_dir) == False:
                #     os.makedirs(param_dir)
                # with open(f'{output_path}/params.pickle', 'wb') as handle:
                #     pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(f'{output_path}/params.pickle', 'rb') as handle:
                    params = pickle.load(handle)

                ## Create folder based on batch.filename
                filename = params['filename'][0]                
                batchgen_dir = f'{output_path}/{filename}'
                if os.path.exists(batchgen_dir) == False:
                    os.makedirs(batchgen_dir)
                
                #move params.pickle and params.json to batchgen_dir
                assert os.path.exists(f'{output_path}/params.pickle')
                assert os.path.exists(f'{output_path}/params.json')
                shutil.copy(f'{output_path}/params.pickle', f'{batchgen_dir}/params.pickle')
                shutil.copy(f'{output_path}/params.json', f'{batchgen_dir}/params.json')

                # change working directory to batchgen_dir
                #os.chdir(batchgen_dir)
            except:
                print('Error loading params from file')
                print('Using default params')  

                ## Thoughtful Params
                params = specs.ODict()          

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
                #params['gnabar_E'] = [0.2/100, 0.2/10, 0.2, 0.2*10, 0.2*100]  
                params['gnabar_E'] = [0.2]  

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
    #start with zero, check if folder exists, if it does, increment by 1
    print(params['filename'][0])
    batch_run_num = 0
    batch.batchLabel = params['filename'][0]
    batch.saveFolder = f"{output_path}/{params['filename'][0]}/"
    # batch_run_path = f"{batch.saveFolder}/{batch.batchLabel}*"
    # while glob.glob(batch_run_path):
    #     batch_run_num += 1
    #     batch.batchLabel = f'{batchLabel}{batch_run_num}'
    #     batch_run_path = f"{batch.saveFolder}/{batch.batchLabel}*"
    # #     batch.saveFolder = f"{output_path}/{params['filename'][0]}/" + batch.batchLabel

    batch.method = method
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
        'skip': skip,
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
    batchRun(
        batchLabel = 'batchRun', 
        method = 'grid', 
        params=None, 
        skip = True
        ) 
