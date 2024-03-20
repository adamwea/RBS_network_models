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
                with open(f'{output_path}/params.pickle', 'rb') as handle:
                    params = pickle.load(handle)

                ## Create folder based on batch.filename
                #filename = params['filename'][0]                
                # batchgen_dir = f'{output_path}/{filename}'
                # if os.path.exists(batchgen_dir) == False:
                #     os.makedirs(batchgen_dir)
                
                #move params.pickle and params.json to batchgen_dir
                assert os.path.exists(f'{output_path}/params.pickle')
                assert os.path.exists(f'{output_path}/params.json')
                # shutil.copy(f'{output_path}/params.pickle', f'{batchgen_dir}/params.pickle')
                # shutil.copy(f'{output_path}/params.json', f'{batchgen_dir}/params.json')

                pops = {}

                #firing rate targets
                pops['rate_targets'] = {}
                pops['rate_targets']['E'] = {'target': 7.5, 'width': 2.5, 'min': 1}
                pops['rate_targets']['I'] = {'target': 30, 'width': 10, 'min': 2}

                #burst peak targets
                pops['burts_peak_targets'] = {'target': 15, 'width': 10, 'min': 1}

                #burst IBI targets
                pops['IBI_targets'] = {'target': 3000, 'width': 2000 , 'min': 1000} #ms

                #baseline targets
                pops['baseline_targets'] = {'target': 1.5, 'width': 1.5 , 'min': 0} #ms

                #firing rate variance targets
                pops['rate_variance'] = {'target': 0, 'width': 1, 'min': 0}

            except:
                print('Error loading params from file')
                print('Using default params')
                raise Exception('Error loading params from file')  

	# fitness function
    fitnessFuncArgs = {}
    fitnessFuncArgs['pops'] = pops
    fitnessFuncArgs['maxFitness'] = 1000

    def fitnessFunc(simData, **kwargs):
        #from netpyne import sim
        import numpy as np
        import sys
        sys.path.insert(0, '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5_optimizing/batch_run_files')
        from aw_batch_tools import measure_network_activity
        from netpyne import sim
        #rasterData = sim.analysis.prepareRaster()
        net_activity_params = {'binSize': .03*500, 'gaussianSigma': .12*500, 'thresholdBurst': 1.0}
        binSize = net_activity_params['binSize']
        gaussianSigma = net_activity_params['gaussianSigma']
        thresholdBurst = net_activity_params['thresholdBurst']

        #
        
        try:
            #prepare raster data
            #rasterData = sim.analysis.prepareRaster()
            rasterData = simData
            if len(rasterData['spkt']) == 0:
                print('No spikes found. Setting fitness to 1000.0.')
                return 1000
            # Generate the network activity plot with a size of (10, 5)
            net_metrics = measure_network_activity(
                rasterData, 
                binSize=binSize, 
                gaussianSigma=gaussianSigma, 
                thresholdBurst=thresholdBurst, 
                #figSize=(network_activity_width, network_activity_height), 
                #saveFig=network_activity_path
            )
            burstPeakValues = net_metrics['burstPeakValues']
            burstPeakTimes = net_metrics['burstPeakTimes'] #not needed
            IBIs = net_metrics['IBIs']
            #firingRate = net_metrics['firingRate'] #solved using other method
            #timeVector = net_metrics['timeVector'] #not needed
            baseline = net_metrics['baseline']            

            # get mean and std of firing rate, make sure it stays within 2.5 std of mean
            # try:
            #     #firingRate fitness
            #     pops = kwargs['pops']
            #     rate_variance = pops['rate_variance']
            #     maxFitness = kwargs['maxFitness']
            #     #popFitness = [None for i in pops.items()]
            #     firingRate_mean = np.mean(firingRate)
            #     firingRate_std = np.std(firingRate)
            #     firingRate_var = np.var(firingRate)
            #     firingRate_std_mean = firingRate_std / firingRate_mean

            #     # Coefficient of variation
            #     firingRate_cv = firingRate_std / firingRate_mean if firingRate_mean != 0 else 0

            #     # Calculate the fitness for each value in the firing rate
            #     varFitness = [
            #         min(abs(rate_variance['target'] - value) / rate_variance['width'], maxFitness)
            #         if value > rate_variance['min'] else maxFitness for value in firingRate
            #     ]

            #     # Penalize high variability
            #     varFitness = [fitness * (1 - firingRate_cv) for fitness in varFitness]


            #     #print average firing rate and its fitness
            #     print('Average firing rate: %.1f, Fitness: %.1f' % (firingRate_mean, np.mean(varFitness)))
            # except Exception as e:
            #     print(f'Error calculating firing rate variance fitness.')
            #     print(f'Error: {e}')
            #     #set fitness values to maxFitness
            #     varFitness = maxFitness
            #     pass  
            
            # Calculate the fitness for each value in burstPeakValues
            try: 
                #burstPeakValue Fitness
                pops = kwargs['pops']
                pops_peaks = pops['burts_peak_targets']
                print(kwargs)
                maxFitness = kwargs['maxFitness']
                #popFitnessBurstPeak = [None for i in pops.items()]
                assert len(burstPeakValues) > 0, 'Error: burstPeakValues has no elements. BurstVal_fitness set to maxFitness.'
                popFitnessBurstPeak = [
                    min(np.exp(abs(pops_peaks['target'] - value) / pops_peaks['width']), maxFitness)
                    if value > pops_peaks['min'] else maxFitness for value in burstPeakValues
                ]

                # Calculate the mean fitness
                BurstVal_fitness = np.mean(popFitnessBurstPeak)

                # Calculate the average burstPeak
                avg_burstPeak = np.mean(burstPeakValues)

                # Print the average burstPeak and its fitness
                print('Average burstPeak: %.1f, Fitness: %.1f' % (avg_burstPeak, BurstVal_fitness))
            except Exception as e:
                print(f'Error calculating burst peak fitness.')
                print(f'Error: {e}')
                #set fitness values to maxFitness
                BurstVal_fitness = maxFitness
                pass

            # IBI Fitness
            try:
                assert len(burstPeakTimes) > 0, 'Error: burstPeakTimes has less than 1 element. IBI could be calculated.'
                assert len(burstPeakValues) > 1, 'Error: burstPeakValues has less than 2 elements. IBI could be calculated.'
                assert len(IBIs) > 0, 'Error: IBIs has no elements. IBI could not be calculated.'
                pops = kwargs['pops']
                pops_IBI = pops['IBI_targets']
                maxFitness = kwargs['maxFitness']

                popFitnessIBI = [
                    min(np.exp(abs(pops_IBI['target'] - value) / pops_IBI['width']), maxFitness)
                    if value > pops_IBI['min'] else maxFitness for value in IBIs
                ]
                IBI_fitness = np.mean(popFitnessIBI)

                #Calc average IBI
                avg_IBI = np.mean(IBIs)

                #print average IBI and its fitness
                print('Average IBI: %.1f, Fitness: %.1f' % (avg_IBI, IBI_fitness))
            except Exception as e:
                print(f'Error calculating IBI fitness.')
                print(f'Error: {e}')
                #set fitness values to maxFitness
                IBI_fitness = maxFitness
                pass

            # Baseline Fitness
            try:
                assert baseline is not None, 'Error: baseline is None. Baseline could not be calculated.'
                #assert len(baseline) > 0, 'Error: baseline has no elements. Baseline could not be calculated.'
                pops = kwargs['pops']
                pops_baseline = pops['baseline_targets']
                maxFitness = kwargs['maxFitness']
                baseline_target = pops_baseline['target']
                baseline_width = pops_baseline['width']
                baseline_min = pops_baseline['min']

                baselineFitness = min(np.exp(abs(baseline_target - baseline) / baseline_width), maxFitness) if baseline > baseline_min else maxFitness

                #print average baseline and its fitness
                print('baseline: %.1f, Fitness: %.1f' % (baseline, baselineFitness)) 
            except:
                print(f'Error calculating baseline fitness.')
                print(f'Error: {e}')
                #set fitness values to maxFitness
                baselineFitness = maxFitness
                pass      

        except Exception as e:
            print(f'Error calculating network activity metrics. Check if rasterData is available.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            BurstVal_fitness = maxFitness
            IBI_fitness = maxFitness
            baselineFitness = maxFitness
            pass

        #Firing Rate Fitness
        try:
            pops = kwargs['pops']
            pops_rate = pops['rate_targets']
            maxFitness = kwargs['maxFitness']
            popFitness = [None for i in pops.items()]
            #simData = sim.allSimData
            # This is a list comprehension that calculates the fitness for each population in the simulation.
            popFitness = [
                # The fitness for a population is calculated using an exponential function.
                # The argument to the exponential function is the absolute difference between the target firing rate and the actual firing rate,
                # divided by the width parameter. This value represents how far the actual firing rate is from the target, scaled by the width.
                # The np.exp() function then transforms this value into a range between 0 (if the actual rate equals the target) and infinity (as the actual rate deviates from the target).
                # The min() function is used to limit the maximum fitness value to 'maxFitness'.
                min(np.exp(abs(v['target'] - simData['popRates'][k]) / v['width']), maxFitness)
                # The if-else statement checks if the actual firing rate is greater than the minimum acceptable firing rate.
                # If the actual firing rate is less than the minimum, the fitness is set to 'maxFitness', which represents the worst possible fitness.
                if simData["popRates"][k] > v['min'] else maxFitness
                # The for loop iterates over each population in the 'pops' dictionary.
                # 'k' is the key (population name) and 'v' is the value (a dictionary containing the target, width, and minimum firing rate for that population).
                for k, v in pops_rate.items()
            ]
            rate_fitness = np.mean(popFitness)
            popInfo = '; '.join(['%s rate=%.1f fit=%1.f'%(p,r,f) for p,r,f in zip(list(simData['popRates'].keys()), list(simData['popRates'].values()), popFitness)])
            print('  '+popInfo)
        except Exception as e:
            print(f'Error calculating firing rate fitness.')
            #print(f'Error: {e}')
            #set fitness values to maxFitness
            rate_fitness = maxFitness
            pass

        #average fitness
        fitness = (rate_fitness + BurstVal_fitness + IBI_fitness + baselineFitness) / 4
        return fitness

	# create Batch object with paramaters to modify, and specifying files to use
    batch = Batch(params=params)

	# Set output folder, grid method (all param combinations), and run configuration
    #batch.batchLabel = 'simple'
    #start with zero, check if folder exists, if it does, increment by 1
    #print(params['filename'][0])
    #batch_run_num = 0
    #batch.batchLabel = params['filename'][0]
    #batch.saveFolder = f"{output_path}/{params['filename'][0]}/"
    batch.saveFolder = f"{output_path}/"

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
    batch.evolCfg = {
    	'evolAlgorithm': 'custom',
    	'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
    	'fitnessFuncArgs': fitnessFuncArgs,
    	'pop_size': 120, 
    	'num_elites': 1, # keep this number of parents for next generation if they are fitter than children
    	'mutation_rate': 0.4,
    	'crossover': 0.5,
    	'maximize': False, # maximize fitness function?
    	'max_generations': 5,
    	'time_sleep': 5, # wait this time before checking again if sim is completed (for each generation)
    	'maxiter_wait': 40, # max number of times to check if sim is completed (for each generation)
    	'defaultFitness': 1000 # set fitness value in case simulation time is over
    }
    batch.run()

# Main code
if __name__ == '__main__':
	#batchEvol('simple')  # 'simple' or 'complex'
    batchRun(
        batchLabel = 'batchRun_testing', 
        #method = 'grid', 
        method = 'evol',
        params=None, 
        skip = True
        ) 
