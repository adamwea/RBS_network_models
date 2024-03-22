# This file is a modified version of the file batchRun.py from the NetPyNE tutorial 9.

# General Imports
import os
import shutil
import json
import pickle
import glob
import sys

# NetPyne Imports
sys.path.insert(0, '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/netpyne')
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
                pops['rate_slope'] = {'target': 0, 'width': 0.5, 'min': 0}

            except:
                print('Error loading params from file')
                print('Using default params')
                raise Exception('Error loading params from file')  

	# Load the dictionary from the JSON file
    with open('batch_config.json', 'r') as f:
        batch_config = json.load(f)
    
    #Global Counters
    # import multiprocessing import Manager, Process
    # global global_pop_val
    # global_pop_val = batch_config['evolCfg']['pop_size']
    
    # global global_gen_counter
    # global_gen_counter = 0
    # global global_cand_counter
    # global_cand_counter = 0
    # global global_batch_counter
    # global_batch_counter = 0
    # manager = Manager()
    # batch_counter = manager.Value('i', 0)
    # gen_counter = manager.Value('i', 0)
    # cand_counter = manager.Value('i', 0)
    # shared_counters = (batch_counter, gen_counter, cand_counter)

    # fitness function
    fitnessFuncArgs = {}
    fitnessFuncArgs['pops'] = pops
    fitnessFuncArgs['maxFitness'] = 1000
    # fitnessFuncArgs['shared_counters'] = shared_counters

    def fitnessFunc(simData, **kwargs):
        ##find relevant batch object in call stack
        # (I dont understand how this works, figure it out later)
        import inspect
        def find_batch_object(frame):
            if frame is None:
                return None

            if 'self' in frame.f_locals and isinstance(frame.f_locals['self'], Batch):
                return frame.f_locals['self']

            #print relative object path
            #print(frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno)
            return find_batch_object(frame.f_back)
        current_frame = inspect.currentframe()
        # Get the caller's frame (i.e., the frame of the batchRun function)
        caller_frame = inspect.getouterframes(current_frame, 3)[1][0]
        # Search recursively through caller frames for the Batch object
        batch = None
        #batch = find_batch_object(caller_frame)
        try: simLabel = caller_frame.f_locals['_']
        except: simLabel = None
        #extract simLabel from batch object
        if batch is not None: simLabel = batch.cfg.simLabel
        elif simLabel is None: print("SimLabel not found in the caller frames.")
        if batch is None and simLabel is None: print("Batch object not found in the caller frames.")

        # #print counters
        # print(f'Calculating fitness:')
        # print(f'Batch: {global_batch_counter}, Generation: {global_gen_counter}, Candidate: {global_cand_counter}')
        
        #from netpyne import sim
        import numpy as np
        import sys
        sys.path.insert(0, '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5_optimizing/batch_run_files')
        from aw_batch_tools import measure_network_activity
        #from netpyne import sim
        from scipy.stats import linregress
        #rasterData = sim.analysis.prepareRaster()
        maxFitness = kwargs['maxFitness']
        net_activity_params = {'binSize': .03*500, 'gaussianSigma': .12*500, 'thresholdBurst': 1.0}
        binSize = net_activity_params['binSize']
        gaussianSigma = net_activity_params['gaussianSigma']
        thresholdBurst = net_activity_params['thresholdBurst']
        output_path = '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5_optimizing/output'

        #
        
        try:
            #prepare raster data
            #rasterData = sim.analysis.prepareRaster()
            rasterData = simData
            maxFitness = kwargs['maxFitness']
            assert len(rasterData['spkt']) > 0, 'Error: rasterData has no elements. burstPeak, baseline, slopeFitness, and IBI fitness set to 1000.'                       
            
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

            # Get mean and std of firing rate, make sure it stays within 2.5 std of mean
            try:
                # Firing rate fitness
                pops = kwargs['pops']
                rate_slope= pops['rate_slope']
                maxFitness = kwargs['maxFitness']

                # Get the firing rate from the network metrics
                firingRate = net_metrics['firingRate']

                # Calculate the trendline of the firing rate
                slope, intercept, r_value, p_value, std_err = linregress(range(len(firingRate)), firingRate)

                # Calculate the fitness as the absolute difference between the slope and the target slope
                slopeFitness = abs(rate_slope['target'] - slope)/rate_slope['width'] if slope > rate_slope['min'] else maxFitness

                # Print the slope and its fitness
                print('Slope of firing rate: %.3f, Fitness: %.3f' % (slope, slopeFitness))
            except Exception as e:
                print(f'Error calculating firing rate slope fitness.')
                print(f'Error: {e}')
                # Set fitness values to maxFitness
                slopeFitness = maxFitness 
            
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
                print('Average burstPeak: %.3f, Fitness: %.3f' % (avg_burstPeak, BurstVal_fitness))
            except Exception as e:
                print(f'Error calculating burst peak fitness.')
                print(f'Error: {e}')
                #set fitness values to maxFitness
                maxFitness = kwargs['maxFitness']
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
                print('Average IBI: %.3f, Fitness: %.3f' % (avg_IBI, IBI_fitness))
            except Exception as e:
                print(f'Error calculating IBI fitness.')
                print(f'Error: {e}')
                #set fitness values to maxFitness
                maxFitness = kwargs['maxFitness']
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
                print('baseline: %.3f, Fitness: %.3f' % (baseline, baselineFitness)) 
            except:
                print(f'Error calculating baseline fitness.')
                print(f'Error: {e}')
                #set fitness values to maxFitness
                maxFitness = kwargs['maxFitness']
                baselineFitness = maxFitness
                pass      

        except Exception as e:
            print(f'Error calculating network activity metrics.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            BurstVal_fitness = maxFitness
            IBI_fitness = maxFitness
            baselineFitness = maxFitness
            slopeFitness = maxFitness
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
            maxFitness = kwargs['maxFitness']
            rate_fitness = maxFitness
            pass

        
        #average fitness
        fitness = (rate_fitness + BurstVal_fitness + IBI_fitness + baselineFitness + slopeFitness) / 5

        #save fitness values
        fitnessResults = {}
        fitnessResults['rate_fitness'] = rate_fitness
        fitnessResults['BurstVal_fitness'] = BurstVal_fitness
        fitnessResults['IBI_fitness'] = IBI_fitness
        fitnessResults['baselineFitness'] = baselineFitness
        fitnessResults['slopeFitness'] = slopeFitness
        fitnessResults['fitness'] = fitness
        fitnessResults['maxFitness'] = maxFitness

        output_path = '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5_optimizing/output'
        #Generate figures for good candidates
        # if fitness < 400:
        #     try:
        #         from aw_batch_tools import generate_all_figures
        #         import shutil
        #         gen_folder = simLabel.split('_cand')[0]
        #         fresh_figs = False
        #         output_path = '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5_optimizing/output'
        #         batchdata_path = f'{output_path}/{gen_folder}/{simLabel}_data.json'
        #         assert os.path.exists(batchdata_path), f'{batchdata_path} does not exist'
        #         cfg_path = batchdata_path.replace('_data.json', '_cfg.json')
        #         assert os.path.exists(cfg_path), f'{cfg_path} does not exist'
        #         fitness_path = batchdata_path.replace('_data.json', '_Fitness.json')
        #         assert os.path.exists(fitness_path), f'{fitness_path} does not exist'
        #         generate_all_figures(
        #             fresh_figs = fresh_figs,
        #             net_activity_params = {'binSize': .03*500, 
        #                                 'gaussianSigma': .12*500, 
        #                                 'thresholdBurst': 1.0},
        #             batchLabel = 'batchRun_evol',
        #             #batchLabel = params['filename'][0],
        #             #minimum peak distance = 0.5 seconds
        #             batch_path = batchdata_path
        #         )
        #         output_path = batchdata_path
        #         output_path = os.path.dirname(output_path)
        #         output_path = f'{output_path}/NetworkBurst_and_Raster_Figs/'
        #         #run_grand_path = os.path.dirname(run_path)
        #         run_grand_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(output_path))))
        #         plot_row_path = f'{run_grand_path}/goodFit_plots_rows/'
        #         if not os.path.exists(plot_row_path):
        #             os.makedirs(plot_row_path)
        #         ## shutil copy any files in output_path with the string 'row' in the name to plot path
        #         for rooti, dirsi, filesi in os.walk(output_path):
        #             if '.archive' in rooti: continue
                    
        #             for filei in filesi:
        #                 if 'row' in filei:
        #                     if os.path.exists(plot_row_path+filei):
        #                         if fresh_figs: os.remove(plot_row_path+filei)
        #                         else: continue
        #                     shutil.copy(rooti+'/'+filei, plot_row_path+filei)
        #     except Exception as e:
        #         print(f'Error generating figures for good candidates.')
        #         print(f'Error: {e}')    
        #         pass

        #save fitness results to file
        import json
        #get folder from simLabel
        gen_folder = simLabel.split('_cand')[0]
        with open(f'{output_path}/{gen_folder}/{simLabel}_Fitness.json', 'w') as f:
            json.dump(fitnessResults, f)
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

    # Now you can access the values in the dictionary
    #print(batch_config)
    
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
    	#'fitnessFuncArgs': fitnessFuncArgs,
        'fitnessFuncArgs': {**fitnessFuncArgs, 'simLabel': 'batch.cfg.simLabel'},
    	'pop_size': batch_config['evolCfg']['pop_size'], # population size
    	'num_elites': 1, # keep this number of parents for next generation if they are fitter than children
    	'mutation_rate': 0.4,
    	'crossover': 0.5,
    	'maximize': False, # maximize fitness function?
    	'max_generations': batch_config['evolCfg']['max_generations'], # how many generations to run
    	'time_sleep': 30, # wait this time before checking again if sim is completed (for each generation)
    	#'maxiter_wait': 4000000000, # max number of times to check if sim is completed (for each generation)
        'maxiter_wait': 300000, # max number of times to check if sim is completed (for each generation)
    	'defaultFitness': 1000, # set fitness value in case simulation time is over
        #'gen': 'gen',
        #'cand': 'cand',
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
