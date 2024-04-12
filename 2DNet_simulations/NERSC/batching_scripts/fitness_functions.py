#from netpyne import sim
import numpy as np
import sys
#sys.path.insert(0, '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5_optimizing/batch_run_files')
from batch_run_tools import measure_network_activity
#from netpyne import sim
from scipy.stats import linregress

def fitnessFunc(simData, **kwargs):
    
    '''
    Functions
    '''
    # (I dont understand how this works, figure it out later)
    import inspect
    def find_batch_object_and_sim_label():
        # Get the current frame
        current_frame = inspect.currentframe()  
        # Iterate through the call stack
        while current_frame:
            caller_frame = inspect.getouterframes(current_frame, 3)#[1][0]
            # Check each local variable in the frame
            #for name, obj in list(current_frame.f_locals.items())[::-1]:
            for name, obj in list(caller_frame[1][0].f_locals.items())[::-1]:
                # If the object is of type Batch, return it and its simLabel
                if name == '_':
                    simLabel = caller_frame[1][0].f_locals['_']
                    batch = caller_frame[2][0].f_locals['batch']
                    return batch, simLabel
            # If not, move to the next frame
            current_frame = current_frame.f_back
        # If no Batch object is found, print a message and return None
        print("Batch object not found in the caller frames.")
        return None, None
    
    '''
    fitnessFunc
    '''
    ## import plotting functions
    # try: from batch_run_files.plot_config import plot_sim_figs
    # except: from plot_config import plot_sim_figs    
    
    ##find relevant batch object in call stack
    # Use the function to get the Batch object and simLabel
    batch, simLabel = find_batch_object_and_sim_label()
    # If no simLabel is found, print a message
    assert simLabel is not None, "SimLabel not found in the caller frames."
    assert batch is not None, "Batch object not found in the caller frames."    

    ## Get the fitness function argumentsS
    #rasterData = sim.analysis.prepareRaster()
    maxFitness = kwargs['maxFitness']
    net_activity_params = {'binSize': .03*1000, 'gaussianSigma': .12*1000, 'thresholdBurst': 1.0}
    binSize = net_activity_params['binSize']
    gaussianSigma = net_activity_params['gaussianSigma']
    thresholdBurst = net_activity_params['thresholdBurst']
    #output_path = '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5_optimizing/output'
    output_path = batch.saveFolder

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
        baseline_diff = net_metrics['baseline_diff']
        burst_peak_variance = net_metrics['normalized_peak_variance']
        burst_peak_frequency = net_metrics['peak_freq']

        # ##
        # try:
        #     #burst_peak_variance Fitness
        #     pops = kwargs['pops']
        #     pops_variance = pops['burst_peak_variance']
        #     maxFitness = kwargs['maxFitness']

        #     # Calculate the fitness as the absolute difference between the variance and the target variance
        #     burst_peak_variance_fitness = min(np.exp(abs(pops_variance['target'] - burst_peak_variance) / pops_variance['width']), maxFitness) if burst_peak_variance > pops_variance['min'] else maxFitness

        #     # Print the variance and its fitness
        #     print('Variance of burst peaks: %.3f, Fitness: %.3f' % (burst_peak_variance, burst_peak_variance_fitness))
        # except Exception as e:
        #     print(f'Error calculating burst peak variance fitness.')
        #     print(f'Error: {e}')
        #     # Set fitness values to maxFitness
        #     burst_peak_variance_fitness = maxFitness

        ##
        try:
            #burst_peak_frequency Fitness
            pops = kwargs['pops']
            pops_frequency = pops['burst_peak_frequency']
            maxFitness = kwargs['maxFitness']

            # Calculate the fitness as the absolute difference between the frequency and the target frequency
            burst_peak_frequency_fitness = min(np.exp(abs(pops_frequency['target'] - burst_peak_frequency) / pops_frequency['width']), maxFitness) if burst_peak_frequency > pops_frequency['min'] else maxFitness

            # Print the frequency and its fitness
            print('Frequency of burst peaks: %.3f, Fitness: %.3f' % (burst_peak_frequency, burst_peak_frequency_fitness))
        except Exception as e:
            print(f'Error calculating burst peak frequency fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            burst_peak_frequency_fitness = maxFitness

        
        ##
        try:
            #baseline_diff Fitness
            pops = kwargs['pops']
            baseline_diff_target = pops['baseline_diff']
            maxFitness = kwargs['maxFitness']

            # Calculate the fitness as the absolute difference between the baseline and the target baseline
            baseline_diff_fitness = min(np.exp(baseline_diff_target['target'] - baseline_diff) / baseline_diff_target['width'], maxFitness) if baseline_diff > baseline_diff_target['min'] else maxFitness

            # Print the baseline_diff and its fitness
            print('Difference between baseline and threshold: %.3f, Fitness: %.3f' % (baseline_diff, baseline_diff_fitness))
        except Exception as e:
            print(f'Error calculating baseline_diff fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            baseline_diff_fitness = maxFitness

        ##
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
            slopeFitness = min(np.exp(abs(rate_slope['target'] - slope)/rate_slope['width']) if abs(slope) < rate_slope['max'] else maxFitness,
                                 maxFitness)

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
                if value < pops_IBI['max'] else maxFitness for value in IBIs
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
            baseline_max = pops_baseline['max']

            baselineFitness = min(np.exp(abs(baseline_target - baseline) / baseline_width), maxFitness) if baseline < baseline_max else maxFitness

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
        baseline_diff_fitness = maxFitness
        burst_peak_frequency_fitness = maxFitness
        pass

    #Firing Rate Fitness
    try:
        pops = kwargs['pops']
        pops_rate = pops['rate_targets']
        maxFitness = kwargs['maxFitness']
        popFitness = [None for i in pops.items()]
        # This is a list comprehension that calculates the fitness for each population in the simulation.
        popFitness = [
            min(np.exp(abs(v['target'] - simData['popRates'][k]) / v['width']), maxFitness)
            if simData["popRates"][k] > v['min'] else maxFitness
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
    fitness = (rate_fitness + BurstVal_fitness + IBI_fitness + baselineFitness + slopeFitness + baseline_diff_fitness + burst_peak_frequency_fitness) / 7

    ##normalized and scaled average fitness - mitigate the effect of outliers
    # Assuming fitness values are stored in a list
    fitness_values = [rate_fitness, BurstVal_fitness, IBI_fitness, baselineFitness, slopeFitness, baseline_diff_fitness, burst_peak_frequency_fitness]
    #print(f'Average fitness values: {fitness_values}')

    # Calculate min and max values
    min_value = min(fitness_values)
    max_value = max(fitness_values)

    # Normalize the fitness values
    if max_value > min_value:
        normalized_fitness_values = [(value - min_value) / (max_value - min_value) for value in fitness_values]
        if np.isnan(normalized_fitness_values[0]):
            #set all values to 1
            normalized_fitness_values = [1, 1, 1, 1, 1, 1, 1]
            #normalized_fitness_values[0] = 1
        # Scale the normalized values back to the original range (0-1000)
        scaled_fitness_values = [value * 1000 for value in normalized_fitness_values]
    else:
        scaled_fitness_values = [1000, 1000, 1000, 1000, 1000, 1000, 1000] 

    # Calculate the average of the scaled fitness values
    average_scaled_fitness = sum(scaled_fitness_values) / len(scaled_fitness_values)
    #print(f'Average scaled fitness values: {scaled_fitness_values}')

    #save fitness values
    fitnessResults = {}
    fitnessResults['rate_fitness'] = rate_fitness
    fitnessResults['BurstVal_fitness'] = BurstVal_fitness
    fitnessResults['IBI_fitness'] = IBI_fitness
    fitnessResults['baselineFitness'] = baselineFitness
    fitnessResults['slopeFitness'] = slopeFitness
    fitnessResults['baseline_diff_fitness'] = baseline_diff_fitness
    #fitnessResults['burst_peak_variance'] = burst_peak_variance
    fitnessResults['burst_peak_frequency'] = burst_peak_frequency_fitness
    fitnessResults['fitness'] = fitness
    fitnessResults['average_scaled_fitness'] = average_scaled_fitness
    fitnessResults['maxFitness'] = maxFitness
    #print fitness results on multiple lines
    print(f'rate_fitness: {rate_fitness}')
    print(f'BurstVal_fitness: {BurstVal_fitness}')
    print(f'IBI_fitness: {IBI_fitness}')
    print(f'baselineFitness: {baselineFitness}')
    print(f'slopeFitness: {slopeFitness}')
    print(f'baseline_diff_fitness: {baseline_diff_fitness}')
    #print(f'burst_peak_variance: {burst_peak_variance}')
    print(f'burst_peak_frequency: {burst_peak_frequency_fitness}')
    print(f'\033[1mfitness: {fitness}\033[0m')
    print(f'\033[1maverage_scaled_fitness: {average_scaled_fitness}\033[0m')

    #output_path = '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5_optimizing/output'
    #output_path = batch.saveFolder
    #output_path = os.getcwd()   

    #save fitness results to file
    import json
    #get folder from simLabel
    gen_folder = simLabel.split('_cand')[0]
    with open(f'{output_path}/{gen_folder}/{simLabel}_Fitness.json', 'w') as f:
        json.dump(fitnessResults, f)
    print(f'Fitness results saved to {output_path}/{gen_folder}/{simLabel}_Fitness.json')
    
    #plot figures - 24Mar24 - I think this is causing crashes...moving this somewhere else
    # fitness_thresh = 500
    # fitness_check = max(fitness, average_scaled_fitness)
    # if fitness_check < fitness_thresh:   
    # #if len(rasterData['spkt']) > 0:
    #     #plot figures
    #     plot_sim_figs(output_path, simLabel)

    #Run at the end of each generation instead?
    #plot on completion of core run (I'm not exactly sure when this would happen chronologically, but it should would)


    #plot figures
    #gen_number = simLabel.split('_gen')[1].split('_cand')[0]
    cand_number = int(simLabel.split('_cand_')[1])
    num_completed_fitness_files = len(glob.glob(f'{output_path}/{gen_folder}/*_Fitness.json'))
    pop_size = kwargs['pop_size']
    plot_report_path = f'{output_path}/{gen_folder}/plot_report.json'
    if not os.path.exists(plot_report_path): # or cand_number == pop_size-1: #this way only the last population process works on plotting
        if num_completed_fitness_files >= pop_size-1: #and 
            #os.makedirs(f'{output_path}/{gen_folder}/plots', exist_ok=True)
            #plot_report = {}
            
            # import signal
            # # Timeout handler function
            # def timeout_handler(signum, frame):
            #     raise TimeoutError("Function call timed out")
            
            # def plot_during_last_cand():#timeout = 900): #10 min
            #     # # Set the signal handler and a timeout alarm
            #     # signal.signal(signal.SIGALRM, timeout_handler)
            #     # signal.alarm(timeout)

            # try:
                #Generation Complete
            timeout = 1200 #20 min
            fitness_thresh = 250
            print(f"Generation Complete. Plotting figures with below threshold fitness: {fitness_thresh}")
            print(f'Plotting...')        
            try: from batch_run_files.plot_config import plot_sim_figs
            except: from plot_config import plot_sim_figs
            batch_run_path = output_path
            gen_path = os.path.join(batch_run_path, gen_folder)
            net_activity_params = {'binSize': .03*1000, 'gaussianSigma': .12*1000, 'thresholdBurst': 1.0}
            plot_report = plot_sim_figs(gen_path, fitness_threshold = fitness_thresh, simLabel = None, net_activity_params = net_activity_params, timeout = timeout)
                #reset the alarm
            #     signal.alarm(0)
            # except TimeoutError as e:
            #     print(f"Error: {e}")
            #     print(f"Plotting timed out after {timeout} seconds.")
            #     #reset the alarm
            #     signal.alarm(0)
            if len(plot_report)>0:
                with open(plot_report_path, 'w') as f:
                    json.dump(plot_report, f)

    return average_scaled_fitness