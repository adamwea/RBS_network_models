#from netpyne import sim
import numpy as np
import sys
#sys.path.insert(0, '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5_optimizing/batch_run_files')
from batch_helper_functions import measure_network_activity
from batch_helper_functions import find_batch_object_and_sim_label
#from netpyne import sim
from scipy.stats import linregress
import os
import glob
import json

# Import USER_INPUTS
from USER_INPUTS import *

def fitnessFunc(simData, plot = False, simLabel = None, batch_saveFolder = None, fitness_save_path = None, **kwargs):   
    maxFitness = kwargs['maxFitness']
    ''' subfuncs '''
    def get_network_activity_metrics(fitnessVals, plot = False):
        net_activity_metrics = {}
        #prepare raster data
        rasterData = simData
        
        # Check if the rasterData has elements
        try: 
            assert USER_raster_convolve_params, 'USER_raster_convolve_params needs to be specified in USER_INPUTS.py'
            net_activity_params = USER_raster_convolve_params #{'binSize': .03*1000, 'gaussianSigma': .12*1000, 'thresholdBurst': 1.0}
            binSize = net_activity_params['binSize']
            gaussianSigma = net_activity_params['gaussianSigma']
            thresholdBurst = net_activity_params['thresholdBurst']
            assert len(rasterData['spkt']) > 0, 'Error: rasterData has no elements. burstPeak, baseline, slopeFitness, and IBI fitness set to 1000.'                       
            # Generate the network activity plot with a size of (10, 5)
            plotting_params = None
            if plot:
                assert USER_plotting_params is not None, 'USER_plotting_params must be set in USER_INPUTS.py'
                plotting_params = USER_plotting_params['NetworkActivity']
                plotting_params['simLabel'] = simLabel
                plotting_params['batch_saveFolder'] = batch_saveFolder
            net_metrics = measure_network_activity(
                rasterData, 
                binSize=binSize, 
                gaussianSigma=gaussianSigma, 
                thresholdBurst=thresholdBurst,
                plot=plot,
                plotting_params = plotting_params,
                crop = USER_raster_crop 
            )
        except Exception as e:
            print(f'Error calculating network activity metrics.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            fitnessVals['burstAmp_Fitness'] = maxFitness
            fitnessVals['IBI_fitness'] = maxFitness
            fitnessVals['baselineFitness'] = maxFitness
            fitnessVals['slopeFitness'] = maxFitness
            fitnessVals['baseline_diff_fitness'] = maxFitness
            fitnessVals['burst_peak_frequency_fitness'] = maxFitness
            fitnessVals['rate_fitness'] = maxFitness
            
            return fitnessVals, net_activity_metrics            
        
        #net_activity_metrics = {}
        net_activity_metrics['burstPeakValues'] = net_metrics['burstPeakValues']
        net_activity_metrics['IBIs'] = net_metrics['IBIs']
        net_activity_metrics['baseline'] = net_metrics['baseline']
        net_activity_metrics['baselineDiff'] = net_metrics['baseline_diff']
        net_activity_metrics['normalizedPeakVariance'] = net_metrics['normalized_peak_variance']
        net_activity_metrics['peakFreq'] = net_metrics['peak_freq']
        net_activity_metrics['firingRate'] = net_metrics['firingRate']
        net_activity_metrics['burstPeakTimes'] = net_metrics['burstPeakTimes']

        return fitnessVals, net_activity_metrics
        
    def fit_burst_freuqency(net_activity_metrics, fitnessVals, plot = False, **kwargs):
        
        ##
        try:
            #burst_peak_frequency Fitness
            burst_peak_frequency = net_activity_metrics['peakFreq']
            pops = kwargs['pops']
            pops_frequency = pops['burst_peak_frequency']
            maxFitness = kwargs['maxFitness']
            # Calculate the fitness as the absolute difference between the frequency and the target frequency
            burst_peak_frequency_fitness = min(np.exp(
                abs(pops_frequency['target'] - burst_peak_frequency) / pops_frequency['width']), maxFitness) if burst_peak_frequency > pops_frequency['min'] else maxFitness
            # Print the frequency and its fitness
            print('Frequency of burst peaks: %.3f, Fitness: %.3f' % (burst_peak_frequency, burst_peak_frequency_fitness))
            fitnessVals['burst_peak_frequency_fitness'] = burst_peak_frequency_fitness
            return fitnessVals
        except Exception as e:
            print(f'Error calculating burst peak frequency fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            fitnessVals['burst_peak_frequency_fitness'] = maxFitness
            return fitnessVals
        
    def fit_baseline_diff(net_activity_metrics, fitnessVals, plot = False, **kwargs):
        ##
        try:
            #baseline_diff Fitness
            pops = kwargs['pops']
            baseline_diff_target = pops['baseline_diff']
            maxFitness = kwargs['maxFitness']
            baseline_diff = net_activity_metrics['baselineDiff']

            # Calculate the fitness as the absolute difference between the baseline and the target baseline
            baseline_diff_fitness = min(np.exp(baseline_diff_target['target'] - baseline_diff) / baseline_diff_target['width'], maxFitness) if baseline_diff > baseline_diff_target['min'] else maxFitness

            # Print the baseline_diff and its fitness
            print('Difference between baseline and threshold: %.3f, Fitness: %.3f' % (baseline_diff, baseline_diff_fitness))
            fitnessVals['baseline_diff_fitness'] = baseline_diff_fitness
            return fitnessVals
        except Exception as e:
            print(f'Error calculating baseline_diff fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            baseline_diff_fitness = maxFitness
            fitnessVals['baseline_diff_fitness'] = baseline_diff_fitness
            return fitnessVals

    def fit_rate_slope(net_activity_metrics, fitnessVals, plot = False, **kwargs):
        try:
            # Firing rate fitness
            pops = kwargs['pops']
            rate_slope= pops['rate_slope']
            maxFitness = kwargs['maxFitness']

            # Get the firing rate from the network metrics
            firingRate = net_activity_metrics['firingRate']

            # Calculate the trendline of the firing rate
            slope, intercept, r_value, p_value, std_err = linregress(range(len(firingRate)), firingRate)

            # Calculate the fitness as the absolute difference between the slope and the target slope
            slopeFitness = min(np.exp(abs(rate_slope['target'] - slope)/rate_slope['width']) if abs(slope) < rate_slope['max'] else maxFitness,
                                maxFitness)

            # Print the slope and its fitness
            print('Slope of firing rate: %.3f, Fitness: %.3f' % (slope, slopeFitness))
            fitnessVals['slopeFitness'] = slopeFitness
            return fitnessVals
        except Exception as e:
            print(f'Error calculating firing rate slope fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            slopeFitness = maxFitness 
            fitnessVals['slopeFitness'] = slopeFitness
            return fitnessVals

    def fit_burst_peak(net_activity_metrics, fitnessVals, plot = False, **kwargs):
        try: 
            #burstPeakValue Fitness
            pops = kwargs['pops']
            pops_peaks = pops['burts_peak_targets']
            print(kwargs)
            maxFitness = kwargs['maxFitness']
            burstPeakValues = net_activity_metrics['burstPeakValues']
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
            fitnessVals['burstAmp_Fitness'] = BurstVal_fitness
            return fitnessVals
        except Exception as e:
            print(f'Error calculating burst peak fitness.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            BurstVal_fitness = maxFitness
            fitnessVals['burstAmp_Fitness'] = BurstVal_fitness
            return fitnessVals

    def fit_IBI(net_activity_metrics, fitnessVals, plot = False, **kwargs):
        try:
            assert len(net_activity_metrics['burstPeakTimes']) > 0, 'Error: burstPeakTimes has less than 1 element. IBI could be calculated.'
            assert len(net_activity_metrics['burstPeakValues']) > 1, 'Error: burstPeakValues has less than 2 elements. IBI could be calculated.'
            assert len(net_activity_metrics['IBIs']) > 0, 'Error: IBIs has no elements. IBI could not be calculated.'
            pops = kwargs['pops']
            pops_IBI = pops['IBI_targets']
            maxFitness = kwargs['maxFitness']
            IBIs = net_activity_metrics['IBIs']

            popFitnessIBI = [
                min(np.exp(abs(pops_IBI['target'] - value) / pops_IBI['width']), maxFitness)
                if value < pops_IBI['max'] else maxFitness for value in IBIs
            ]
            IBI_fitness = np.mean(popFitnessIBI)

            #Calc average IBI
            avg_IBI = np.mean(IBIs)

            #print average IBI and its fitness
            print('Average IBI: %.3f, Fitness: %.3f' % (avg_IBI, IBI_fitness))
            fitnessVals['IBI_fitness'] = IBI_fitness
            return fitnessVals
        except Exception as e:
            print(f'Error calculating IBI fitness.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            IBI_fitness = maxFitness
            fitnessVals['IBI_fitness'] = IBI_fitness
            return fitnessVals

    def fit_baseline(net_activity_metrics, fitnessVals, plot = False, **kwargs):
        try:
            assert net_activity_metrics['baseline'] is not None, 'Error: baseline is None. Baseline could not be calculated.'
            #assert len(baseline) > 0, 'Error: baseline has no elements. Baseline could not be calculated.'
            pops = kwargs['pops']
            pops_baseline = pops['baseline_targets']
            maxFitness = kwargs['maxFitness']
            baseline = net_activity_metrics['baseline']
            baseline_target = pops_baseline['target']
            baseline_width = pops_baseline['width']
            baseline_max = pops_baseline['max']

            baselineFitness = min(np.exp(abs(baseline_target - baseline) / baseline_width), maxFitness) if baseline < baseline_max else maxFitness

            #print average baseline and its fitness
            print('baseline: %.3f, Fitness: %.3f' % (baseline, baselineFitness)) 
            fitnessVals['baselineFitness'] = baselineFitness
            return fitnessVals
        except Exception as e:
            print(f'Error calculating baseline fitness.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            baselineFitness = maxFitness
            fitnessVals['baselineFitness'] = baselineFitness
            return fitnessVals

    def fit_firing_rate(net_activity_metrics, fitnessVals, simData, plot = False, **kwargs):
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
            popInfo = '; '.join(['%s rate=%.1f fit=%1.f'%(p,r,f) for p,r,f in zip(
                list(simData['popRates'].keys()), 
                list(simData['popRates'].values()), popFitness)])
            print('  '+popInfo)
            fitnessVals['rate_fitness'] = rate_fitness
            return fitnessVals
        except Exception as e:
            print(f'Error calculating firing rate fitness.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            rate_fitness = maxFitness
            fitnessVals['rate_fitness'] = rate_fitness
            return fitnessVals
    
    def fitness_summary_metrics(fitnessVals):
        #average fitness
        average_fitness = sum(fitnessVals.values()) / len(fitnessVals)

        ##normalized and scaled average fitness - mitigate the effect of outliers
        # Assuming fitness values are stored in a dictionary
        fitness_values = list(fitnessVals.values())

        # Calculate min and max values
        min_value = min(fitness_values)
        max_value = max(fitness_values)

        # Normalize the fitness values
        if max_value > min_value:
            normalized_fitness_values = [(value - min_value) / (max_value - min_value) for value in fitness_values]
            if any(np.isnan(value) for value in normalized_fitness_values):
                #set all values to 1
                normalized_fitness_values = [1 for _ in fitness_values]
            # Scale the normalized values back to the original range (0-1000)
            scaled_fitness_values = [value * 1000 for value in normalized_fitness_values]
        else:
            scaled_fitness_values = [1000 for _ in fitness_values] 

        # Calculate the average of the scaled fitness values
        avg_scaled_fitness = sum(scaled_fitness_values) / len(scaled_fitness_values)

        return average_fitness, avg_scaled_fitness
    
    def get_fitness(simData, plot = False, **kwargs):
        '''init'''
        maxFitness = kwargs['maxFitness']
        fitnessVals = {}
        ## Get the fitness function arguments
        output_path = batch_saveFolder
        print(f'Calculating net_actiity_metrics...')
        assert USER_plot_fitness_bool is not None, 'USER_plot_fitness_bool must be set in USER_INPUTS.py'
        fitnessVals, net_activity_metrics = get_network_activity_metrics(fitnessVals, plot = plot)
        if net_activity_metrics == {}:
            return maxFitness

        ## Get the fitness values
        # Get burst peak fitness, optionally plot
        print('Calculating burst peak fit...')
        fitnessVals = fit_burst_peak(net_activity_metrics, fitnessVals, plot = plot, **kwargs)
        # Get IBI fitness, optionally plot
        print('Calculating IBI fit...')
        fitnessVals = fit_IBI(net_activity_metrics, fitnessVals, plot = plot, **kwargs)
        # Get baseline fitness, optionally plot
        print('Calculating baseline fit...')
        fitnessVals = fit_baseline(net_activity_metrics, fitnessVals, plot = plot, **kwargs)
        # Get rate slope fitness, optionally plot
        print('Calculating rate slope fit...')
        fitnessVals = fit_rate_slope(net_activity_metrics, fitnessVals, plot = plot, **kwargs)
        #Get baseline fitness, optionally plot
        print('Calculating baseline diff fit...')
        fitnessVals = fit_baseline_diff(net_activity_metrics, fitnessVals, plot = plot, **kwargs)
        # Get burst freq fitness, optionally plot
        print('Calculating burst frequency fit...')
        fitnessVals = fit_burst_freuqency(net_activity_metrics, fitnessVals, plot = plot, **kwargs)       
        # Get firing rate fitness, optionally plot
        print('Calculating firing rate fit...')
        fitnessVals = fit_firing_rate(net_activity_metrics, fitnessVals, simData, plot = plot, **kwargs)

        # Get the fitness summary metrics        
        average_fitness, avg_scaled_fitness = fitness_summary_metrics(fitnessVals)
        print(f'Average Fitness: {average_fitness}')
        print(f'Average Scaled Fitness: {avg_scaled_fitness}')

        ##save fitness values
        maxFitness = kwargs['maxFitness']
        fitnessResults = {}
        for key, value in fitnessVals.items():
            fitnessResults[f'{key}'] = value
        fitnessResults['maxFitness'] = maxFitness
        fitnessResults['average_fitness'] = average_fitness
        fitnessResults['average_scaled_fitness'] = avg_scaled_fitness

        ##print fitness results on multiple lines
        # for key, value in fitnessResults.items():
        #     print(f'{key}: {value}')

        ##save fitness results to file        
        #get folder from simLabel
        gen_folder = simLabel.split('_cand')[0]
        if fitness_save_path is None:
            #typical case, during simulations
            with open(f'{output_path}/{gen_folder}/{simLabel}_Fitness.json', 'w') as f:
                json.dump(fitnessResults, f, indent=4)
        else:
            #while plotting
            with open(f'{fitness_save_path}/{simLabel}_Fitness.json', 'w') as f:
                json.dump(fitnessResults, f, indent=4)

        print(f'Fitness results saved to {output_path}/{gen_folder}/{simLabel}_Fitness.json')
                
        return avg_scaled_fitness
    
    '''
    Get Fitness (Main Fitness Function)
    '''    
    ##find relevant batch object in call stack
    # Use the function to get the Batch object and simLabel
    if batch_saveFolder is None and simLabel is None:
        batch, simLabel = find_batch_object_and_sim_label()
        batch_saveFolder = batch.saveFolder
    assert simLabel is not None, "SimLabel undefined."
    assert batch_saveFolder is not None, "Batch save folder undefined."

    ##get fitness
    if plot is None: plot = USER_plot_fitness_bool
    avg_scaled_fitness = get_fitness(simData, plot = plot, **kwargs)
    return avg_scaled_fitness