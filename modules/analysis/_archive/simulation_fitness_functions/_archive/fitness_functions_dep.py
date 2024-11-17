# Standard library imports
import os
import sys
import json
import glob

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
from scipy.signal import convolve, find_peaks, butter, filtfilt
from scipy.stats import linregress, norm, stats

# Local application imports
from helper_functions import find_batch_object_and_sim_label
from USER_INPUTS import *

# Netpyne imports
import netpyne

'''helper functions'''
def load_clean_sim_object(data_file_path):
        # #netpyne.sim.initialize() #initialize netpyne
        # print('Loading data from:', data_file_path)
        # netpyne.sim.loadAll(data_file_path)
        print('clearing all data')
        try: netpyne.sim.clearAll() #clear all sim data
        except: pass
        print('loading all data')
        netpyne.sim.loadAll(data_file_path)
        #print('test concluded')
'''tunable fitness functions'''
def fit_baseline(net_activity_metrics, **kwargs):
    def fitness_function(baseline, baseline_target, baseline_min=0, baseline_max=700, maxFitness=1000, scale_factor=1.0):
        if baseline_min <= baseline <= baseline_max:
            # Calculate the fitness value using a smooth decreasing function
            #print(f'baseline: {baseline}, baseline_target: {baseline_target}')
            #print(f'baseline_min: {baseline_min}, baseline_max: {baseline_max}, maxFitness: {maxFitness}')
            # Calculate the fitness value using a smooth decreasing function
            fitness = 1 + (maxFitness - 1) * (1 - np.exp(-abs(baseline - baseline_target) / (baseline_max - baseline_min)))*scale_factor

        else:
            fitness = maxFitness
        fitness = min(fitness, maxFitness)
        return fitness
    try:
        print('Calculating baseline fitness...')
        assert net_activity_metrics['baseline'] is not None, 'Error: baseline is None. Baseline could not be calculated.'
        pops = kwargs['pops']
        pops_baseline = pops['baseline_target']
        maxFitness = kwargs['maxFitness']
        baseline = net_activity_metrics['baseline']
        baseline_target = pops_baseline['target']
        #baseline_width = pops_baseline['width']
        baseline_max = pops_baseline['max']
        baseline_min = pops_baseline['min']
        width = pops_baseline['width']
        scale_factor = pops_baseline['scale_factor']

        # Calculate the fitness value
        baselineFitness = fitness_function(baseline, baseline_target, baseline_min, baseline_max, maxFitness, scale_factor=scale_factor) 

        # baselineFitness = [min(np.exp(abs(baseline_target - baseline)), maxFitness) 
        #                     if baseline <= baseline_max and baseline >=baseline_min else maxFitness]
        # baselineFitness = baselineFitness[0]

        #print average baseline and its fitness
        print('baseline: %.3f, Fitness: %.3f' % (baseline, baselineFitness)) 
        #fitnessVals['baseline_fitness'] = {'Value': baseline, 'Fit': baselineFitness}
        return {'Value': baseline, 'Fit': baselineFitness}
    except Exception as e:
        print(f'Error calculating baseline fitness.')
        print(f'Error: {e}')
        #set fitness values to maxFitness
        maxFitness = kwargs['maxFitness']
        print('baseline: '+f'{None} fit={maxFitness:.3f}')
        #fitnessVals['baseline_fitness'] = {'Value': None, 'Fit': maxFitness}
        return {'Value': None, 'Fit': maxFitness}
def fit_burst_frequency(net_activity_metrics, **kwargs):
    def fitness_function(burst_peak_frequency, target, burst_freq_min=0, burst_freq_max=700, maxFitness=1000, scale_factor=1.0):
        if burst_freq_min < burst_peak_frequency <= burst_freq_max:
            # Calculate the fitness value using a smooth decreasing function
            #print(f'baseline: {baseline}, baseline_target: {baseline_target}')
            #print(f'baseline_min: {baseline_min}, baseline_max: {baseline_max}, maxFitness: {maxFitness}')
            # Calculate the fitness value using a smooth decreasing function
            fitness = 1 + (maxFitness - 1) * (1 - np.exp(-abs(burst_peak_frequency - target) 
                                                         / (burst_freq_max - burst_freq_min)
                                                         ))*scale_factor

        else:
            fitness = maxFitness
        fitness = min(fitness, maxFitness)
        return fitness
    ##
    try:
        #burst_peak_frequency Fitness
        burst_peak_frequency = net_activity_metrics['peakFreq']
        pops = kwargs['pops']
        pops_frequency = pops['burst_frequency_target']
        maxFitness = kwargs['maxFitness']
        scale_factor = pops_frequency['scale_factor']
        max_freq = pops_frequency['max']
        min_freq = pops_frequency['min']
        # Calculate the fitness as the absolute difference between the frequency and the target frequency
        burst_peak_frequency_fitness = fitness_function(burst_peak_frequency, pops_frequency['target'], min_freq, max_freq, maxFitness, scale_factor=scale_factor)
        # burst_peak_frequency_fitness = [min(np.exp(
        #     abs(pops_frequency['target'] - burst_peak_frequency)), maxFitness) 
        #     if burst_peak_frequency > pops_frequency['min'] and burst_peak_frequency < pops_frequency['max'] else maxFitness]
        # burst_peak_frequency_fitness = burst_peak_frequency_fitness[0]
        # Print the frequency and its fitness
        print('Burst Frequency: %.3f, Fitness: %.3f' % (burst_peak_frequency, burst_peak_frequency_fitness))
        #fitnessVals['burst_peak_frequency_fitness'] = {'Value': burst_peak_frequency, 'Fit': burst_peak_frequency_fitness}
        #if plot: plot_burst_freq(fitnessVals)
        return {'Value': burst_peak_frequency, 'Fit': burst_peak_frequency_fitness}
    except Exception as e:
        print(f'Error calculating burst peak frequency fitness.')
        print(f'Error: {e}')
        # Set fitness values to maxFitness
        maxFitness = kwargs['maxFitness']
        print('Burst Frequency: '+f'{None} fit={maxFitness:.3f}')
        #fitnessVals['burst_peak_frequency_fitness'] = {'Value': None, 'Fit': maxFitness}
        return {'Value': None, 'Fit': maxFitness} 
def fit_threshold(net_activity_metrics, **kwargs):
    def fitness_function(thresh, target, thresh_min=0, thresh_max=700, maxFitness=1000, scale_factor=1.0):
        if thresh_min <= thresh <= thresh_max:
            # Calculate the fitness value using a smooth decreasing function
            #print(f'baseline: {baseline}, baseline_target: {baseline_target}')
            #print(f'baseline_min: {baseline_min}, baseline_max: {baseline_max}, maxFitness: {maxFitness}')
            # Calculate the fitness value using a smooth decreasing function
            fitness = 1 + (maxFitness - 1) * (1 - np.exp(-abs(thresh - target) 
                                                         / (thresh_max - thresh_min)
                                                         ))*scale_factor

        else:
            fitness = maxFitness
        fitness = min(fitness, maxFitness)
        return fitness
    
    try:
        pops = kwargs['pops']
        maxFitness = kwargs['maxFitness']
        thresh = net_activity_metrics['threshold']
        thresh_target = pops['threshold_target']
        scale_factor = thresh_target['scale_factor']
        max_thresh = thresh_target['max']
        min_thresh = thresh_target['min']

        # # Calculate the fitness as the absolute difference between the threshold and the target threshold
        # thresh_fit = [min(np.exp((thresh_target['target'] - thresh)), maxFitness) 
        #                 if thresh < thresh_target['max'] else maxFitness]
        # thresh_fit = thresh_fit[0]
        thresh_fit = fitness_function(thresh, thresh_target['target'], min_thresh, max_thresh, maxFitness, scale_factor=scale_factor)

        # Print the threshold and its fitness
        print('Thresh: %.3f, Fitness: %.3f' % (thresh, thresh_fit))
        #fitnessVals['thresh'] = {'Value': thresh, 'Fit': thresh_fit}
        #return fitnessVals
        return {'Value': thresh, 'Fit': thresh_fit}
    except Exception as e:
        print(f'Error calculating thresh fitness.')
        print(f'Error: {e}')
        # Set fitness values to maxFitness
        maxFitness = kwargs['maxFitness']
        print('Thresh: '+f'{None} fit={maxFitness:.3f}')
        # fitnessVals['thresh'] = {'Value': None, 'Fit': maxFitness}
        # return fitnessVals 
        return {'Value': None, 'Fit': maxFitness}
def fit_number_bursts(net_activity_metrics, **kwargs):
    def fitness_function(baseline, baseline_target, baseline_min=0, baseline_max=700, maxFitness=1000, scale_factor=1.0):
        if baseline_min <= baseline <= baseline_max:
            # Calculate the fitness value using a smooth decreasing function
            #print(f'baseline: {baseline}, baseline_target: {baseline_target}')
            #print(f'baseline_min: {baseline_min}, baseline_max: {baseline_max}, maxFitness: {maxFitness}')
            # Calculate the fitness value using a smooth decreasing function
            fitness = 1 + (maxFitness - 1) * (1 - np.exp(-abs(baseline - baseline_target) / (baseline_max - baseline_min)))*scale_factor

        else:
            fitness = maxFitness
        fitness = min(fitness, maxFitness)
        return fitness
    try:
        print('Calculating baseline fitness...')
        assert net_activity_metrics['baseline'] is not None, 'Error: baseline is None. Baseline could not be calculated.'
        pops = kwargs['pops']
        pops_baseline = pops['baseline_target']
        maxFitness = kwargs['maxFitness']
        baseline = net_activity_metrics['baseline']
        baseline_target = pops_baseline['target']
        #baseline_width = pops_baseline['width']
        baseline_max = pops_baseline['max']
        baseline_min = pops_baseline['min']
        width = pops_baseline['width']
        scale_factor = pops_baseline['scale_factor']

        # Calculate the fitness value
        baselineFitness = fitness_function(baseline, baseline_target, baseline_min, baseline_max, maxFitness, scale_factor=scale_factor) 

        # baselineFitness = [min(np.exp(abs(baseline_target - baseline)), maxFitness) 
        #                     if baseline <= baseline_max and baseline >=baseline_min else maxFitness]
        # baselineFitness = baselineFitness[0]

        #print average baseline and its fitness
        print('baseline: %.3f, Fitness: %.3f' % (baseline, baselineFitness)) 
        #fitnessVals['baseline_fitness'] = {'Value': baseline, 'Fit': baselineFitness}
        return {'Value': baseline, 'Fit': baselineFitness}
    except Exception as e:
        print(f'Error calculating baseline fitness.')
        print(f'Error: {e}')
        #set fitness values to maxFitness
        maxFitness = kwargs['maxFitness']
        print('baseline: '+f'{None} fit={maxFitness:.3f}')
        #fitnessVals['baseline_fitness'] = {'Value': None, 'Fit': maxFitness}
        return {'Value': None, 'Fit': maxFitness}

def fitnessFunc(simData, plot = None, simLabel = None, data_file_path = None, batch_saveFolder = None, fitness_save_path = None, plot_save_path = None, exp_mode = False, **kwargs):   
   
    '''analysis functions'''
    def measure_network_activity(
        rasterData, min_peak_distance = None, binSize=None, gaussianSigma=None, 
        thresholdBurst = None, plot=False, plotting_params = None, crop = None): # thresholdBurst=1.2, crop = None): #, figSize=(10, 6), saveFig=False, figName='NetworkActivity.png'):
    
        def burstPeakQualityControl(burstPeakTimes, burstPeakValues):
            # Get the indices of the valid peaks
            #valid_peak_indices = np.where((burstPeakTimes > timeVector[0]) & (burstPeakTimes < timeVector[-1]))[0]
            # Filter burstPeakTimes and burstPeakValues using the valid peak indices
            #burstPeakTimes = burstPeakTimes[valid_peak_indices]
            #burstPeakValues = burstPeakValues[valid_peak_indices]
            #remove negative values
            burstPeakValues = np.array(burstPeakValues)
            burstPeakTimes = np.array(burstPeakTimes)
            burstPeakValues = burstPeakValues[burstPeakValues > 0]
            burstPeakTimes = burstPeakTimes[burstPeakValues > 0]
            
            # identify indices of statistical outlier PeakValues in the positive direction
            outliers_bool = False
            if outliers_bool:
                z = np.abs(stats.zscore(burstPeakValues))
                outliers = np.where((z > 3) & (burstPeakValues > np.mean(burstPeakValues)))[0]
                #check if outliers occur during the first 10% of the simulation
                if len(outliers) > 0:
                    early_outliers = outliers[outliers < len(burstPeakValues)*0.1]
                    if len(early_outliers) > 0:
                        #identify the latest outlier in the early group
                        latest_early_outlier = early_outliers[-1]
                        #remove values before the latest early outlier
                        burstPeakValues = burstPeakValues[latest_early_outlier+1:]
                        burstPeakTimes = burstPeakTimes[latest_early_outlier+1:]
                #check if outliers occur during final 10% of sim
                if len(outliers) > 0:
                    late_outliers = outliers[outliers > len(burstPeakValues)*0.9]
                    if len(late_outliers) > 0:
                        #identify the earliest outlier in the late group
                        earliest_late_outlier = late_outliers[0]
                        #remove values after the earliest late outlier
                        burstPeakValues = burstPeakValues[:earliest_late_outlier-1]
                        burstPeakTimes = burstPeakTimes[:earliest_late_outlier-1]
                    
            #identify indicies of burstPeakStarts and burstPeakEnds (where signal crosses threshold)
            burstPeakStarts = []
            burstPeakEnds = []
            threshold = thresholdBurst * rmsFiringRate
            for i in range(1, len(firingRate)):
                if firingRate[i-1] < threshold and firingRate[i] >= threshold:
                    burstPeakStarts.append(timeVector[i])
                elif firingRate[i-1] >= threshold and firingRate[i] < threshold:
                    burstPeakEnds.append(timeVector[i])
            #eliminate any values from burstPeakValues and Times that are not the max value between starts and stops
            new_burstPeakValues = []
            new_burstPeakTimes = []
            for i in range(len(burstPeakStarts)):
                start = burstPeakStarts[i]
                end = burstPeakEnds[i]
                max_value = np.max(firingRate[np.where((timeVector >= start) & (timeVector <= end))])
                max_index = np.where(firingRate == max_value)[0][0]
                if max_value in burstPeakValues:
                    new_burstPeakValues.append(max_value)
                    new_burstPeakTimes.append(timeVector[max_index])
            burstPeakValues = new_burstPeakValues
            burstPeakTimes = new_burstPeakTimes
            assert len(burstPeakValues) == len(burstPeakTimes), 'burstPeakValues and burstPeakTimes must be the same length'
            return burstPeakTimes, burstPeakValues, burstPeakStarts, burstPeakEnds
    
        '''
        Init
        '''    
        #Get relative spike times data
        try: SpikeTimes = rasterData['spkTimes']
        except: 
            try: SpikeTimes = rasterData['spkt']
            except: 
                print('Error: No spike times found in rasterData')
                return None
        relativeSpikeTimes = SpikeTimes
        relativeSpikeTimes = np.array(relativeSpikeTimes)
        relativeSpikeTimes = relativeSpikeTimes - relativeSpikeTimes[0]  # Set the first spike time to 0
        assert binSize is not None, 'binSize must be specified'
        assert gaussianSigma is not None, 'gaussianSigma must be specified'
        assert thresholdBurst is not None, 'thresholdBurst must be specified'

        '''
        Convolve Spike Data
        '''    
        # Step 1: Bin all spike times into small time windows
        timeVector = np.arange(0, max(relativeSpikeTimes) + binSize, binSize)  # Time vector for binning
        binnedTimes, _ = np.histogram(relativeSpikeTimes, bins=timeVector)  # Bin spike times
        binnedTimes = np.append(binnedTimes, 0)  # Append 0 to match MATLAB's binnedTimes length

        # Step 2: Smooth the binned spike times with a Gaussian kernel
        kernelRange = np.arange(-3*gaussianSigma, 3*gaussianSigma + binSize, binSize)  # Range for Gaussian kernel
        kernel = norm.pdf(kernelRange, 0, gaussianSigma)  # Gaussian kernel
        kernel *= binSize  # Normalize kernel by bin size
        firingRate = convolve(binnedTimes, kernel, mode='same') / binSize  # Convolve and normalize by bin size
        
        # Step 3: Crop signal. Exclude extreeme values at begining and end of simulation
        # raw_mean = np.mean(firingRate)
        # base_locs = np.where(np.round(firingRate) == np.round(raw_mean))
        # base_locs = base_locs[0]

        # if crop is not None:
        #     #crop the firing rate
        #     firingRate = firingRate[crop[0]:crop[1]]
        #     timeVector = timeVector[crop[0]:crop[1]]
        # else:
        #     #crop the firing rate
        #     firingRate = firingRate[base_locs[0]:base_locs[-1]]
        #     timeVector = timeVector[base_locs[0]:base_locs[-1]]

        # Step 4: Peak detection on the smoothed and cropped firing rate curve
        rmsFiringRate = np.sqrt(np.mean(firingRate**2))  # Calculate RMS of the firing rate
        peaks, properties = find_peaks(firingRate, height=thresholdBurst * rmsFiringRate, distance=min_peak_distance)  # Find peaks above the threshold
        #convert peak indices to times
        burstPeakTimes = timeVector[peaks]  # Convert peak indices to times
        burstPeakValues = properties['peak_heights']  # Get the peak values

        # Convert min_peak_distance from ms to seconds
        min_distance_check = True
        if min_distance_check:
            try:
                assert len(burstPeakValues) > 1, 'There must be at least 2 peaks to adjust minimum peak distance'
                if min_peak_distance is None: min_peak_distance = 1 #seconds
                else: min_peak_distance = min_peak_distance #seconds
                #min_peak_distance /= 1000 #convert to seconds
                avg_IBI = np.mean(np.diff(burstPeakTimes))
                print(f'Average IBI before 10s correction: {avg_IBI}')
                avg_IBI = np.mean(np.diff(burstPeakTimes))
                print(f'Average IBI before 10s correction: {avg_IBI}')
                culled_peak_times = [burstPeakTimes[0]]  # Start with the first peak
                culled_peak_values = [burstPeakValues[0]]  # Start with the first peak value
                for peak in range(1, len(burstPeakTimes)):
                    if burstPeakTimes[peak] - culled_peak_times[-1] < min_peak_distance:  # Compare with the last added peak
                        if burstPeakValues[peak] > culled_peak_values[-1]:  # If current peak has higher amplitude
                            culled_peak_times[-1] = burstPeakTimes[peak]  # Replace the last added peak
                            culled_peak_values[-1] = burstPeakValues[peak]  # Replace the last added peak value
                    else:  # If not less than 10s apart
                        culled_peak_times.append(burstPeakTimes[peak])
                        culled_peak_values.append(burstPeakValues[peak])

                burstPeakValues = culled_peak_values
                burstPeakTimes = culled_peak_times
                avg_IBI = np.mean(np.diff(burstPeakTimes))
                print(f'Average IBI after 10s correction: {avg_IBI}')
            except Exception as e: 
                print(e)
                pass       

        trim = False
        if trim:
             # Quality Control - which, currently, does nothing
            burstPeakTimes, burstPeakValues, burstPeakStarts, burstPeakEnds = burstPeakQualityControl(burstPeakTimes, burstPeakValues)
            ##Adjust lengths of timeVector and firingRate to start with the latest start before earliest peak and earliest end after latest peak
            #get the latest start before the earliest pe
            # Find the differences between the first peak time and all start times
            differences = burstPeakTimes[0] - burstPeakStarts
            # Only consider start times that are before the first peak time
            valid_starts = np.where(differences >= 0)
            # Find the start time with the smallest difference
            start = burstPeakStarts[np.argmin(differences[valid_starts])]
            start_index = np.where(timeVector == start)[0][0]
            #get the earliest end after the latest peak
            differences = burstPeakEnds - burstPeakTimes[-1]
            #flip order of differences to set end values first
            differences = differences[::-1]
            burstPeakEnds_flipped = burstPeakEnds[::-1]
            # Only consider end times that are after the last peak time
            valid_ends = np.where(differences > 0)
            # Find the end time with the smallest difference
            end = burstPeakEnds_flipped[np.argmin(differences[valid_ends])]
            end_index = np.where(timeVector == end)[0][0]
            #crop the firing rate
            assert start_index < end_index, 'start_index must be less than end_index'
        
            #for later, get len timeVector for comparison after crop
            og_timeVector_len = len(timeVector)

            firingRate = firingRate[start_index:end_index]
            timeVector = timeVector[start_index:end_index]
        else: og_timeVector_len = len(timeVector)

        '''
        Optionally Plot
        '''
        try:
            if plot: 
                assert plotting_params is not None, 'plotting_params must be specified'
                plot_network_activity(plotting_params, timeVector, firingRate, burstPeakTimes, burstPeakValues, thresholdBurst, rmsFiringRate)
        except Exception as e:
            print(f'Error plotting network activity.')
            print(f'Error: {e}')
            
        '''
        Network Metric Outputs
        '''
        # Calculate Baseline
        try: baseline = np.mean(firingRate)
        except: baseline = None
        #measure frequency of peaks
        #peak_freq = len(burstPeakTimes) / (timeVector[-1] / 1000) #convert to seconds
        try: 
            assert len(burstPeakTimes) > 0, 'Error: No burst peaks found. peak_freq set to None.'
            assert len(burstPeakTimes) > 1, 'Error: Only one burst peak found. peak_freq set to None.'
            peak_freq = len(burstPeakTimes) / (timeVector[-1]) #now in seconds
        except: peak_freq = None
        # # Calculate peak variance
        # peak_variance = np.var(burstPeakValues)
        # # Calculate the range of burstPeakValues
        # value_range = np.max(burstPeakValues) - np.min(burstPeakValues)
        # # Calculate the maximum possible variance
        # max_possible_variance = value_range**2
        # # Normalize the variance to a 0 to 1 scale
        # normalized_peak_variance = peak_variance / max_possible_variance if max_possible_variance != 0 else 0
        #measure IBI
        try:
            burstPeakTimes = np.array(burstPeakTimes)
            IBIs = np.diff(burstPeakTimes) #
        except: IBIs = None
        #measure baseline diff
        #height = thresholdBurst * rmsFiringRate
        #baseline_diff = height - baseline    
        #IBI = IBI / 1000 #convert to seconds
        #sustained_osci100 = (len(timeVector)/og_timeVector_len)*100
        measurements = {
            'burstPeakValues': burstPeakValues, #-(thresholdBurst * rmsFiringRate),
            'burstPeakTimes': burstPeakTimes,
            'IBIs': IBIs,
            'firingRate': firingRate,
            'timeVector': timeVector,
            'baseline': baseline,
            #'baseline_diff': baseline_diff,
            'peak_freq': peak_freq,
            #'normalized_peak_variance': normalized_peak_variance,
            'threshold': thresholdBurst * rmsFiringRate,
            #'sustain': sustained_osci100,
            #'base_locs': base_locs,
        }    
        return measurements 
    def get_network_activity_metrics():
        net_activity_metrics = {}
        #prepare raster data
        rasterData = simData.copy()
        #adjust units for simulation data
        if not exp_mode:
            rasterData['spkt'] = np.array(rasterData['spkt'])/1000
            rasterData['t'] = np.array(rasterData['t'])/1000
        
        # Check if the rasterData has elements
        try: 
            assert USER_raster_convolve_params, 'USER_raster_convolve_params needs to be specified in USER_INPUTS.py'
            net_activity_params = USER_raster_convolve_params #{'binSize': .03*1000, 'gaussianSigma': .12*1000, 'thresholdBurst': 1.0}
            binSize = net_activity_params['binSize']
            gaussianSigma = net_activity_params['gaussianSigma']
            thresholdBurst = net_activity_params['thresholdBurst']
            min_peak_distance = net_activity_params['min_peak_distance']
            #experimental data fitting mode
            # if exp_mode: 
            #     rasterData = {}
            #     rasterData['spkt'] = simData
            assert len(rasterData['spkt']) > 0, 'Error: rasterData has no elements. burstPeak, baseline, slopeFitness, and IBI fitness set to 1000.'                       
            # Generate the network activity plot with a size of (10, 5)
            plotting_params = None
            plot = False
            if plot:
                assert USER_plotting_params is not None, 'USER_plotting_params must be set in USER_INPUTS.py'
                plotting_params = USER_plotting_params['NetworkActivity']
                plotting_params['simLabel'] = simLabel
                plotting_params['batch_saveFolder'] = batch_saveFolder
                plotting_params['fresh_plots'] = USER_plotting_params['fresh_plots']
                plotting_params['figsize'] = USER_plotting_params['figsize']
                #make rect instead of square
                plotting_params['figsize'] = (plotting_params['figsize'][0], plotting_params['figsize'][1]/2)
                print('plot_save_path:', plot_save_path)

            net_metrics = measure_network_activity(
                rasterData, 
                binSize=binSize, 
                gaussianSigma=gaussianSigma, 
                thresholdBurst=thresholdBurst,
                min_peak_distance = min_peak_distance,
                plot=plot,
                plotting_params = plotting_params,
                crop = USER_raster_crop 
            )
        except Exception as e:
            print(f'Error calculating network activity metrics.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            #fitnessVals = set_all_fitness_to_max()            
            #net_activity_metrics = {}
            net_activity_metrics['burstPeakValues'] = None
            net_activity_metrics['IBIs'] = None
            net_activity_metrics['baseline'] = None
            #net_activity_metrics['baselineDiff'] = net_metrics['baseline_diff']
            #net_activity_metrics['normalizedPeakVariance'] = net_metrics['normalized_peak_variance']
            net_activity_metrics['peakFreq'] = None
            net_activity_metrics['firingRate'] = None
            net_activity_metrics['burstPeakTimes'] = None
            net_activity_metrics['timeVector'] = None
            net_activity_metrics['threshold'] = None
            #net_activity_metrics['sustained_oscillation'] = net_metrics['sustain'] 
            return net_activity_metrics
           
        
        #net_activity_metrics = {}
        net_activity_metrics['burstPeakValues'] = net_metrics['burstPeakValues']
        net_activity_metrics['IBIs'] = net_metrics['IBIs']
        net_activity_metrics['baseline'] = net_metrics['baseline']
        #net_activity_metrics['baselineDiff'] = net_metrics['baseline_diff']
        #net_activity_metrics['normalizedPeakVariance'] = net_metrics['normalized_peak_variance']
        net_activity_metrics['peakFreq'] = net_metrics['peak_freq']
        net_activity_metrics['firingRate'] = net_metrics['firingRate']
        net_activity_metrics['burstPeakTimes'] = net_metrics['burstPeakTimes']
        net_activity_metrics['timeVector'] = net_metrics['timeVector']
        net_activity_metrics['threshold'] = net_metrics['threshold']
        #net_activity_metrics['sustained_oscillation'] = net_metrics['sustain']

        return net_activity_metrics       
    def get_individual_neuron_metrics():
        '''subfunctions'''
        def get_spike_data_from_simulated_data():
            load_clean_sim_object(data_file_path)
            spike_data = netpyne.sim.analysis.prepareSpikeData()
            return spike_data
        def get_spike_data_from_experimental_data():
            #load npz file
            real_spike_path = data_file_path
            real_spike_data = np.load(real_spike_path, allow_pickle = True)
            data = real_spike_data['spike_array']
            spike_data = {}
            spike_data['exp'] = data
            #print(spike_data.shape[0])
            #print(spike_data.shape[1])
            #spike data is a 2D array. The yaxis is the neuron index, the xaxis is 0s and 1s for spiking.
            #loop through rows to get the spike times for each neuron
            firing_rates = {}
            total_time = len(data[0])/10000 #convert to seconds
            for i in range(len(data)):
                firing_rates[i] = sum(data[i])/total_time
                #print(firing_rates[i])

            #get the top 30% of firing rates, assume these are inhibitory neurons
            #assume the rest are excitatory
            firing_rates = np.array(list(firing_rates.values()))
            I_neurons = firing_rates[firing_rates > np.percentile(firing_rates, 70)]
            E_neurons = firing_rates[firing_rates <= np.percentile(firing_rates, 70)]
            #create a cellpop array to store the type of each neuron
            cellPops = np.array(['E' if i in E_neurons else 'I' for i in firing_rates])
            spike_data['cellPops'] = cellPops
            
            ##everything before this point in the else is pretty much working
            spkTimes = np.arange(0, len(data[0]))/10000*1000 #convert to ms
            spike_times = {}
            spikeInds = {}
            for i in range(len(data)): #basically neuron index in exp data
                spike_times[i] = spkTimes[data[i] == 1]
                spikeInds[i] = np.repeat(i, np.sum(data[i] == 1))
            spike_times = [item for sublist in spike_times.values() for item in sublist]
            spikeInds = [item for sublist in spikeInds.values() for item in sublist]
            # Get the indices that would sort spike_times
            sort_indices = np.argsort(spike_times)
            # Use these indices to sort both spike_times and spikeInds
            spike_times = np.array(spike_times)[sort_indices]
            spikeInds = np.array(spikeInds)[sort_indices]
            spike_data['spkInds'] = spikeInds
            spike_data['spkTimes'] = spike_times
            #spike_data['spkTimes'] = list(spike_times)

            return spike_data, firing_rates
        '''main function'''
        print('Calculating individual firing rates and average ISIs...')
        try:        
            if not exp_mode: spike_data = get_spike_data_from_simulated_data()            
            elif exp_mode: spike_data, firing_rates = get_spike_data_from_experimental_data()          
            else: raise Exception('Error: exp_mode must be set to True or False')
            # Initialize dictionaries to store firing rates and average ISIs
            E_average_ISIs = {}
            I_average_ISIs = {}
            E_average_firing_rates ={}
            I_average_firing_rates = {}

            # Get unique neuron indices
            neuron_indices = np.unique(spike_data['spkInds'])

            # get cell pop type by index
            cellPops = spike_data['cellPops']

            # Calculate FR and average ISI for each neuron
            for neuron_index in neuron_indices:
                # Get spike times for this neuron
                spkTimes_arr = np.array(spike_data['spkTimes']) / 1000 # Convert to seconds
            
                spike_times = spkTimes_arr[spike_data['spkInds'] == neuron_index]                
                if not exp_mode: firing_rate = len(spike_times) / (spike_times[-1] - spike_times[0]) # Calculate firing rate: number of spikes divided by total time
                else: firing_rate = firing_rates[neuron_index]   #handle experimental data tunning case                

                # elif exp_mode:
                #     raster = data[neuron_index]
                #     spike_times = raster #only need this for length of spike times
                #     firing_rate = firing_rates[neuron_index]

                cellType = cellPops[neuron_index]            
                if np.isinf(firing_rate): firing_rate = np.nan
                if 'E' in cellType: E_average_firing_rates[neuron_index] = firing_rate
                if 'I' in cellType: I_average_firing_rates[neuron_index] = firing_rate

                # Calculate average ISI: mean difference between consecutive spike times
                if len(spike_times) > 1:
                    ISIs = np.diff(spike_times)
                    average_ISI = np.mean(ISIs)
                else:
                    average_ISI = np.nan  # No ISI for neurons with less than 2 spikes
                if 'E' in cellType: E_average_ISIs[neuron_index] = average_ISI
                if 'I' in cellType: I_average_ISIs[neuron_index] = average_ISI
            
            neuron_metrics = {
                'E_average_firing_rates': E_average_firing_rates,
                'I_average_firing_rates': I_average_firing_rates,
                'E_average_ISIs': E_average_ISIs,
                'I_average_ISIs': I_average_ISIs
            }
            return neuron_metrics
        except Exception as e:
            print(f'Error calculating individual firing rates and average ISIs.')
            print(f'Error: {e}')
            return None
    
    '''fitting functions'''
    def fit_E_firing_rate(neuron_metrics, **kwargs):
        # Define the tuning curve function
        #TODO: implement tuning fit function?
        # def tune_fit(x, a, b, c, d):           
        #     return a * np.exp(-b * x) + c * x + d
        # if kwargs['tune_fit']: tune_fit()
        try:
            assert neuron_metrics is not None, 'neuron_metrics must be specified'
            print('Calculating excitatory firing rate fitness...')
            pops = kwargs['pops']
            pops_rate = pops['E_rate_target']
            E_FRs = list(neuron_metrics['E_average_firing_rates'].values())
            maxFitness = kwargs['maxFitness']
            target = pops_rate['target']
            min_FR = pops_rate['min']
            max_FR = pops_rate['max']
            width = pops_rate['width']
            '''rate fitness function'''
            popFitness = [
                min(np.exp(abs(target - FR)/width), maxFitness)
                if FR >= min_FR and FR <= max_FR else maxFitness
                for FR in E_FRs
            ]
            E_rate_fitness = np.mean(popFitness)
            E_rate_mean = np.nanmean(E_FRs)
            #rate_fitness = np.nanmean(E_FRs)
            '''kurtosis and skewness'''
            try:
                target = pops_rate['stdev']
                stdev = np.nanstd(E_FRs)
                stdev_fitness = min(np.exp(abs(stdev-target)), maxFitness)
            except: 
                stdev = None
                stdev_fitness = maxFitness

            try:
                target = pops_rate['kurtosis']
                kurtosis = stats.kurtosis(np.nan_to_num(E_FRs))
                kurtosis_fitness = min(np.exp(abs(kurtosis-target)), maxFitness)
            except:
                kurtosis = None
                kurtosis_fitness = maxFitness

            try:
                target = pops_rate['skew']
                skewness = stats.skew(np.nan_to_num(E_FRs))
                skew_fitness = min(np.exp(abs(skewness-target)), maxFitness)
            except:
                skewness = None
                skew_fitness = maxFitness
            '''prioritize'''
            if E_rate_fitness == maxFitness:
                stdev_fitness = maxFitness
                kurtosis_fitness = maxFitness
                skew_fitness = maxFitness
            
            '''combine fitness metrics into a single value'''
            fitness = np.nanmean([E_rate_fitness, kurtosis_fitness, skew_fitness])
            features = {
                'E_rate_mean': E_rate_mean,
                'E_rate_fitness': E_rate_fitness,
                'stdev': stdev,
                'stdev_fitness': stdev_fitness,
                'kurtosis': kurtosis,
                'kurtosis_fitness': kurtosis_fitness,
                'skewness': skewness,
                'skew_fitness': skew_fitness,
                'fitness': fitness
            }
            '''print results'''
            print('  Maximum FR: %.3f' % np.nanmax(E_FRs), 'Neuron index:', E_FRs.index(np.nanmax(E_FRs)))
            print('  Minimum FR: %.3f' % np.nanmin(E_FRs), 'Neuron index:', E_FRs.index(np.nanmin(E_FRs)))
            #print('  Normality: %.3f' % stats.normaltest(E_FRs).pvalue) #Normality is a binary test, not a metric of the distribution
            print('  Mean FR: %.3f' % E_rate_mean, 'Fit:', E_rate_fitness)
            print('  Stdev: %.3f' % stdev, 'Fit:', stdev_fitness)
            print('  Kurtosis: %.3f' % kurtosis, 'Fit:', kurtosis_fitness)
            print('  Skewness: %.3f' % skewness, 'Fit:', skew_fitness)
            # popInfo_E = f'E avg rate={value:.10f} fit={E_rate_fitness:.3f}'
            # print('  Excitatory FR: '+popInfo_E)
            print(f'  Excitatory FR: {E_rate_mean} Overall Fitness: %.3f' % (fitness))
            return {'Value': E_rate_mean, 'Fit': fitness, 'Features': features} #reporting E_rate as main fitness for now
        except Exception as e:
            print(f'Error calculating excitatory firing rate fitness.')
            print(f'Error: {e}')
            maxFitness = kwargs['maxFitness']
            print('  Excitatory FR: '+f'E avg rate={None} fit={maxFitness:.3f}')
            return {'Value': None, 'Fit': maxFitness}
    def fit_I_firing_rate(neuron_metrics, **kwargs):
        try:
            assert neuron_metrics is not None, 'neuron_metrics must be specified'
            print('Calculating inhibitory firing rate fitness...')
            pops = kwargs['pops']
            pops_rate = pops['I_rate_target']
            I_FRs = list(neuron_metrics['I_average_firing_rates'].values())
            maxFitness = kwargs['maxFitness']
            target = pops_rate['target']
            min_FR = pops_rate['min']
            max_FR = pops_rate['max']
            width = pops_rate['width']
            '''rate fitness function'''
            popFitness = [
                min(np.exp(abs(target - FR)/width), maxFitness)
                if FR >= min_FR and FR <= max_FR else maxFitness
                for FR in I_FRs
            ]
            I_rate_fitness = np.mean(popFitness)
            I_rate_mean = np.nanmean(I_FRs)
            '''kurtosis and skewness'''
            try:
                target = pops_rate['stdev']
                stdev = np.nanstd(I_FRs)
                stdev_fitness = min(np.exp(abs(stdev-target)), maxFitness)
            except: 
                stdev = None
                stdev_fitness = maxFitness

            try:
                target = pops_rate['kurtosis']
                kurtosis = stats.kurtosis(np.nan_to_num(I_FRs))
                kurtosis_fitness = min(np.exp(abs(kurtosis-target)), maxFitness)
            except:
                kurtosis = None
                kurtosis_fitness = maxFitness

            try:
                target = pops_rate['skew']
                skewness = stats.skew(np.nan_to_num(I_FRs))
                skew_fitness = min(np.exp(abs(skewness-target)), maxFitness)
            except:
                skewness = None
                skew_fitness = maxFitness
            '''prioritize'''
            if I_rate_fitness == maxFitness:
                stdev_fitness = maxFitness
                kurtosis_fitness = maxFitness
                skew_fitness = maxFitness
            
            '''combine fitness metrics into a single value'''
            fitness = np.nanmean([I_rate_fitness, kurtosis_fitness, skew_fitness])
            features = {
                'I_rate_mean': I_rate_mean,
                'I_rate_fitness': I_rate_fitness,
                'stdev': stdev,
                'stdev_fitness': stdev_fitness,
                'kurtosis': kurtosis,
                'kurtosis_fitness': kurtosis_fitness,
                'skewness': skewness,
                'skew_fitness': skew_fitness,
                'fitness': fitness
            }
            '''print results'''
            print('  Maximum FR: %.3f' % np.nanmax(I_FRs), 'Neuron index:', I_FRs.index(np.nanmax(I_FRs)))
            print('  Minimum FR: %.3f' % np.nanmin(I_FRs), 'Neuron index:', I_FRs.index(np.nanmin(I_FRs)))
            print('  Mean FR: %.3f' % I_rate_mean, 'Fit:', I_rate_fitness)
            print('  Stdev: %.3f' % stdev, 'Fit:', stdev_fitness)
            print('  Kurtosis: %.3f' % kurtosis, 'Fit:', kurtosis_fitness)
            print('  Skewness: %.3f' % skewness, 'Fit:', skew_fitness)
            print(f'  Inhibitory FR: {I_rate_mean} Overall Fitness: %.3f' % (fitness))
            return {'Value': I_rate_mean, 'Fit': fitness, 'Features': features}
        except Exception as e:
            print(f'Error calculating inhibitory firing rate fitness.')
            print(f'Error: {e}')
            maxFitness = kwargs['maxFitness']
            print('  Inhibitory FR: '+f'I avg rate={None} fit={maxFitness:.3f}')
            return {'Value': None, 'Fit': maxFitness}
    def fit_E_ISI(neuron_metrics, **kwargs):
        try:
            assert neuron_metrics is not None, 'neuron_metrics must be specified'
            print('Calculating excitatory ISI fitness...')
            pops = kwargs['pops']
            pops_ISI = pops['E_ISI_target']
            E_ISIs = list(neuron_metrics['E_average_ISIs'].values())
            maxFitness = kwargs['maxFitness']
            target = pops_ISI['target']
            min_ISI = pops_ISI['min']
            max_ISI = pops_ISI['max']
            width = pops_ISI['width']
            '''ISI fitness function'''
            popFitness = [
                min(np.exp(abs(target - ISI)/width), maxFitness)
                if ISI >= min_ISI and ISI <= max_ISI else maxFitness
                for ISI in E_ISIs
            ]
            E_ISI_fitness = np.mean(popFitness)
            E_ISI_mean = np.nanmean(E_ISIs)
            '''kurtosis and skewness'''
            try:
                target = pops_ISI['stdev']
                stdev = np.nanstd(E_ISIs)
                stdev_fitness = min(np.exp(abs(stdev-target)), maxFitness)
            except: 
                stdev = None
                stdev_fitness = maxFitness

            try:
                target = pops_ISI['kurtosis']
                kurtosis = stats.kurtosis(np.nan_to_num(E_ISIs))
                kurtosis_fitness = min(np.exp(abs(kurtosis-target)), maxFitness)
            except:
                kurtosis = None
                kurtosis_fitness = maxFitness

            try:
                target = pops_ISI['skew']
                skewness = stats.skew(np.nan_to_num(E_ISIs))
                skew_fitness = min(np.exp(abs(skewness-target)), maxFitness)
            except:
                skewness = None
                skew_fitness = maxFitness
            '''prioritize'''
            if E_ISI_fitness == maxFitness:
                stdev_fitness = maxFitness
                kurtosis_fitness = maxFitness
                skew_fitness = maxFitness
            
            '''combine fitness metrics into a single value'''
            fitness = np.nanmean([E_ISI_fitness, kurtosis_fitness, skew_fitness])
            features = {
                'E_ISI_mean': E_ISI_mean,
                'E_ISI_fitness': E_ISI_fitness,
                'stdev': stdev,
                'stdev_fitness': stdev_fitness,
                'kurtosis': kurtosis,
                'kurtosis_fitness': kurtosis_fitness,
                'skewness': skewness,
                'skew_fitness': skew_fitness,
                'fitness': fitness
            }
            '''print results'''
            print('  Maximum ISI: %.3f' % np.nanmax(E_ISIs), 'Neuron index:', E_ISIs.index(np.nanmax(E_ISIs)))
            print('  Minimum ISI: %.3f' % np.nanmin(E_ISIs), 'Neuron index:', E_ISIs.index(np.nanmin(E_ISIs)))
            print('  Mean ISI: %.3f' % E_ISI_mean, 'Fit:', E_ISI_fitness)
            print('  Stdev: %.3f' % stdev, 'Fit:', stdev_fitness)
            print('  Kurtosis: %.3f' % kurtosis, 'Fit:', kurtosis_fitness)
            print('  Skewness: %.3f' % skewness, 'Fit:', skew_fitness)
            print(f'  Excitatory ISI: {E_ISI_mean} Overall Fitness: %.3f' % (fitness))
            return {'Value': E_ISI_mean, 'Fit': fitness, 'Features': features}
        except Exception as e:
            print(f'Error calculating excitatory ISI fitness.')
            print(f'Error: {e}')
            maxFitness = kwargs['maxFitness']
            print('  Excitatory ISI: '+f'E avg ISI={None} fit={maxFitness:.3f}')
            return {'Value': None, 'Fit': maxFitness}
    def fit_I_ISI(neuron_metrics, **kwargs):
        try:
            assert neuron_metrics is not None, 'neuron_metrics must be specified'
            print('Calculating inhibitory ISI fitness...')
            pops = kwargs['pops']
            pops_ISI = pops['I_ISI_target']
            I_ISIs = list(neuron_metrics['I_average_ISIs'].values())
            maxFitness = kwargs['maxFitness']
            target = pops_ISI['target']
            min_ISI = pops_ISI['min']
            max_ISI = pops_ISI['max']
            width = pops_ISI['width']
            '''ISI fitness function'''
            popFitness = [
                min(np.exp(abs(target - ISI)/width), maxFitness)
                if ISI >= min_ISI and ISI <= max_ISI else maxFitness
                for ISI in I_ISIs
            ]
            I_ISI_fitness = np.mean(popFitness)
            I_ISI_mean = np.nanmean(I_ISIs)
            '''kurtosis and skewness'''
            try:
                target = pops_ISI['stdev']
                stdev = np.nanstd(I_ISIs)
                stdev_fitness = min(np.exp(abs(stdev-target)), maxFitness)
            except: 
                stdev = None
                stdev_fitness = maxFitness

            try:
                target = pops_ISI['kurtosis']
                kurtosis = stats.kurtosis(np.nan_to_num(I_ISIs))
                kurtosis_fitness = min(np.exp(abs(kurtosis-target)), maxFitness)
            except:
                kurtosis = None
                kurtosis_fitness = maxFitness

            try:
                target = pops_ISI['skew']
                skewness = stats.skew(np.nan_to_num(I_ISIs))
                skew_fitness = min(np.exp(abs(skewness-target)), maxFitness)
            except:
                skewness = None
                skew_fitness = maxFitness
            '''prioritize'''
            if I_ISI_fitness == maxFitness:
                stdev_fitness = maxFitness
                kurtosis_fitness = maxFitness
                skew_fitness = maxFitness
            '''combine fitness metrics into a single value'''
            fitness = np.nanmean([I_ISI_fitness, kurtosis_fitness, skew_fitness])
            features = {
                'I_ISI_mean': I_ISI_mean,
                'I_ISI_fitness': I_ISI_fitness,
                'stdev': stdev,
                'stdev_fitness': stdev_fitness,
                'kurtosis': kurtosis,
                'kurtosis_fitness': kurtosis_fitness,
                'skewness': skewness,
                'skew_fitness': skew_fitness,
                'fitness': fitness
            }
            '''print results'''
            print('  Maximum ISI: %.3f' % np.nanmax(I_ISIs), 'Neuron index:', I_ISIs.index(np.nanmax(I_ISIs)))
            print('  Minimum ISI: %.3f' % np.nanmin(I_ISIs), 'Neuron index:', I_ISIs.index(np.nanmin(I_ISIs)))
            print('  Mean ISI: %.3f' % I_ISI_mean, 'Fit:', I_ISI_fitness)
            print('  Stdev: %.3f' % stdev, 'Fit:', stdev_fitness)
            print('  Kurtosis: %.3f' % kurtosis, 'Fit:', kurtosis_fitness)
            print('  Skewness: %.3f' % skewness, 'Fit:', skew_fitness)
            print(f'  Inhibitory ISI: {I_ISI_mean} Overall Fitness: %.3f' % (fitness))
            return {'Value': I_ISI_mean, 'Fit': fitness, 'Features': features}
        except Exception as e:
            print(f'Error calculating inhibitory ISI fitness.')
            print(f'Error: {e}')
            maxFitness = kwargs['maxFitness']
            print('  Inhibitory ISI: '+f'I avg ISI={None} fit={maxFitness:.3f}')
            return {'Value': None, 'Fit': maxFitness}
    def fit_big_burst_amplitude(net_activity_metrics, **kwargs):
        try:
            print('Calculating big burst amplitude fitness...')
            pops = kwargs['pops']
            cutoff = pops['big-small_cutoff']
            assert len(net_activity_metrics['burstPeakValues']) > 0, 'Error: burstPeakValues has no elements. Big burst amplitude could not be calculated.'
            burstPeakValues = net_activity_metrics['burstPeakValues']
            big_bursts = [value for value in burstPeakValues if value > cutoff]
            assert len(big_bursts) > 0, 'Error: big_bursts has no elements. Big burst amplitude could not be calculated.'
            pops_peaks = pops['big_burst_target']
            target = pops_peaks['target']
            burst_max = pops_peaks['max']
            width = pops_peaks['width']
            maxFitness = kwargs['maxFitness']
            burst_min = pops_peaks['min']

            popFitnessBurstPeak = [
                min(np.exp(abs(target - value) / width), maxFitness)
                if value <= burst_max and value >= burst_min
                else maxFitness for value in big_bursts
            ]
            Big_BurstVal_fitness = np.mean(popFitnessBurstPeak)
            big_val = np.nanmean(big_bursts)

            '''kurtosis and skewness'''
            try:
                target = pops_peaks['stdev']
                stdev = np.nanstd(big_bursts)
                stdev_fitness = min(np.exp(abs(stdev-target)), maxFitness)
            except: 
                stdev = None
                stdev_fitness = maxFitness

            try:
                target = pops_peaks['kurtosis']
                kurtosis = stats.kurtosis(np.nan_to_num(big_bursts))
                kurtosis_fitness = min(np.exp(abs(kurtosis-target)), maxFitness)
            except:
                kurtosis = None
                kurtosis_fitness = maxFitness

            try:
                target = pops_peaks['skew']
                skewness = stats.skew(np.nan_to_num(big_bursts))
                skew_fitness = min(np.exp(abs(skewness-target)), maxFitness)
            except:
                skewness = None
                skew_fitness = maxFitness

            '''prioritize'''
            if Big_BurstVal_fitness == maxFitness:
                stdev_fitness = maxFitness
                kurtosis_fitness = maxFitness
                skew_fitness = maxFitness

            '''summarize results'''
            fitness = np.nanmean([Big_BurstVal_fitness, stdev_fitness, kurtosis_fitness, skew_fitness])

            print('Max Burst Peak: %.3f' % np.nanmax(big_bursts), 'Min Burst Peak: %.3f' % np.nanmin(big_bursts))
            print('Big Burst Peak: %.3f, Fitness: %.3f' % (big_val, Big_BurstVal_fitness))
            print('Big Burst Standard Deviation: %.3f, Fitness: %.3f' % (stdev, stdev_fitness))
            print('Big Burst Kurtosis: %.3f, Fitness: %.3f' % (kurtosis, kurtosis_fitness))
            print('Big Burst Skewness: %.3f, Fitness: %.3f' % (skewness, skew_fitness))

            print('Big Burst avg Amplitude: %.3f, Overall fit: %.3f' % (np.mean(big_bursts), fitness))
            return {'Value': np.mean(big_bursts), 'Fit': fitness, 'features': {
                'StdDev': stdev, 
                'Kurtosis': kurtosis, 
                'Skewness': skewness, 
                'StdDevFit': stdev_fitness, 
                'KurtosisFit': kurtosis_fitness, 
                'SkewnessFit': skew_fitness}}
        except Exception as e:
            maxFitness = kwargs['maxFitness']
            print(f"An error occurred in fit_big_burst_amplitude: {e}")
            print('Big Burst avg Amplitude: %s, Fitness: %.3f' % (str(None), maxFitness))
            return {'Value': None, 'Fit': maxFitness}
    def fit_small_burst_amplitude(net_activity_metrics, **kwargs):
        try:
            print('Calculating small burst amplitude fitness...')
            pops = kwargs['pops']
            cutoff = pops['big-small_cutoff']
            assert len(net_activity_metrics['burstPeakValues']) > 0, 'Error: burstPeakValues has no elements. Small burst amplitude could not be calculated.'
            burstPeakValues = net_activity_metrics['burstPeakValues']
            small_bursts = [value for value in burstPeakValues if value <= cutoff]
            assert len(small_bursts) > 0, 'Error: small_bursts has no elements. Small burst amplitude could not be calculated.'
            pops_peaks = pops['small_burst_target']
            target = pops_peaks['target']
            burst_min = pops_peaks['min']
            burst_max = pops_peaks['max']
            width = pops_peaks['width']
            maxFitness = kwargs['maxFitness']
            small_burst_mean = np.mean(small_bursts)

            popFitnessBurstPeak = [
                min(np.exp(abs(target - value) / width), maxFitness)
                if value >= burst_min and value <= burst_max
                else maxFitness for value in small_bursts
            ]
            Small_BurstVal_fitness = np.mean(popFitnessBurstPeak)

            '''kurtosis and skewness'''
            try:
                target = pops_peaks['stdev']
                stdev = np.nanstd(small_bursts)
                stdev_fitness = min(np.exp(abs(stdev-target)), maxFitness)
            except: 
                stdev = None
                stdev_fitness = maxFitness

            try:
                target = pops_peaks['kurtosis']
                kurtosis = stats.kurtosis(np.nan_to_num(small_bursts))
                kurtosis_fitness = min(np.exp(abs(kurtosis-target)), maxFitness)
            except:
                kurtosis = None
                kurtosis_fitness = maxFitness

            try:
                target = pops_peaks['skew']
                skewness = stats.skew(np.nan_to_num(small_bursts))
                skew_fitness = min(np.exp(abs(skewness-target)), maxFitness)
            except:
                skewness = None
                skew_fitness = maxFitness

            '''prioritize'''
            if Small_BurstVal_fitness == maxFitness:
                stdev_fitness = maxFitness
                kurtosis_fitness = maxFitness
                skew_fitness = maxFitness
            
            '''summarize results'''
            fitness = np.nanmean([Small_BurstVal_fitness, stdev_fitness, kurtosis_fitness, skew_fitness])

            print('Min Burst Peak: %.3f' % np.nanmin(small_bursts), 'Max Burst Peak: %.3f' % np.nanmax(small_bursts))
            print('Small Burst Peak: %.3f, Fitness: %.3f' % (small_burst_mean, Small_BurstVal_fitness))
            print('Small Burst Standard Deviation: %.3f, Fitness: %.3f' % (stdev, stdev_fitness))
            print('Small Burst Kurtosis: %.3f, Fitness: %.3f' % (kurtosis, kurtosis_fitness))
            print('Small Burst Skewness: %.3f, Fitness: %.3f' % (skewness, skew_fitness))

            print('Small Burst avg Amplitude: %.3f, Overall fit: %.3f' % (np.mean(small_bursts), fitness))
            return {'Value': np.mean(small_bursts), 'Fit': fitness, 'features': {
                'StdDev': stdev, 
                'Kurtosis': kurtosis, 
                'Skewness': skewness, 
                'StdDevFit': stdev_fitness, 
                'KurtosisFit': kurtosis_fitness, 
                'SkewnessFit': skew_fitness}}
        except Exception as e:
            maxFitness = kwargs['maxFitness']
            print(f"An error occurred in fit_small_burst_amplitude: {e}")
            print('Small Burst avg Amplitude: %s, Fitness: %.3f' % (str(None), maxFitness))
            return {'Value': None, 'Fit': maxFitness}
            #return {'Value': None, 'Fit': maxFitness}
    def fit_bimodal_burst_amplitude(net_activity_metrics, **kwargs):
        try:
            print('Calculating bimodal burst amplitude fitness...')
            pops = kwargs['pops']
            cutoff = pops['big-small_cutoff']
            assert len(net_activity_metrics['burstPeakValues']) > 0, 'Error: burstPeakValues has no elements. Bimodal burst amplitude could not be calculated.'
            burstPeakValues = net_activity_metrics['burstPeakValues']
            big_bursts = [value for value in burstPeakValues if value > cutoff]
            small_bursts = [value for value in burstPeakValues if value <= cutoff]
            assert len(big_bursts) > 0 and len(small_bursts) > 0, 'Error: big_bursts and small_bursts have no elements. Bimodal burst amplitude could not be calculated.'
            actual_ratio = len(big_bursts) / len(small_bursts)
            print('big_bursts:', len(big_bursts), 'small_bursts:', len(small_bursts))
            print('Bimodal Burst Amplitude Ratio(big:small): %.3f' % actual_ratio)
            pops_peaks = pops['bimodal_burst_target']
            target = pops_peaks['target']
            desired_ratio = target
            maxFitness = kwargs['maxFitness']
            ratio_fitness = min(np.exp(abs(desired_ratio - actual_ratio)), maxFitness)

            # Calculate mean, kurtosis, skew, and std of all bursting amplitudes
            mean_value = np.mean(np.num_to_nan(burstPeakValues))
            stdev_value = np.std(np.num_to_nan(burstPeakValues))
            kurtosis_value = stats.kurtosis(np.num_to_nan(burstPeakValues))
            skew_value = stats.skew(np.num_to_nan(burstPeakValues))

            mean_fitness = min(np.exp(abs(mean_value - pops_peaks['mean'])), maxFitness)
            stdev_fitness = min(np.exp(abs(stdev_value - pops_peaks['stdev'])), maxFitness)
            kurtosis_fitness = min(np.exp(abs(kurtosis_value - pops_peaks['kurtosis'])), maxFitness)
            skew_fitness = min(np.exp(abs(skew_value - pops_peaks['skew'])), maxFitness)

            # '''prioritize''' # dont need to prioritize here
            # if Small_BurstVal_fitness == maxFitness:
            #     stdev_fitness = maxFitness
            #     kurtosis_fitness = maxFitness
            #     skew_fitness = maxFitness
            
            print('Max Burst Peak: %.3f' % np.nanmax(burstPeakValues), 'Min Burst Peak: %.3f' % np.nanmin(burstPeakValues))
            print('Mean Burst Amplitude: %.3f' % mean_value, 'Mean Burst Amplitude Fitness: %.3f' % mean_fitness)
            print('Stdev Burst Amplitude: %.3f' % stdev_value, 'Stdev Burst Amplitude Fitness: %.3f' % stdev_fitness)
            print('Kurtosis Burst Amplitude: %.3f' % kurtosis_value, 'Kurtosis Burst Amplitude Fitness: %.3f' % kurtosis_fitness)
            print('Skewness Burst Amplitude: %.3f' % skew_value, 'Skewness Burst Amplitude Fitness: %.3f' % skew_fitness)

            # Calculate combined fitness
            combined_fitness = np.mean([ratio_fitness, mean_fitness, stdev_fitness, kurtosis_fitness, skew_fitness])
            print('Bimodal Fitness: %.3f' % combined_fitness)

            return {'Value': actual_ratio, 'Fit': combined_fitness, 'Features': {
                'mean': {'Value': mean_value, 'Fit': mean_fitness},
                'stdev': {'Value': stdev_value, 'Fit': stdev_fitness},
                'kurtosis': {'Value': kurtosis_value, 'Fit': kurtosis_fitness},
                'skew': {'Value': skew_value, 'Fit': skew_fitness}
            }}
        except Exception as e:
            maxFitness = kwargs['maxFitness']
            print(f"An error occurred in fit_bimodal_burst_amplitude: {e}")
            print('Bimodal Burst Amplitude Ratio: %s, Fitness: %.3f' % (str(None), maxFitness))
            return {'Value': None, 'Fit': maxFitness}
    def fit_IBI(net_activity_metrics, **kwargs):
        try:
            print('Calculating IBI fitness...')
            assert len(net_activity_metrics['burstPeakTimes']) > 0, 'Error: burstPeakTimes has less than 1 element. IBI could not be calculated.'
            assert len(net_activity_metrics['burstPeakValues']) > 1, 'Error: burstPeakValues has less than 2 elements. IBI could not be calculated.'
            assert len(net_activity_metrics['IBIs']) > 0, 'Error: IBIs has no elements. IBI could not be calculated.'
            pops = kwargs['pops']
            pops_IBI = pops['IBI_target']
            maxFitness = kwargs['maxFitness']
            IBIs = net_activity_metrics['IBIs']
            min_IBI = pops_IBI['min']
            max_IBI = pops_IBI['max']
            width = pops_IBI['width']

            popFitnessIBI = [
                min(np.exp(abs(pops_IBI['target'] - value) / pops_IBI['width']), maxFitness)
                if value <= pops_IBI['max'] and value >= pops_IBI['min']
                else maxFitness for value in IBIs
            ]
            IBI_fitness = np.mean(popFitnessIBI)

            # Calculate average IBI
            avg_IBI = np.nanmean(IBIs)

            # Calculate standard deviation, skewness, and kurtosis
            stdev_IBI = np.nanstd(IBIs)
            skew_IBI = stats.skew(np.nan_to_num(IBIs))
            kurt_IBI = stats.kurtosis(np.nan_to_num(IBIs))

            # Calculate fitness for standard deviation, skewness, and kurtosis
            try:
                target = pops_IBI['stdev']
                stdev_fitness = min(np.exp(abs(target - stdev_IBI)), maxFitness)
            except Exception as e:
                print(f'Error calculating stdev_fitness: {e}')
                stdev_fitness = maxFitness

            try:
                target = pops_IBI['skew']
                skew_fitness = min(np.exp(abs(target - skew_IBI)), maxFitness)
            except Exception as e:
                print(f'Error calculating skew_fitness: {e}')
                skew_fitness = maxFitness

            try:
                target = pops_IBI['kurtosis']
                kurt_fitness = min(np.exp(abs(target - kurt_IBI)), maxFitness)
            except Exception as e:
                print(f'Error calculating kurt_fitness: {e}')
                kurt_fitness = maxFitness

            '''prioritize'''
            if IBI_fitness == maxFitness:
                stdev_fitness = maxFitness
                kurtosis_fitness = maxFitness
                skew_fitness = maxFitness
            
            # Combine fitness metrics into a single value
            overall_fitness = np.mean([IBI_fitness, stdev_fitness, skew_fitness, kurt_fitness])

            # Print results
            print(f'Max IBI: {np.nanmax(IBIs):.3f}, Min IBI: {np.nanmin(IBIs):.3f}')
            print(f'avg IBI: {avg_IBI:.3f}, Fitness: {IBI_fitness:.3f}')
            print(f'Stdev IBI: {stdev_IBI:.3f}, Fitness: {stdev_fitness:.3f}')
            print(f'Skewness IBI: {skew_IBI:.3f}, Fitness: {skew_fitness:.3f}')
            print(f'Kurtosis IBI: {kurt_IBI:.3f}, Fitness: {kurt_fitness:.3f}')
            print(f'Overall Fitness: {overall_fitness:.3f}')

            return {'Value': avg_IBI, 'Fit': overall_fitness, 'Features': {
                'avg_IBI': {'Value': avg_IBI, 'Fit': IBI_fitness},
                'Stdev': {'Value': stdev_IBI, 'Fit': stdev_fitness},
                'Skew': {'Value': skew_IBI, 'Fit': skew_fitness},
                'Kurtosis': {'Value': kurt_IBI, 'Fit': kurt_fitness}
            }}
        except Exception as e:
            print(f'Error calculating IBI fitness.')
            print(f'Error: {e}')
            maxFitness = kwargs['maxFitness']
            print(f'avg IBI: None, fit={maxFitness:.3f}')
            return {'Value': None, 'Fit': maxFitness}
    def fit_sustain(net_activity_metrics, **kwargs):
        try:
            # Sustain Fitness
            pops = kwargs['pops']
            maxFitness = kwargs['maxFitness']
            sustained_osci100 = net_activity_metrics['sustained_oscillation']
            sustained_osci_target = pops['sustained_activity_target']

            # Calculate the fitness as the absolute difference between the sustain duration and the target sustain duration
            sustain_fit = min(np.exp(np.abs(sustained_osci_target['target'] - sustained_osci100)), maxFitness
                             ) # if sustained_osci100 > sustained_osci_target['min'] else maxFitness

            # Print the sustain duration and its fitness
            print('Percent Duration: %.3f, Fitness: %.3f' % (sustained_osci100, sustain_fit))
            # fitnessVals['sustain_oscillation_fitness'] = {'Value': sustained_osci100, 'Fit': sustain_fit}
            # #if plot: plot_sustain(fitnessVals)
            # return fitnessVals
            return {'Value': sustained_osci100, 'Fit': sustain_fit}
        except Exception as e:
            print(f'Error calculating sustain fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            # fitnessVals['sustain_oscillation_fitness'] = {'Value': None, 'Fit': maxFitness}
            # return fitnessVals
            print('Percent Duration: '+f'{None} fit={maxFitness:.3f}')
            return {'Value': None, 'Fit': maxFitness}
    def fit_rate_slope(net_activity_metrics, **kwargs):
        try:
            # Firing rate fitness
            pops = kwargs['pops']
            rate_slope= pops['slope_target']
            maxFitness = kwargs['maxFitness']

            # Get the firing rate from the network metrics
            firingRate = net_activity_metrics['firingRate']

            # Calculate the trendline of the firing rate
            slope, intercept, r_value, p_value, std_err = linregress(range(len(firingRate)), firingRate)

            # Calculate the fitness as the absolute difference between the slope and the target slope
            slopeFitness = min(np.exp(abs(rate_slope['target'] - slope)) 
                               #if abs(slope) < rate_slope['max'] else maxFitness
                               , maxFitness)

            # Print the slope and its fitness
            print('Slope of firing rate: %.3f, Fitness: %.3f' % (slope, slopeFitness))
            # fitnessVals['slopeFitness'] = {'Value': slope, 'Fit': slopeFitness}
            # return fitnessVals
            return {'Value': slope, 'Fit': slopeFitness}
        except Exception as e:
            print(f'Error calculating firing rate slope fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            # fitnessVals['slopeFitness'] = {'Value': None, 'Fit': maxFitness}
            # return fitnessVals
            print('Slope of firing rate: '+f'{None} fit={maxFitness:.3f}')
            return {'Value': None, 'Fit': maxFitness}
    def prioritize_fitness(fitnessVals, **kwargs):
        print('Prioritizing fitness values...')
        maxFitness = kwargs['maxFitness']
        # priorities = [
        #     ['E_rate_fit', 'I_rate_fit', 'E_ISI_fit', 'I_ISI_fit'],  # Priority 1
        #     ['baseline_fit'],  # Priority 2
        #     ['IBI_fitness', 'burst_frequency_fitness'],  # Priority 3
        #     ['big_burst_fit', 'small_burst_fit', 'thresh_fit'],  # Priority 4
        #     ['bimodal_burst_fit', 
        #         #'sustain_fit', 
        #         'slope_fit']  # Priority 5
        # ]
        #TODO: Make priorities easier until I finish all tunning all the fitness functions
        priorities = [
            ['E_rate_fit', 'I_rate_fit', 'E_ISI_fit', 'I_ISI_fit'],  # Priority 1
            ['baseline_fit'],  # Priority 2
            ['IBI_fitness', 'burst_frequency_fitness', 'big_burst_fit', 'small_burst_fit', 'thresh_fit', 'bimodal_burst_fit', 'slope_fit']  # Priority 3
        ]


        lowest_priority_achieved = None #5 is low, 1 is high
        for i, priority in enumerate(priorities, start=1):
            lowest_priority_achieved = i
            if any(fitnessVals[fit]['Fit'] == maxFitness for fit in priority):
                for lower_priority in priorities[i:]:
                    for fit in lower_priority:
                        fitnessVals[fit]['Fit'] = maxFitness
                        fitnessVals[fit]['deprioritized'] = True
                break

        return fitnessVals, lowest_priority_achieved
    def fitness_summary_metrics(fitnessVals):
        
        # Extract fitness values
        fitness_values = {key: fitnessVals[key]['Fit'] for key in fitnessVals if isinstance(fitnessVals[key], dict) and 'Fit' in fitnessVals[key]}

        # Average fitness
        fitness_values = [value for value in fitness_values.values() if value is not None]
        average_fitness = np.nansum(fitness_values) / len(fitness_values)

        # Calculate min and max values
        min_value = min(fitness_values)
        max_value = max(fitness_values)

        # Normalize the fitness values
        if max_value > min_value:
            normalized_fitness_values = [(value - min_value) / (max_value - min_value) for value in fitness_values]
            if any(np.isnan(value) for value in normalized_fitness_values):
                # Set all values to 1
                normalized_fitness_values = [1 for _ in fitness_values]
            # Scale the normalized values back to the original range (0-1000)
            #scaled_fitness_values = [value * 1000 for value in normalized_fitness_values]
            scaled_fitness_values = [value for value in normalized_fitness_values]
        else:
            scaled_fitness_values = [1 for _ in fitness_values] 

        # Calculate the average of the scaled fitness values
        avg_scaled_fitness = sum(scaled_fitness_values) / len(scaled_fitness_values)
        print(f'Average Fitness: {average_fitness}')
        print(f'Average Scaled Fitness: {avg_scaled_fitness}')

        return average_fitness, avg_scaled_fitness

    '''plotting functions'''
    def plot_network_activity(plotting_params, timeVector, firingRate, burstPeakTimes, burstPeakValues, thresholdBurst, rmsFiringRate, svg_mode = False): #rasterData, min_peak_distance = 1.0, binSize=0.02*1000, gaussianSigma=0.16*1000, thresholdBurst=1.2, figSize=(10, 6), saveFig=False, timeRange = None, figName='NetworkActivity.png'):
        #activate svg mode if specified
        if USER_svg_mode: svg_mode = True; print('SVG mode enabled')
        
        # Create a new figure with a specified size (width, height)
        figsize = plotting_params['figsize']
        assert figsize, 'figsize must be set to a tuple of two integers' #e.g. (10, 10)
        plt.figure(figsize=figsize)

        # Plot
        plt.subplot(1, 1, 1)
        plt.plot(timeVector, firingRate, color='black')
        plt.xlim([timeVector[0], timeVector[-1]])  # Restrict the plot to the first and last 100 ms   

        fig_ylim = plotting_params['ylim']
        if fig_ylim:       
            plt.ylim(fig_ylim)  # Set y-axis limits to min and max of firingRate
        else:
            yhigh100 = plotting_params['yhigh100']
            ylow100 = plotting_params['ylow100'] 
            assert yhigh100, 'USER_Activity_yhigh100 must be set to a float' #e.g. 1.05
            assert ylow100, 'USER_Activity_ylow100 must be set to a float' #e.g. 0.95
            plt.ylim([min(firingRate)*ylow100, max(firingRate)*yhigh100])  # Set y-axis limits to min and max of firingRate
        plt.ylabel('Spike Count')
        plt.xlabel('Time [s]')
        title_font = plotting_params['title_font']
        assert title_font, 'title_font must be set to an interger' #e.g. {'fontsize': 11}
        plt.title('Network Activity', fontsize=title_font)

        # Plot the threshold line and burst peaks
        plt.axhline(thresholdBurst * rmsFiringRate, color='gray', linestyle='--', label='Threshold')
        plt.plot(burstPeakTimes, burstPeakValues, 'or')  # Plot burst peaks as red circles

        default_name = 'NetworkActivity.png'
        figname = default_name

        if plotting_params['fitplot'] is not None:
            name = 'Network Activity - Fitness'
            #net_activity_metrics = plotting_params['net_activity_metrics']
            targets = plotting_params['targets']
            if targets is not None:
                peak_amp_target = targets['pops']['big_burst_target']['target']
                baseline_target = targets['pops']['baseline_target']['target']
                plt.axhline(peak_amp_target, color='r', linestyle='--', label='Burst Target')
                plt.axhline(baseline_target, color='b', linestyle='--', label='Baseline Target')
                plt.legend()
        
        saveFig = plotting_params['saveFig']
        if saveFig:
            assert saveFig, 'saveFig should be set to a relative path written as a string' #e.g. 'NERSC/plots/'
            batch_saveFolder = plotting_params['batch_saveFolder']
            assert batch_saveFolder, 'batch_saveFolder should be set to a relative path written as a string'
            simLabel = plotting_params['simLabel']
            assert simLabel, 'simLabel should be a string'
            #job_name = os.path.basename(batch_saveFolder)
            job_name = os.path.basename(os.path.dirname(batch_saveFolder))
            gen_folder = simLabel.split('_cand')[0]
            fig_path = os.path.join(saveFig, f'{job_name}/{gen_folder}/{simLabel}_{figname}')
            fig_dir = os.path.dirname(fig_path)
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            if 'output' in fig_path:
                print('')
            USER_fresh_plots = plotting_params['fresh_plots']
            if os.path.exists(fig_path) and USER_fresh_plots: pass
            elif os.path.exists(fig_path) and not USER_fresh_plots: 
                print(f'File {fig_path} already exists and USER_fresh_plots is set to False. Skipping plot.')
                return
            elif os.path.exists(fig_path) is False: pass
            else: raise ValueError(f'Idk how we got here. Logically.')
            #print(f'File {fig_path} already exists')
            plt.savefig(fig_path, bbox_inches='tight')
            print(f'Network Activity plot saved to {fig_path}')
        else:
            plt.show()
    def plot_network_activity_fitness(net_activity_metrics, svg_mode = False):
        #activate svg mode if specified
        if USER_svg_mode: svg_mode = True; print('SVG mode enabled')
        
        #net_activity_metrics = net_activity_metrics
        rasterData = simData.copy()
        if not exp_mode:
            rasterData['spkt'] = np.array(rasterData['spkt'])/1000
            rasterData['t'] = np.array(rasterData['t'])/1000
        assert USER_raster_convolve_params, 'USER_raster_convolve_params needs to be specified in USER_INPUTS.py'
        net_activity_params = USER_raster_convolve_params #{'binSize': .03*1000, 'gaussianSigma': .12*1000, 'thresholdBurst': 1.0}
        binSize = net_activity_params['binSize']
        gaussianSigma = net_activity_params['gaussianSigma']
        thresholdBurst = net_activity_params['thresholdBurst']
        min_peak_distance = net_activity_params['min_peak_distance']

        assert len(rasterData['spkt']) > 0, 'Error: rasterData has no elements. burstPeak, baseline, slopeFitness, and IBI fitness set to 1000.'                       
        # Generate the network activity plot with a size of (10, 5)
        plotting_params = None
        #if plot:
        assert USER_plotting_params is not None, 'USER_plotting_params must be set in USER_INPUTS.py'
        plotting_params = USER_plotting_params['NetworkActivity']
        plotting_params['simLabel'] = simLabel
        plotting_params['batch_saveFolder'] = batch_saveFolder
        print('plotting_params:', plotting_params)
        #sys.exit()
        #prep to plot fitplot
        below_baseline_target = kwargs['pops']['baseline_target']['target'] * 0.95
        above_amplitude_target = kwargs['pops']['big_burst_target']['target']
        max_fire_rate = max(net_activity_metrics['firingRate'])
        min_fire_rate = min(net_activity_metrics['firingRate'])
        yhigh = max(above_amplitude_target, max_fire_rate) * 1.05
        ylow = min(below_baseline_target, min_fire_rate) * 0.95
        plotting_params['ylim'] = [ylow, yhigh]
        plotting_params['fitnessVals'] = fitnessVals
        plotting_params['targets'] = kwargs
        plotting_params['fitplot'] = True
        plotting_params['net_activity_metrics'] = net_activity_metrics
        plotting_params['fresh_plots'] = USER_plotting_params['fresh_plots']
        plotting_params['figsize'] = USER_plotting_params['figsize']
        #make rectangle instead of square
        plotting_params['figsize'] = (plotting_params['figsize'][0], plotting_params['figsize'][1]/2)
        #if plot_save_path defined, replace saveFig with plot_save_path
        #plot_save_path = USER_plotting_path
        if plot_save_path is not None: plotting_params['saveFig'] = plot_save_path
        print('plot_save_path:', plot_save_path)
        net_metrics = measure_network_activity(
            rasterData, 
            binSize=binSize, 
            gaussianSigma=gaussianSigma, 
            thresholdBurst=thresholdBurst,
            min_peak_distance=min_peak_distance,
            plot=plot,
            plotting_params = plotting_params,
            crop = USER_raster_crop
        )
    def plot_raster(svg_mode = False):
        #activate svg mode if specified
        if USER_svg_mode: svg_mode = True; print('SVG mode enabled')
        
        #Attempt to generate the raster plot
        figname = 'raster_plot.png'
        timeVector = net_activity_metrics['timeVector']*1000 #convert back to ms
        timeRange = [timeVector[0], timeVector[-1]]
        #raster_plot_path = f'{batch_saveFolder}/{simLabel}_raster_plot.svg'
        job_name = os.path.basename(os.path.dirname(batch_saveFolder))
        #job_name = os.path.basename(batch_saveFolder)
        gen_folder = simLabel.split('_cand')[0]
        saveFig = USER_plotting_params['saveFig']
        #if plot_save_path defined, replace saveFig with plot_save_path
        print('plot_save_path:', plot_save_path)
        if plot_save_path is not None: saveFig = plot_save_path
        print('saveFig:', saveFig) 
        fig_path = os.path.join(saveFig, f'{job_name}/{gen_folder}/{simLabel}_{figname}')
        print('fig_path:', fig_path)
        USER_fresh_plots = USER_plotting_params['fresh_plots']
        fig_size = USER_plotting_params['figsize']
        #make rectangle instead of square
        fig_size = (fig_size[0], fig_size[1]/2)
        if os.path.exists(fig_path) and USER_fresh_plots: pass
        elif os.path.exists(fig_path) and not USER_fresh_plots: 
            print(f'File {fig_path} already exists and USER_fresh_plots is set to False. Skipping plot.')
            return
        elif os.path.exists(fig_path) is False: pass
        else: raise ValueError(f'Idk how we got here. Logically.')
        #Apply SVG mode
        if svg_mode: fig_path = fig_path.replace('.png', '.svg')
        
        try:
            netpyne.sim.analysis.plotRaster(saveFig=fig_path, 
                                        #timeRange = raster_activity_timeRange,
                                        timeRange = timeRange,
                                        showFig=False,
                                        labels = None, 
                                        figSize=fig_size)#, dpi=600)
            # #redo as png
            # cairosvg.svg2png(url=raster_plot_path, write_to=raster_plot_path.replace('.svg', '.png'))
            print('Raster plot saved to:', fig_path)
        except:
            print(f'Error generating raster plot from Data at: {data_file_path}')
            # raster_plot_path = None
            pass
    def most_active_time_range(timeVector, sim_obj):
            '''subfunc'''
            def electric_slide(time_points, voltage_trace):
                #print('Getting most active time range...')
                #spike threshold, anything above zero is a spike
                spike_threshold = 0
                
                # Define the window size and step size in milliseconds
                window_size = 1000  # 1 second
                step_size = 1  # 1 millisecond

                # Convert the time points to an array for easier indexing
                time_points = np.array(time_points)

                # Initialize the maximum spike count and the start time of the window with the maximum spike count
                max_spike_count = 0
                max_spike_start_time = None

                # Convert the voltage trace to an array once, outside the loop
                voltage_trace = np.array(voltage_trace)

                # Detect zero-crossings for the entire voltage trace
                zero_crossings = np.where(np.diff(np.sign(voltage_trace)))[0]
                zero_crossing_times = time_points[zero_crossings]

                # Initialize the maximum spike count and start time
                max_spike_count = 0
                max_spike_start_time = None

                # Slide the window over the voltage trace
                for start_time in np.arange(time_points[0], time_points[-1] - window_size + step_size, step_size):
                    # Get the end time of the current window
                    end_time = start_time + window_size

                    # Count the number of zero-crossings in the current window
                    spike_count = np.sum((zero_crossing_times >= start_time) & (zero_crossing_times < end_time))

                    # If the current window has more spikes than the previous maximum, update the maximum
                    if spike_count > max_spike_count:
                        max_spike_count = spike_count
                        max_spike_start_time = start_time
                
                #if no spikes are found, return full time range
                if max_spike_start_time is None:
                    return [0, time_points[-1]]                    
                # The time range with the most spiking activity is from max_spike_start_time to max_spike_start_time + window_size
                timeRange = [max_spike_start_time, max_spike_start_time + window_size]
                #if any values are < 0, make them 0
                return timeRange
            '''main'''
            # Get the time range of the most active part of the simulation for each neuron
            # Get the voltage trace for a specific cell
            # Get the keys (GIDs) of the neurons
            neuron_gids = list(sim_obj.allSimData['soma_voltage'].keys())
            time_points = sim_obj.allSimData['t']
            time_ranges = {}

            for gid in neuron_gids:
                # Get the voltage trace for the neuron
                voltage_trace = sim_obj.allSimData['soma_voltage'][gid]
                # Create a zip object from time_points and voltage trace
                pairs = zip(time_points, voltage_trace)
                # Filter voltage trace based on filtered time_points
                voltage_trace = [v for t, v in pairs if t >= timeVector[0] and t <= timeVector[-1]]
                # Filter time_points
                time_points_filtered = [t for t in time_points if t >= timeVector[0] and t <= timeVector[-1]]
                # Get time range
                time_range = electric_slide(time_points_filtered, voltage_trace)
                # Store time range in dictionary
                time_ranges[gid] = time_range

            return time_ranges
    def plot_trace_example(svg_mode = False):
        #activate svg mode if specified
        if USER_svg_mode: svg_mode = True; print('SVG mode enabled')
        # Attempt to generate sample trace for an excitatory example neuron
        try:
            figname = 'sample_trace'
            #job_name = os.path.basename(batch_saveFolder)
            job_name = os.path.basename(os.path.dirname(batch_saveFolder))
            gen_folder = simLabel.split('_cand')[0]
            saveFig = USER_plotting_params['saveFig']
            #if plot_save_path defined, replace saveFig with plot_save_path
            if plot_save_path is not None: saveFig = plot_save_path
            fig_path = os.path.join(saveFig, f'{job_name}/{gen_folder}/{simLabel}_{figname}')
            USER_fresh_plots = USER_plotting_params['fresh_plots']
            if os.path.exists(fig_path) and USER_fresh_plots: pass
            elif os.path.exists(fig_path) and not USER_fresh_plots: 
                print(f'File {fig_path} already exists and USER_fresh_plots is set to False. Skipping plot.')
                return
            elif os.path.exists(fig_path) is False: pass
            else: raise ValueError(f'Idk how we got here. Logically.')

            sim_obj = netpyne.sim
            timeVector = np.array(net_activity_metrics['timeVector']*1000) #convert back to ms
            print('Getting most active time ranges for sample traces...')
            timeRanges = most_active_time_range(timeVector, sim_obj)
            #timeRanges = [excite_timeVector, inhib_timeVector]
            #titles = ['E0_highFR', 'I0_highFR']
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            # Create individual plots and save as PNG
            num_cells = len(timeRanges)
            fig_height_per_cell = USER_plotting_params['figsize'][1] / num_cells
            titles = []

            for gid, timeRange in timeRanges.items():
                #include = [gid]
                gid_num = int(gid.split('_')[-1])
                if gid_num < 400 * .3: 
                    title = f'{gid}_excitatory'
                    type = 'E'
                    num = gid_num
                else: 
                    title = f'{gid}_inibitory'
                    type = 'I'
                    num = gid_num - 400 * .3
                titles.append(title)
                include =[type, num]
                sample_trace = sim_obj.analysis.plotTraces(
                    include=include,
                    overlay=True,
                    oneFigPer='trace',
                    title=title,
                    timeRange=timeRange,
                    showFig=False,
                    figSize=(USER_plotting_params['figsize'][0], fig_height_per_cell)
                )
                fig = sample_trace[0]['_trace_soma_voltage']
                fig.suptitle(title)
                fig.tight_layout(rect=[0, 0.03, 1, 1])
                fig_path_path = f'{fig_path}_{title}.png'
                fig.savefig(fig_path_path)

            fig, axs = plt.subplots(num_cells, 1, figsize=USER_plotting_params['figsize'])
            for i, title in enumerate(titles):
                img = mpimg.imread(f'{fig_path}_{title}.png')
                axs[i].imshow(img)
                axs[i].axis('off')

            fig.tight_layout(rect=[0, 0.03, 1, 1])
            fig.savefig(f'{fig_path}_combined.png')
            print(f'Sample trace plot saved to {fig_path}_combined.png')
            #fig.suptitle('Middlemost 1 second of simulation')
            # Save the figure with the title
            #fig.savefig(sample_trace_path_E)
            # redo as png
            #cairosvg.svg2png(url=sample_trace_path_E, write_to=sample_trace_path_E.replace('.svg', '.png'))
        except:
            print(f'Error generating sample trace plot from Data at: {data_file_path}')
            #sample_trace_path_E = None
            pass
    def plot_connections(svg_mode = False):
        #activate svg mode if specified
        if USER_svg_mode: svg_mode = True; print('SVG mode enabled')
        
        # Attempt to generate sample trace for an excitatory example neuron
        try:
            print('Generating connections plot...')
            figname = 'connections'
            # job_name = os.path.basename(batch_saveFolder)
            job_name = os.path.basename(os.path.dirname(batch_saveFolder))
            gen_folder = simLabel.split('_cand')[0]
            saveFig = USER_plotting_params['saveFig']
            #if plot_save_path defined, replace saveFig with plot_save_path
            if plot_save_path is not None: saveFig = plot_save_path
            fig_path = os.path.join(saveFig, f'{job_name}/{gen_folder}/{simLabel}_{figname}')
            print('fig_path:', fig_path)
            #sys.exit()
            USER_fresh_plots = USER_plotting_params['fresh_plots']
            if os.path.exists(fig_path) and USER_fresh_plots: pass
            elif os.path.exists(fig_path) and not USER_fresh_plots: 
                print(f'File {fig_path} already exists and USER_fresh_plots is set to False. Skipping plot.')
                return
            elif os.path.exists(fig_path) is False: pass
            else: raise ValueError(f'Idk how we got here. Logically.')
            #Apply SVG mode
            if svg_mode: fig_path = fig_path.replace('.png', '.svg')

            sim_obj = netpyne.sim
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            # Create individual plots and save as PNG
            sim_obj.analysis.plot2Dnet(saveFig=fig_path, showFig=False, showConns=True, figSize=(USER_plotting_params['figsize'][0], USER_plotting_params['figsize'][1]/2))

        except:
            print(f'Error generating connections from Data at: {data_file_path}')
            #sample_trace_path_E = None
            pass
    
    '''main functions'''
    def get_fitness(**kwargs):
        '''init'''
        def save_fitness_results():
            ##save fitness values
            maxFitness = kwargs['maxFitness']
            fitnessResults = {}
            for key, value in fitnessVals.items():
                fitnessResults[f'{key}'] = value
            fitnessResults['maxFitness'] = maxFitness
            fitnessResults['average_fitness'] = average_fitness
            fitnessResults['average_scaled_fitness'] = avg_scaled_fitness


            if exp_mode:
                #gen_folder = 'exp_data_fit'
                destination = os.path.join(output_path, f'{simLabel}_Fitness.json')
                destination_dir = os.path.dirname(destination)
                if not os.path.exists(destination_dir): os.makedirs(destination_dir)
                with open(f'{output_path}/{simLabel}_Fitness.json', 'w') as f:
                    json.dump(fitnessResults, f, indent=4)
                print(f'Experimental Fitness results saved to {output_path}/{simLabel}_Fitness.json')
                return

            gen_folder = simLabel.split('_cand')[0]
            if fitness_save_path is None and batch_mode:
                #typical case, during simulations
                with open(f'{output_path}/{gen_folder}/{simLabel}_Fitness.json', 'w') as f:
                    json.dump(fitnessResults, f, indent=4)
                print(f'Fitness results saved to {output_path}/{simLabel}_Fitness.json')
            elif fitness_save_path is None and data_file_path is not None:
                #during re-eval of fitness without plotting
                with open(data_file_path.replace('_data', '_Fitness'), 'w') as f:
                    json.dump(fitnessResults, f, indent=4)
                print(f'Fitness results saved to {data_file_path.replace("_data", "_Fitness")}')
            else:
                assert data_file_path is not None, 'data_file_path must be specified to save fitness results'
                #while plotting
                with open(data_file_path.replace('_data', '_Fitness'), 'w') as f:
                    json.dump(fitnessResults, f, indent=4)
                print(f'Fitness results saved to {data_file_path.replace("_data", "_Fitness")}')
                # with open(f'{fitness_save_path}/{simLabel}_Fitness.json', 'w') as f:
                #     json.dump(fitnessResults, f, indent=4)
        output_path = batch_saveFolder

        '''get network activity metrics'''
        print(f'Calculating net_actiity_metrics...')
        net_activity_metrics = get_network_activity_metrics()

        '''get individual neuron metrics'''
        print(f'Calculating individual neuron metrics...')
        neuron_metrics = get_individual_neuron_metrics()

        '''fitness values'''
        fitnessVals = {}
        fitnessVals = {
            #Priority 1
            'E_rate_fit': fit_E_firing_rate(neuron_metrics, **kwargs),
            'I_rate_fit': fit_I_firing_rate(neuron_metrics, **kwargs),
            'E_ISI_fit': fit_E_ISI(neuron_metrics, **kwargs),
            'I_ISI_fit': fit_I_ISI(neuron_metrics, **kwargs),
            #Priority 2
            'baseline_fit': fit_baseline(net_activity_metrics, **kwargs),
            #Priority 3
            'IBI_fitness': fit_IBI(net_activity_metrics, **kwargs),
            'burst_frequency_fitness': fit_burst_frequency(net_activity_metrics, **kwargs),
            #Priority 4
            'big_burst_fit': fit_big_burst_amplitude(net_activity_metrics, **kwargs),
            'small_burst_fit': fit_small_burst_amplitude(net_activity_metrics, **kwargs),
            'thresh_fit': fit_threshold(net_activity_metrics, **kwargs),
            #Priority 5
            'bimodal_burst_fit': fit_bimodal_burst_amplitude(net_activity_metrics, **kwargs),
            #'sustain_fit': fit_sustain(net_activity_metrics, **kwargs),
            'slope_fit': fit_rate_slope(net_activity_metrics, **kwargs),
        }        

        #prioritize and report summary fitness metrics
        prioritize = True
        if prioritize: fitnessVals, priority_level = prioritize_fitness(fitnessVals, **kwargs)
        average_fitness, avg_scaled_fitness = fitness_summary_metrics(fitnessVals)
        save_fitness_results()
        return average_fitness, avg_scaled_fitness, fitnessVals, net_activity_metrics, priority_level
    def plot_fitness(fitnessVals, net_activity_metrics, simData, plot_save_path, **kwargs):
        print('Plotting fitness...')
        
        '''handle plotting sim or experimental data'''
        if not exp_mode: load_clean_sim_object(data_file_path)       
        if exp_mode: 
            output_path = os.path.abspath(batch_saveFolder) #check this again in experimental data context.
            plot_save_path = f'{output_path}'
        
        '''plotting'''
        plot_network_activity_fitness(net_activity_metrics)
        if not exp_mode: plot_raster()
        if not exp_mode: plot_trace_example()
        if not exp_mode: plot_connections()
        #sys.exit()
    
    '''
    Get Fitness (Main Fitness Function)
    '''    
    ##find relevant batch object in call stack
    # Use the function to get the Batch object and simLabel
    if exp_mode:
        batch_saveFolder = os.path.join(fitness_save_path, 'experimental_data') #yea this is weird. I'll fix it later.
        simLabel = "experimental_data"
    if batch_saveFolder is None and simLabel is None:
        batch, simLabel = find_batch_object_and_sim_label()
        batch_saveFolder = batch.saveFolder
        gen_folder = simLabel.split('_cand')[0]
        data_file_path = os.path.join(batch_saveFolder, gen_folder, simLabel+'_data.json')
        batch_mode = True #typically mode during batch simulations
    assert simLabel is not None, "SimLabel undefined."
    assert batch_saveFolder is not None, "Batch save folder undefined."
    assert data_file_path is not None, "Data file path undefined."

    ##get fitness
    average_fitness, avg_scaled_fitness, fitnessVals, net_activity_metrics, priority_level = get_fitness(**kwargs)
    #print(f'Fitness results saved to {output_path}/{gen_folder}/{simLabel}_Fitness.json')
    if plot is None: plot = USER_plot_fitness
    if priority_level < 3: plot = False #if priority level is 1 or 2, FRs, ISIs, and/or baseline not good. Not worth looking at.
    if plot: plot_fitness(fitnessVals, net_activity_metrics, simData, plot_save_path, **kwargs)
    #return avg_scaled_fitness
    return average_fitness