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
from workspace.RBS_network_simulations._archive.temp_user_args import *
from helper_functions import load_clean_sim_object, find_batch_object_and_sim_label
from plotting_functions import plot_network_activity_fitness, plot_raster, plot_trace_example, plot_connections
from analysis_functions import get_network_activity_metrics, get_individual_neuron_metrics

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
        mean_value = np.mean(np.nan_to_num(burstPeakValues))
        stdev_value = np.std(np.nan_to_num(burstPeakValues))
        kurtosis_value = stats.kurtosis(np.nan_to_num(burstPeakValues))
        skew_value = stats.skew(np.nan_to_num(burstPeakValues))

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
        combined_fitness = np.nanmean([ratio_fitness, mean_fitness, stdev_fitness, kurtosis_fitness, skew_fitness])
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
        overall_fitness = np.nanmean([IBI_fitness, stdev_fitness, skew_fitness, kurt_fitness])

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

    print('Prioritized.')
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

'''Main fitness function'''
def fitnessFunc(simData, plot = None, simLabel = None, data_file_path = None, batch_saveFolder = None, fitness_save_path = None, plot_save_path = None, exp_mode = False, **kwargs):   
   
    '''sub functions'''
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
        net_activity_metrics = get_network_activity_metrics(simData, exp_mode=exp_mode)

        '''get individual neuron metrics'''
        print(f'Calculating individual neuron metrics...')
        neuron_metrics = get_individual_neuron_metrics(data_file_path, exp_mode=exp_mode)

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
        return average_fitness, avg_scaled_fitness, fitnessVals, net_activity_metrics, neuron_metrics, priority_level
    def plot_fitness(fitnessVals, net_activity_metrics, neuron_metrics, simData, plot_save_path, simLabel, batch_saveFolder, **kwargs):
        print('Plotting fitness...')
        
        '''handle plotting sim or experimental data'''
        if not exp_mode: load_clean_sim_object(data_file_path)       
        if exp_mode: 
            output_path = os.path.abspath(batch_saveFolder) #check this again in experimental data context.
            plot_save_path = f'{output_path}'
        
        '''plotting'''
        plot_network_activity_fitness(simData, net_activity_metrics, simLabel, batch_saveFolder, plot_save_path, fitnessVals, exp_mode=exp_mode, **kwargs)
        print(f'Fitness plots saved to {plot_save_path}/{simLabel}_Fitness')

        if not exp_mode: plot_raster(net_activity_metrics, plot_save_path, batch_saveFolder, simLabel, data_file_path)
        if not exp_mode: plot_trace_example(neuron_metrics, net_activity_metrics, plot_save_path, batch_saveFolder, simLabel, data_file_path)
        if not exp_mode: plot_connections(plot_save_path, batch_saveFolder, simLabel, data_file_path)
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
    average_fitness, avg_scaled_fitness, fitnessVals, net_activity_metrics, neuron_metrics, priority_level = get_fitness(**kwargs)
    #print(f'Fitness results saved to {output_path}/{gen_folder}/{simLabel}_Fitness.json')
    if plot is None: plot = USER_plot_fitness
    if priority_level < 3: plot = False; print(f'Simulation {simLabel} priority level is {priority_level}. Not worth plotting.') #if priority level is 1 or 2, FRs, ISIs, and/or baseline not good. Not worth looking at.
    if plot: 
        plot_fitness(fitnessVals, net_activity_metrics, neuron_metrics, simData, plot_save_path, simLabel, batch_saveFolder, **kwargs)
    #return avg_scaled_fitness
    return average_fitness