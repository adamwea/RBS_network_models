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
import netpyne

# Import USER_INPUTS
from USER_INPUTS import *

def fitnessFunc(simData, plot = False, simLabel = None, data_file_path = None, batch_saveFolder = None, fitness_save_path = None, plot_save_path = None, exp_mode = False, **kwargs):   

    ''' subfuncs '''
    def set_all_fitness_to_max():
        fitnessVals = {}
        maxFitness = kwargs['maxFitness']
        fitnessVals['burstAmp_Fitness'] = {'Value': None, 'Fit': maxFitness}
        fitnessVals['IBI_fitness'] = {'Value': None, 'Fit': maxFitness}
        fitnessVals['baselineFitness'] = {'Value': None, 'Fit': maxFitness}
        fitnessVals['slopeFitness'] = {'Value': None, 'Fit': maxFitness}
        #fitnessVals['baseline_diff_fitness'] = {'Value': None, 'Fit': maxFitness}
        fitnessVals['burst_peak_frequency_fitness'] = {'Value': None, 'Fit': maxFitness}
        #fitnessVals['rate_fitness'] = {'Value': None, 'Fit': maxFitness}
        fitnessVals['E_rate_fitness'] = {'Value': None, 'Fit': maxFitness}
        fitnessVals['I_rate_fitness'] = {'Value': None, 'Fit': maxFitness}
        fitnessVals['sustain_oscillation_fitness'] = {'Value': None, 'Fit': maxFitness}
        return fitnessVals
    def get_network_activity_metrics(fitnessVals, plot = False):
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
            #experimental data fitting mode
            # if exp_mode: 
            #     rasterData = {}
            #     rasterData['spkt'] = simData
            assert len(rasterData['spkt']) > 0, 'Error: rasterData has no elements. burstPeak, baseline, slopeFitness, and IBI fitness set to 1000.'                       
            # Generate the network activity plot with a size of (10, 5)
            plotting_params = None
            if plot:
                assert USER_plotting_params is not None, 'USER_plotting_params must be set in USER_INPUTS.py'
                plotting_params = USER_plotting_params['NetworkActivity']
                plotting_params['simLabel'] = simLabel
                plotting_params['batch_saveFolder'] = batch_saveFolder
                plotting_params['fresh_plots'] = USER_plotting_params['fresh_plots']
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
            fitnessVals = set_all_fitness_to_max()            
            return fitnessVals, net_activity_metrics            
        
        #net_activity_metrics = {}
        net_activity_metrics['burstPeakValues'] = net_metrics['burstPeakValues']
        net_activity_metrics['IBIs'] = net_metrics['IBIs']
        net_activity_metrics['baseline'] = net_metrics['baseline']
        #net_activity_metrics['baselineDiff'] = net_metrics['baseline_diff']
        net_activity_metrics['normalizedPeakVariance'] = net_metrics['normalized_peak_variance']
        net_activity_metrics['peakFreq'] = net_metrics['peak_freq']
        net_activity_metrics['firingRate'] = net_metrics['firingRate']
        net_activity_metrics['burstPeakTimes'] = net_metrics['burstPeakTimes']
        net_activity_metrics['timeVector'] = net_metrics['timeVector']
        net_activity_metrics['threshold'] = net_metrics['threshold']
        net_activity_metrics['sustained_oscillation'] = net_metrics['sustain']

        return fitnessVals, net_activity_metrics
    def plot_burst_freq_and_IBI(fitnessVals):
        rasterData = simData.copy()
        if not exp_mode:
            rasterData['spkt'] = np.array(rasterData['spkt'])/1000
            rasterData['t'] = np.array(rasterData['t'])/1000
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
            #prep to plot fitplot
            plotting_params['ylim'] = None
            plotting_params['fitplot'] = 'burst_freq'
            plotting_params['fitnessVals'] = fitnessVals
            plotting_params['targets'] = kwargs
        net_metrics = measure_network_activity(
            rasterData, 
            binSize=binSize, 
            gaussianSigma=gaussianSigma, 
            thresholdBurst=thresholdBurst,
            plot=plot,
            plotting_params = plotting_params,
            crop = USER_raster_crop,
        )  
    def fit_burst_freq(net_activity_metrics, fitnessVals, plot = False, **kwargs):
        
        ##
        try:
            #burst_peak_frequency Fitness
            burst_peak_frequency = net_activity_metrics['peakFreq']
            pops = kwargs['pops']
            pops_frequency = pops['burst_peak_frequency']
            maxFitness = kwargs['maxFitness']
            # Calculate the fitness as the absolute difference between the frequency and the target frequency
            burst_peak_frequency_fitness = [min(np.exp(
                abs(pops_frequency['target'] - burst_peak_frequency)), maxFitness) 
                if burst_peak_frequency > pops_frequency['min'] and burst_peak_frequency < pops_frequency['max'] else maxFitness]
            burst_peak_frequency_fitness = burst_peak_frequency_fitness[0]
            # Print the frequency and its fitness
            print('Frequency of burst peaks: %.3f, Fitness: %.3f' % (burst_peak_frequency, burst_peak_frequency_fitness))
            fitnessVals['burst_peak_frequency_fitness'] = {'Value': burst_peak_frequency, 'Fit': burst_peak_frequency_fitness}
            #if plot: plot_burst_freq(fitnessVals)
            return fitnessVals
        except Exception as e:
            print(f'Error calculating burst peak frequency fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            fitnessVals['burst_peak_frequency_fitness'] = {'Value': None, 'Fit': maxFitness}
            return fitnessVals  
    def fit_sustain(net_activity_metrics, fitnessVals, plot = False, **kwargs):
        try:
            # Sustain Fitness
            pops = kwargs['pops']
            maxFitness = kwargs['maxFitness']
            sustained_osci100 = net_activity_metrics['sustained_oscillation']
            sustained_osci_target = pops['sustained_osci']

            # Calculate the fitness as the absolute difference between the sustain duration and the target sustain duration
            sustain_fit = min(np.exp((sustained_osci_target['target'] - sustained_osci100)), maxFitness
                             ) # if sustained_osci100 > sustained_osci_target['min'] else maxFitness

            # Print the sustain duration and its fitness
            print('Percent Duration: %.3f, Fitness: %.3f' % (sustained_osci100, sustain_fit))
            fitnessVals['sustain_oscillation_fitness'] = {'Value': sustained_osci100, 'Fit': sustain_fit}
            #if plot: plot_sustain(fitnessVals)
            return fitnessVals
        except Exception as e:
            print(f'Error calculating sustain fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            fitnessVals['sustain_oscillation_fitness'] = {'Value': None, 'Fit': maxFitness}
            return fitnessVals
    def fit_thresh(net_activity_metrics, fitnessVals, plot = False, **kwargs):
        try:
            pops = kwargs['pops']
            maxFitness = kwargs['maxFitness']
            thresh = net_activity_metrics['threshold']
            thresh_target = pops['thresh_target']

            # Calculate the fitness as the absolute difference between the threshold and the target threshold
            thresh_fit = [min(np.exp((thresh_target['target'] - thresh)), maxFitness) 
                          if thresh < thresh_target['max'] else maxFitness]
            thresh_fit = thresh_fit[0]

            # Print the threshold and its fitness
            print('Thresh: %.3f, Fitness: %.3f' % (thresh, thresh_fit))
            fitnessVals['thresh'] = {'Value': thresh, 'Fit': thresh_fit}
            return fitnessVals
        except Exception as e:
            print(f'Error calculating thresh fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            fitnessVals['thresh'] = {'Value': None, 'Fit': maxFitness}
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
            slopeFitness = min(np.exp(abs(rate_slope['target'] - slope)) 
                               #if abs(slope) < rate_slope['max'] else maxFitness
                               , maxFitness)

            # Print the slope and its fitness
            print('Slope of firing rate: %.3f, Fitness: %.3f' % (slope, slopeFitness))
            fitnessVals['slopeFitness'] = {'Value': slope, 'Fit': slopeFitness}
            return fitnessVals
        except Exception as e:
            print(f'Error calculating firing rate slope fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            fitnessVals['slopeFitness'] = {'Value': None, 'Fit': maxFitness}
            return fitnessVals
    def plot_burst_peak_and_baseline(fitnessVals):
        rasterData = simData.copy()
        if not exp_mode:
            rasterData['spkt'] = np.array(rasterData['spkt'])/1000
            rasterData['t'] = np.array(rasterData['t'])/1000
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
            #prep to plot fitplot
            plotting_params['ylim'] = None
            plotting_params['fitplot'] = 'burst_peak'
            plotting_params['fitnessVals'] = fitnessVals
            plotting_params['targets'] = kwargs
        net_metrics = measure_network_activity(
            rasterData, 
            binSize=binSize, 
            gaussianSigma=gaussianSigma, 
            thresholdBurst=thresholdBurst,
            plot=plot,
            plotting_params = plotting_params,
            crop = USER_raster_crop,
        )  
    def fit_burst_peak(net_activity_metrics, fitnessVals, plot = False, **kwargs):
        burstPeakValues = net_activity_metrics['burstPeakValues']
        pops = kwargs['pops']
        maxFitness = kwargs['maxFitness']
        cutoff = pops['burts_peak_targets']['cutoff']
        try: 
            assert len(burstPeakValues) > 0, 'Error: burstPeakValues has no elements. BurstVal_fitness set to maxFitness.'
            # Big bursts peak fitness
            pops_peaks = pops['burts_peak_targets']['big_bursts']            
            max_burst = pops_peaks['max']
            min_burst = cutoff
            width = pops_peaks['width']
            big_bursts = [value for value in burstPeakValues if value > cutoff]
            popFitnessBurstPeak = [
                min(np.exp(abs(pops_peaks['target'] - value) / width), maxFitness)
                if value < max_burst else maxFitness for value in big_bursts
            ]
            # Calculate the mean fitness
            Big_BurstVal_fitness = np.mean(popFitnessBurstPeak)
            # Calculate the average burstPeak
            avg_burstPeak = np.mean(big_bursts)
            fitnessVals['BigBurstVal_Fitness'] = {'Value': avg_burstPeak, 'Fit': Big_BurstVal_fitness}
            print('Max Burst Peak: %.3f' % (np.max(big_bursts)))
            print('Min Burst Peak: %.3f' % (np.min(big_bursts)))
            print('Big Burst Peak: %.3f, Fitness: %.3f' % (avg_burstPeak, Big_BurstVal_fitness))
        except Exception as e:
            print(f'Error calculating big burst peak fitness.')
            print(f'Error: {e}')
            fitnessVals['BigBurstVal_Fitness'] = {'Value': None, 'Fit': maxFitness}
            print('Big Burst Peak: %.3f, Fitness: %.3f' % (None, maxFitness))

        try:
            assert len(burstPeakValues) > 0, 'Error: burstPeakValues has no elements. BurstVal_fitness set to maxFitness.'
            # Number of big bursts
            percent_big_bursts = len(big_bursts)/len(burstPeakValues) * 100
            #percent_width = 1+pops_peaks['num_width']
            #width = pops_peaks['num_target']*percent_width
            num_fitness = [
                min(np.exp(abs(pops_peaks['num_target'] - percent_big_bursts)), maxFitness) 
                if percent_big_bursts >= pops_peaks['num_min'] else maxFitness]
            num_fitness = num_fitness[0]
            fitnessVals['numBig_Fitness'] = {'Value': len(big_bursts), 'Percent':percent_big_bursts, 'Fit': num_fitness}
            print('Number of Big Bursts: %.3f, Percentage: %.3f%%, Fitness: %.3f' % (len(big_bursts), percent_big_bursts, num_fitness))        
        except Exception as e:
            print(f'Error calculating number of big bursts fitness.')
            print(f'Error: {e}')
            fitnessVals['numBig_Fitness'] = {'Value': None, 'Fit': maxFitness}
            print('Number of Big Bursts: %.3f, Fitness: %.3f' % (None, maxFitness))

        try:
            assert len(burstPeakValues) > 0, 'Error: burstPeakValues has no elements. BurstVal_fitness set to maxFitness.'
            # Small bursts peak fitness
            pops_peaks = pops['burts_peak_targets']['lil_bursts']            
            max_burst = cutoff
            min_burst = pops_peaks['min']
            width = pops_peaks['width']
            small_bursts = [value for value in burstPeakValues if value <= cutoff]            
            popFitnessBurstPeak = [
                min(np.exp(abs(pops_peaks['target'] - value) / width), maxFitness)
                if value > min_burst else maxFitness for value in small_bursts
            ]
            # Calculate the mean fitness
            Small_BurstVal_fitness = np.mean(popFitnessBurstPeak)
            # Calculate the average burstPeak
            avg_burstPeak = np.mean(small_bursts)
            fitnessVals['SmallBurstVal_Fitness'] = {'Value': avg_burstPeak, 'Fit': Small_BurstVal_fitness}
            print('Max Burst Peak: %.3f' % (np.max(small_bursts)))
            print('Min Burst Peak: %.3f' % (np.min(small_bursts)))
            print('Small Burst Peak: %.3f, Fitness: %.3f' % (avg_burstPeak, Small_BurstVal_fitness))
        except Exception as e:
            print(f'Error calculating small burst peak fitness.')
            print(f'Error: {e}')
            fitnessVals['SmallBurstVal_Fitness'] = {'Value': None, 'Fit': maxFitness}
            print(f'Small Burst Peak: {None}, Fitness: %.3f' % (maxFitness,))

        try:
            assert len(burstPeakValues) > 0, 'Error: burstPeakValues has no elements. BurstVal_fitness set to maxFitness.'
            # Number of small bursts
            percent_small_bursts = len(small_bursts)/len(burstPeakValues) * 100
            #width = pops_peaks['num_width']
            num_fitness = [
                min(np.exp(abs(pops_peaks['num_target'] - percent_small_bursts)), maxFitness) 
                if percent_small_bursts >= pops_peaks['num_min'] else maxFitness]
            num_fitness = num_fitness[0]
            fitnessVals['numSmall_Fitness'] = {'Value': len(small_bursts), 'Percent':percent_small_bursts, 'Fit': num_fitness}
            print('Number of Small Bursts: %.3f, Percentage: %.3f%%, Fitness: %.3f' % (len(small_bursts), percent_small_bursts, num_fitness))        
        except Exception as e:
            print(f'Error calculating number of small bursts fitness.')
            print(f'Error: {e}')
            fitnessVals['numSmall_Fitness'] = {'Value': None, 'Fit': maxFitness}
            print('Number of Small Bursts: %.3f, Fitness: %.3f' % (None, maxFitness))

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

            # Calculate average IBI
            avg_IBI = np.mean(IBIs)

            # Print average IBI and its fitness
            print('Average IBI: %.3f, Fitness: %.3f' % (avg_IBI, IBI_fitness))
            fitnessVals['IBI_fitness'] = {'Value': avg_IBI, 'Fit': IBI_fitness}
            return fitnessVals
        except Exception as e:
            print(f'Error calculating IBI fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            fitnessVals['IBI_fitness'] = {'Value': None, 'Fit': maxFitness}
            return fitnessVals
    def fit_baseline(net_activity_metrics, fitnessVals, plot = False, **kwargs):
        try:
            assert net_activity_metrics['baseline'] is not None, 'Error: baseline is None. Baseline could not be calculated.'
            pops = kwargs['pops']
            pops_baseline = pops['baseline_targets']
            maxFitness = kwargs['maxFitness']
            baseline = net_activity_metrics['baseline']
            baseline_target = pops_baseline['target']
            #baseline_width = pops_baseline['width']
            baseline_max = pops_baseline['max']

            baselineFitness = [min(np.exp(abs(baseline_target - baseline)), maxFitness) 
                               if baseline < baseline_max else maxFitness]
            baselineFitness = baselineFitness[0]

            #print average baseline and its fitness
            print('baseline: %.3f, Fitness: %.3f' % (baseline, baselineFitness)) 
            fitnessVals['baseline_fitness'] = {'Value': baseline, 'Fit': baselineFitness}
            return fitnessVals
        except Exception as e:
            print(f'Error calculating baseline fitness.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            fitnessVals['baseline_fitness'] = {'Value': None, 'Fit': maxFitness}
            return fitnessVals
    def fit_firing_rate(net_activity_metrics, fitnessVals, simData, plot = False, **kwargs):
        try:
            pops = kwargs['pops']
            pops_rate = pops['rate_targets']
            maxFitness = kwargs['maxFitness']
            popFitness = [None for i in pops.items()]
            # This is a list comprehension that calculates the fitness for each population in the simulation.
            popFitness = [
                min(np.exp(abs(v['target'] - simData['popRates'][k])), maxFitness)
                if simData["popRates"][k] > v['min'] else maxFitness
                for k, v in pops_rate.items()
            ]
            E_rate_fitness = popFitness[0]
            popInfo_E = '; '.join(['%s rate=%.10f fit=%.3f'%(p,r,f) for p,r,f in zip(
                list(simData['popRates'].keys()), 
                list(simData['popRates'].values()), popFitness) if 'E' in p])
            print('  Excitatory: '+popInfo_E)

            I_rate_fitness = popFitness[1]
            popInfo_I = '; '.join(['%s rate=%.10f fit=%.3f'%(p,r,f) for p,r,f in zip(
                list(simData['popRates'].keys()), 
                list(simData['popRates'].values()), popFitness) if 'I' in p])
            print('  Inhibitory: '+popInfo_I)
            fitnessVals['E_rate_fitness'] = {'Value': simData['popRates']['E'], 'Fit': E_rate_fitness}
            fitnessVals['I_rate_fitness'] = {'Value': simData['popRates']['I'], 'Fit': I_rate_fitness}
            return fitnessVals
        except Exception as e:
            print(f'Error calculating firing rate fitness.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            #fitnessVals['rate_fitness'] = {'Value': None, 'Fit': maxFitness}
            fitnessVals['E_rate_fitness'] = {'Value': None, 'Fit': maxFitness}
            fitnessVals['I_rate_fitness'] = {'Value': None, 'Fit': maxFitness}
            return fitnessVals
    def fitness_summary_metrics(fitnessVals):
        # Extract fitness values
        fitness_values = {key: fitnessVals[key]['Fit'] for key in fitnessVals if isinstance(fitnessVals[key], dict) and 'Fit' in fitnessVals[key]}

        # Average fitness
        fitness_values = [value for value in fitness_values.values() if value is not None]
        average_fitness = sum(fitness_values) / len(fitness_values)

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
            scaled_fitness_values = [1000 for _ in fitness_values] 

        # Calculate the average of the scaled fitness values
        avg_scaled_fitness = sum(scaled_fitness_values) / len(scaled_fitness_values)
        print(f'Average Fitness: {average_fitness}')
        print(f'Average Scaled Fitness: {avg_scaled_fitness}')

        return average_fitness, avg_scaled_fitness
    def get_fitness(simData, plot = False, **kwargs):
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
                with open(f'{output_path}/{simLabel}_Fitness.json', 'w') as f:
                    json.dump(fitnessResults, f, indent=4)
                print(f'Experimental Fitness results saved to {output_path}/{simLabel}_Fitness.json')
                return

            gen_folder = simLabel.split('_cand')[0]
            if fitness_save_path is None and data_file_path is None:
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

        
        maxFitness = kwargs['maxFitness']
        fitnessVals = {}
        ## Get the fitness function arguments
        output_path = batch_saveFolder
        print(f'Calculating net_actiity_metrics...')
        assert USER_plot_fitness_bool is not None, 'USER_plot_fitness_bool must be set in USER_INPUTS.py'
        fitnessVals, net_activity_metrics = get_network_activity_metrics(fitnessVals, plot = None)
        if net_activity_metrics == {}:
            # Get the fitness summary metrics        
            average_fitness, avg_scaled_fitness = fitness_summary_metrics(fitnessVals)
            # Save the fitness results
            save_fitness_results()
            return maxFitness

        ## Get the fitness values
        # Get burst peak fitness, optionally plot
        print('Calculating burst peak fit...')
        fitnessVals = fit_burst_peak(net_activity_metrics, fitnessVals, plot = None, **kwargs)
        # Get burst freq fitness, optionally plot
        print('Calculating burst frequency fit...')
        fitnessVals = fit_burst_freq(net_activity_metrics, fitnessVals, plot = None, **kwargs)       
        # Get IBI fitness, optionally plot
        print('Calculating IBI fit...')
        fitnessVals = fit_IBI(net_activity_metrics, fitnessVals, plot = None, **kwargs)
        # Get baseline fitness, optionally plot
        print('Calculating baseline fit...')
        fitnessVals = fit_baseline(net_activity_metrics, fitnessVals, plot = None, **kwargs)
        # Get rate slope fitness, optionally plot
        print('Calculating rate slope fit...')
        fitnessVals = fit_rate_slope(net_activity_metrics, fitnessVals, plot = None, **kwargs)
        #Get baseline fitness, optionally plot
        print('Calculating thresh fit...')
        fitnessVals = fit_thresh(net_activity_metrics, fitnessVals, plot = None, **kwargs) 
        #Get baseline fitness, optionally plot
        print('Calculating sustain fit...')
        fitnessVals = fit_sustain(net_activity_metrics, fitnessVals, plot = None, **kwargs)        
        # Get firing rate fitness, optionally plot
        print('Calculating firing rate fit...')
        fitnessVals = fit_firing_rate(net_activity_metrics, fitnessVals, simData, plot = None, **kwargs)

        # Get the fitness summary metrics        
        average_fitness, avg_scaled_fitness = fitness_summary_metrics(fitnessVals)
        # Save the fitness results
        save_fitness_results()
        
        #print(f'Fitness results saved to {output_path}/{gen_folder}/{simLabel}_Fitness.json')
        if plot: plot_fitness(fitnessVals, simData, net_activity_metrics, plot_save_path, **kwargs)
        return average_fitness, avg_scaled_fitness
    def plot_fitness(fitnessVals, simData, net_activity_metrics, plot_save_path, **kwargs):
        print('Plotting fitness...')
        if not exp_mode: netpyne.sim.loadAll(data_file_path)        
        if exp_mode: 
            output_path = os.path.abspath(batch_saveFolder)
            plot_save_path = f'{output_path}'
        
        '''plotting functions'''
        def plot_network_activity_fitness(net_activity_metrics):
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
            assert len(rasterData['spkt']) > 0, 'Error: rasterData has no elements. burstPeak, baseline, slopeFitness, and IBI fitness set to 1000.'                       
            # Generate the network activity plot with a size of (10, 5)
            plotting_params = None
            if plot:
                assert USER_plotting_params is not None, 'USER_plotting_params must be set in USER_INPUTS.py'
                plotting_params = USER_plotting_params['NetworkActivity']
                plotting_params['simLabel'] = simLabel
                plotting_params['batch_saveFolder'] = batch_saveFolder
                print('plotting_params:', plotting_params)
                #sys.exit()
                #prep to plot fitplot
                below_baseline_target = kwargs['pops']['baseline_targets']['target'] * 0.95
                above_amplitude_target = kwargs['pops']['burts_peak_targets']['big_bursts']['target']
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
                #if plot_save_path defined, replace saveFig with plot_save_path
                if plot_save_path is not None: plotting_params['saveFig'] = plot_save_path
            net_metrics = measure_network_activity(
                rasterData, 
                binSize=binSize, 
                gaussianSigma=gaussianSigma, 
                thresholdBurst=thresholdBurst,
                plot=plot,
                plotting_params = plotting_params,
                crop = USER_raster_crop
            )
        def plot_raster():
            #Attempt to generate the raster plot
            figname = 'raster_plot.png'
            timeVector = net_activity_metrics['timeVector']*1000 #convert back to ms
            timeRange = [timeVector[0], timeVector[-1]]
            #raster_plot_path = f'{batch_saveFolder}/{simLabel}_raster_plot.svg'
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
            
            try:
                netpyne.sim.analysis.plotRaster(saveFig=fig_path, 
                                            #timeRange = raster_activity_timeRange,
                                            timeRange = timeRange,
                                            showFig=False,
                                            labels = None, 
                                            figSize=USER_plotting_params['figsize'])#, dpi=600)
                # #redo as png
                # cairosvg.svg2png(url=raster_plot_path, write_to=raster_plot_path.replace('.svg', '.png'))
            except:
                print(f'Error generating raster plot from Data at: {data_file_path}')
                # raster_plot_path = None
                pass
        def most_active_time_range(timeVector, sim_obj):
                '''subfunc'''
                def electric_slide(time_points, voltage_trace):
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

                    # Slide the window over the voltage trace
                    for start_time in np.arange(time_points[0], time_points[-1] - window_size + step_size, step_size):
                        # Get the end time of the current window
                        end_time = start_time + window_size

                        # Get the voltage trace for the current window
                        voltage_trace = np.array(voltage_trace)
                        window_voltage_trace = voltage_trace[(time_points >= start_time) & (time_points < end_time)]

                        # print(f'test')
                        # sys.exit()
                        # Detect zero-crossings: points where the signal changes from positive to negative
                        zero_crossings = np.where(np.diff(np.sign(window_voltage_trace)) < spike_threshold)[0]

                        # Count the number of zero-crossings, which corresponds to the number of spikes
                        spike_count = len(zero_crossings)

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
                # Get the voltage trace for the first (excitatory) neuron
                # Get the voltage trace for the first (excitatory) neuron
                excite_voltage_trace = sim_obj.allSimData['soma_voltage'][neuron_gids[0]]
                # Get the voltage trace for the second (inhibitory) neuron
                inhib_voltage_trace = sim_obj.allSimData['soma_voltage'][neuron_gids[1]]                
                # Get the time points
                time_points = sim_obj.allSimData['t']
                # Create a zip object from time_points and voltage traces
                excite_pairs = zip(time_points, excite_voltage_trace)
                inhib_pairs = zip(time_points, inhib_voltage_trace)
                # Filter voltage traces based on filtered time_points
                excite_voltage_trace = [v for t, v in excite_pairs if t >= timeVector[0] and t <= timeVector[-1]]
                inhib_voltage_trace = [v for t, v in inhib_pairs if t >= timeVector[0] and t <= timeVector[-1]]
                time_points = [t for t in time_points if t >= timeVector[0] and t <= timeVector[-1]]                
                #print(f'test')                
                excite_timeRange = electric_slide(time_points, excite_voltage_trace)
                inhib_timeRange = electric_slide(time_points, inhib_voltage_trace)
                return excite_timeRange, inhib_timeRange
        def plot_trace_example():
            # Attempt to generate sample trace for an excitatory example neuron
            try:
                figname = 'sample_trace'
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
                excite_timeVector, inhib_timeVector = most_active_time_range(timeVector, sim_obj)
                timeRanges = [excite_timeVector, inhib_timeVector]
                titles = ['E0_highFR', 'I0_highFR']
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                # Create individual plots and save as PNG
                for i, timeRange in enumerate(timeRanges):
                    title = titles[i]
                    # Prepare the sample trace
                    sample_trace = sim_obj.analysis.plotTraces(
                        include=[('E', 0), ('I', 0)],
                        overlay=True,
                        oneFigPer='trace',
                        title=title,
                        timeRange=timeRange,
                        showFig=False,
                        figSize=(USER_plotting_params['figsize'][0], USER_plotting_params['figsize'][1]/2)
                    )
                    # Get the figure
                    fig = sample_trace[0]['_trace_soma_voltage']
                    # Add title to the figure
                    fig.suptitle(title)
                    # Move title all the way to the left
                    fig.tight_layout(rect=[0, 0.03, 1, 1])
                    # Save the figure with the title
                    fig_path_path = f'{fig_path}_{title}.png'
                    fig.savefig(fig_path_path)

                # Create a composite figure
                fig, axs = plt.subplots(2, 1, figsize=USER_plotting_params['figsize'])
                for i, title in enumerate(titles):
                    # Load the image
                    img = mpimg.imread(f'{fig_path}_{title}.png')
                    # Add the image to the subplot
                    axs[i].imshow(img)
                    axs[i].axis('off')

                # Adjust the layout
                fig.tight_layout(rect=[0, 0.03, 1, 1])

                # Save the composite figure
                fig.savefig(f'{fig_path}_combined.png')
                #fig.suptitle('Middlemost 1 second of simulation')
                # Save the figure with the title
                #fig.savefig(sample_trace_path_E)
                # redo as png
                #cairosvg.svg2png(url=sample_trace_path_E, write_to=sample_trace_path_E.replace('.svg', '.png'))
            except:
                print(f'Error generating sample trace plot from Data at: {data_file_path}')
                #sample_trace_path_E = None
                pass
        def plot_connections():
            # Attempt to generate sample trace for an excitatory example neuron
            try:
                figname = 'connections'
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
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                # Create individual plots and save as PNG
                sim_obj.analysis.plot2Dnet(saveFig=fig_path, showFig=False, showConns=True, figSize=(USER_plotting_params['figsize'][0], USER_plotting_params['figsize'][1]/2))

            except:
                print(f'Error generating sample trace plot from Data at: {data_file_path}')
                #sample_trace_path_E = None
                pass
        
        '''plotting'''
        try: 
            plot_network_activity_fitness(net_activity_metrics)
        except: 
            print(f'Error generating network activity plot from Data at: {data_file_path}')
            pass
        try: 
            if not exp_mode: plot_raster()
        except: 
            print(f'Error generating raster plot from Data at: {data_file_path}')
            pass
        try: 
            if not exp_mode: plot_trace_example()
        except: 
            print(f'Error generating sample trace plot from Data at: {data_file_path}')
            pass
        try: 
            if not exp_mode: plot_connections()
        except: 
            print(f'Error generating connections plot from Data at: {data_file_path}')
            pass
    
    '''
    Get Fitness (Main Fitness Function)
    '''    
    ##find relevant batch object in call stack
    # Use the function to get the Batch object and simLabel
    if exp_mode:
        batch_saveFolder = fitness_save_path
        simLabel = "experimental_data"
    if batch_saveFolder is None and simLabel is None:
        batch, simLabel = find_batch_object_and_sim_label()
        batch_saveFolder = batch.saveFolder
    assert simLabel is not None, "SimLabel undefined."
    assert batch_saveFolder is not None, "Batch save folder undefined."

    ##get fitness
    if plot is None: plot = USER_plot_fitness_bool
    average_fitness, avg_scaled_fitness = get_fitness(simData, plot = plot, **kwargs)
    #return avg_scaled_fitness
    return average_fitness