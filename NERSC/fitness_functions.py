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

def fitnessFunc(simData, sim_obj = None, plot = False, simLabel = None, data_file_path = None, batch_saveFolder = None, fitness_save_path = None, **kwargs):   
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
            maxFitness = kwargs['maxFitness']
            fitnessVals['burstAmp_Fitness'] = {'Value': None, 'Fit': maxFitness}
            fitnessVals['IBI_fitness'] = {'Value': None, 'Fit': maxFitness}
            fitnessVals['baselineFitness'] = {'Value': None, 'Fit': maxFitness}
            fitnessVals['slopeFitness'] = {'Value': None, 'Fit': maxFitness}
            #fitnessVals['baseline_diff_fitness'] = {'Value': None, 'Fit': maxFitness}
            fitnessVals['burst_peak_frequency_fitness'] = {'Value': None, 'Fit': maxFitness}
            fitnessVals['rate_fitness'] = {'Value': None, 'Fit': maxFitness}
            fitnessVals['sustain_oscillation_fitness'] = {'Value': None, 'Fit': maxFitness}
            
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
        rasterData = simData
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
            burst_peak_frequency_fitness = min(np.exp(
                abs(pops_frequency['target'] - burst_peak_frequency) / pops_frequency['width']), maxFitness) if burst_peak_frequency > pops_frequency['min'] else maxFitness
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
            sustain_fit = min(np.exp((sustained_osci_target['target'] - sustained_osci100) / sustained_osci_target['width']), maxFitness
                             ) if sustained_osci100 > sustained_osci_target['min'] else maxFitness

            # Print the sustain duration and its fitness
            print('Sustain Duration: %.3f, Fitness: %.3f' % (sustained_osci100, sustain_fit))
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
            thresh_fit = min(np.exp((thresh_target['target'] - thresh)/ 
                             thresh_target['width']), maxFitness
                             ) if thresh > thresh_target['min'] and thresh < thresh_target['max'] else maxFitness

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
            slopeFitness = min(np.exp(abs(rate_slope['target'] - slope)/rate_slope['width']) if abs(slope) < rate_slope['max'] else maxFitness,
                                maxFitness)

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
        rasterData = simData
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
        try: 
            # BurstPeakValue Fitness
            pops = kwargs['pops']
            pops_peaks = pops['burts_peak_targets']
            maxFitness = kwargs['maxFitness']
            burstPeakValues = net_activity_metrics['burstPeakValues']
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
            fitnessVals['burstAmp_Fitness'] = {'Value': avg_burstPeak, 'Fit': BurstVal_fitness}
            return fitnessVals
        except Exception as e:
            print(f'Error calculating burst peak fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            fitnessVals['burstAmp_Fitness'] = {'Value': None, 'Fit': maxFitness}
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
            baseline_width = pops_baseline['width']
            baseline_max = pops_baseline['max']

            baselineFitness = min(np.exp(abs(baseline_target - baseline) / baseline_width), maxFitness) if baseline < baseline_max else maxFitness

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
                min(np.exp(abs(v['target'] - simData['popRates'][k]) / v['width']), maxFitness)
                if simData["popRates"][k] > v['min'] else maxFitness
                for k, v in pops_rate.items()
            ]
            rate_fitness = np.mean(popFitness)
            popInfo = '; '.join(['%s rate=%.1f fit=%1.f'%(p,r,f) for p,r,f in zip(
                list(simData['popRates'].keys()), 
                list(simData['popRates'].values()), popFitness)])
            print('  '+popInfo)
            fitnessVals['rate_fitness'] = {'Value': np.mean(list(simData['popRates'].values())), 'Fit': rate_fitness}
            return fitnessVals
        except Exception as e:
            print(f'Error calculating firing rate fitness.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            fitnessVals['rate_fitness'] = {'Value': None, 'Fit': maxFitness}
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
        fitnessVals, net_activity_metrics = get_network_activity_metrics(fitnessVals, plot = USER_plot_NetworkActivity)
        if net_activity_metrics == {}:
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
        # if data_file_path is not None and 'overnight' in data_file_path:
        #     print('overnight')
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

        #print(f'Fitness results saved to {output_path}/{gen_folder}/{simLabel}_Fitness.json')
        if plot: plot_fitness(fitnessVals, simData, net_activity_metrics, **kwargs)
        return avg_scaled_fitness
    
    def plot_fitness(fitnessVals, simData, net_activity_metrics, **kwargs):
        def plot_network_activity_fitness(net_activity_metrics):
            #net_activity_metrics = net_activity_metrics
            rasterData = simData
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
                below_baseline_target = kwargs['pops']['baseline_targets']['target'] * 0.95
                above_amplitude_target = kwargs['pops']['burts_peak_targets']['target']
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
            net_metrics = measure_network_activity(
                rasterData, 
                binSize=binSize, 
                gaussianSigma=gaussianSigma, 
                thresholdBurst=thresholdBurst,
                plot=plot,
                plotting_params = plotting_params,
                crop = USER_raster_crop,
            )
        def plot_raster():
            #Attempt to generate the raster plot
            assert sim_obj is not None, 'sim_obj must be defined to generate raster plot'
            figname = 'raster_plot.png'
            rasterData = sim_obj.analysis.prepareRaster()
            timeVector = net_activity_metrics['timeVector']
            timeRange = [timeVector[0], timeVector[-1]]
            #raster_plot_path = f'{batch_saveFolder}/{simLabel}_raster_plot.svg'
            job_name = os.path.basename(os.path.dirname(batch_saveFolder))
            gen_folder = simLabel.split('_cand')[0]
            saveFig = USER_plotting_params['saveFig']
            fig_path = os.path.join(saveFig, f'{job_name}/{gen_folder}/{simLabel}_{figname}')
            USER_fresh_plots = USER_plotting_params['fresh_plots']
            if os.path.exists(fig_path) and USER_fresh_plots: pass
            elif os.path.exists(fig_path) and not USER_fresh_plots: 
                print(f'File {fig_path} already exists and USER_fresh_plots is set to False. Skipping plot.')
                return
            elif os.path.exists(fig_path) is False: pass
            else: raise ValueError(f'Idk how we got here. Logically.')
            
            try:
                sim_obj.analysis.plotRaster(saveFig=fig_path, 
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
        def plot_trace_example():
            # Attempt to generate sample trace for an excitatory example neuron
            try:
                figname = 'sample_trace.png'
                job_name = os.path.basename(os.path.dirname(batch_saveFolder))
                gen_folder = simLabel.split('_cand')[0]
                saveFig = USER_plotting_params['saveFig']
                fig_path = os.path.join(saveFig, f'{job_name}/{gen_folder}/{simLabel}_{figname}')
                USER_fresh_plots = USER_plotting_params['fresh_plots']
                if os.path.exists(fig_path) and USER_fresh_plots: pass
                elif os.path.exists(fig_path) and not USER_fresh_plots: 
                    print(f'File {fig_path} already exists and USER_fresh_plots is set to False. Skipping plot.')
                    return
                elif os.path.exists(fig_path) is False: pass
                else: raise ValueError(f'Idk how we got here. Logically.')

                timeVector = np.array(net_activity_metrics['timeVector'])
                # Find the indices of the closest values in timeVector to the desired values
                start_index = (np.abs(timeVector - (timeVector[int(len(timeVector)/2)] - 500))).argmin()
                end_index = (np.abs(timeVector - (timeVector[int(len(timeVector)/2)] + 500))).argmin()

                # Ensure start_index and end_index are not the same and start_index comes before end_index
                if start_index == end_index:
                    if start_index > 0:
                        start_index -= 1
                    elif end_index < len(timeVector) - 1:
                        end_index += 1
                elif start_index > end_index:
                    start_index, end_index = end_index, start_index

                # Use these indices to get the closest values in timeVector
                timeRange = [timeVector[start_index], timeVector[end_index]]
                # Prepare the sample trace
                sample_trace_E = sim_obj.analysis.plotTraces(
                    include=[('E', 0), ('I', 0)],
                    overlay=True,
                    oneFigPer='trace',
                    title='Middlemost 1 second of simulation',
                    timeRange=timeRange,
                    #saveFig=sample_trace_path_E,
                    showFig=False,
                    figSize=USER_plotting_params['figsize']
                )
                # Add title to the figure
                fig = sample_trace_E[0]['_trace_soma_voltage']
                fig.suptitle('Middlemost 1 second of simulation')
                #move title all the way to the left
                fig.tight_layout(rect=[0, 0.03, 1, 1])
                # Save the figure with the title
                fig.savefig(fig_path)
                #fig.suptitle('Middlemost 1 second of simulation')
                # Save the figure with the title
                #fig.savefig(sample_trace_path_E)
                # redo as png
                #cairosvg.svg2png(url=sample_trace_path_E, write_to=sample_trace_path_E.replace('.svg', '.png'))
            except:
                print(f'Error generating sample trace plot from Data at: {data_file_path}')
                #sample_trace_path_E = None
                pass
        
        try: plot_network_activity_fitness(net_activity_metrics)
        except: 
            print(f'Error generating network activity plot from Data at: {data_file_path}')
            pass
        try: plot_raster()
        except: 
            print(f'Error generating raster plot from Data at: {data_file_path}')
            pass
        try: plot_trace_example()
        except: 
            print(f'Error generating sample trace plot from Data at: {data_file_path}')
            pass
    
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