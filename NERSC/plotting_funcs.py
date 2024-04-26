#Imports
import matplotlib.pyplot as plt
import numpy as np
import os
from netpyne import sim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks
from scipy.stats import norm
from scipy.signal import butter, filtfilt
from scipy import stats
#from USER_INPUTS import *

def plot_network_activity(plotting_params, timeVector, firingRate, burstPeakTimes, burstPeakValues, thresholdBurst, rmsFiringRate): #rasterData, min_peak_distance = 1.0, binSize=0.02*1000, gaussianSigma=0.16*1000, thresholdBurst=1.2, figSize=(10, 6), saveFig=False, timeRange = None, figName='NetworkActivity.png'):
    
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
        #assert fig_ylim, 'ylim must be set to a list of two integers' #e.g. [1.5, 5]
        plt.ylim(fig_ylim)  # Set y-axis limits to min and max of firingRate
    else:
        yhigh100 = plotting_params['yhigh100']
        ylow100 = plotting_params['ylow100'] 
        assert yhigh100, 'USER_Activity_yhigh100 must be set to a float' #e.g. 1.05
        assert ylow100, 'USER_Activity_ylow100 must be set to a float' #e.g. 0.95
        plt.ylim([min(firingRate)*ylow100, max(firingRate)*yhigh100])  # Set y-axis limits to min and max of firingRate
    plt.ylabel('Firing Rate [Hz]')
    plt.xlabel('Time [ms]')
    title_font = plotting_params['title_font']
    assert title_font, 'title_font must be set to an interger' #e.g. {'fontsize': 11}
    plt.title('Network Activity', fontsize=title_font)

    # Plot the threshold line and burst peaks
    plt.plot(np.arange(timeVector[-1]), thresholdBurst * rmsFiringRate * np.ones(np.ceil(timeVector[-1]).astype(int)), color='gray')
    plt.plot(burstPeakTimes, burstPeakValues, 'or')  # Plot burst peaks as red circles

    default_name = 'NetworkActivity.png'
    figname = default_name
    # try: 
    #     fitplot = plotting_params['fitplot']
    #     if fitplot == 'burst_freq' or fitplot == 'burst_IBI':            
    #         x_min, x_max = plt.xlim()
    #         y_min, y_max = plt.ylim()
    #         x_coord = x_min*1.05
    #         y_coord = y_max*0.990
    #         decrement = 0.025*(y_max-y_min)
    #         targets = plotting_params['targets']['pops']['burst_peak_frequency']
    #         for key, target in targets.items():
    #             plt.text(x_coord, y_coord, f'{key}: {target}', fontsize=11)
    #             y_coord -= decrement  # Adjust this value to change the spacing between lines
    #         targets = plotting_params['targets']['pops']['IBI_targets']
    #         fitnessVal = plotting_params['fitnessVals']['burst_peak_frequency_fitness']
    #         plt.text(x_coord, y_coord, f'Fitness: {fitnessVal}', fontsize=11)
    #         y_coord -= decrement
    #         #measure frequency of peaks
    #         peak_freq = len(burstPeakTimes) / (timeVector[-1] / 1000) #convert to seconds
    #         plt.text(x_coord, y_coord, f'Peak Frequency: {peak_freq}', fontsize=11)
    #         y_coord -= decrement
    #         y_coord -= decrement
    #         for key, target in targets.items():
    #             plt.text(x_coord, y_coord, f'{key}: {target}', fontsize=11)
    #             y_coord -= decrement  # Adjust this value to change the spacing between lines
    #         fitnessVal = plotting_params['fitnessVals']['IBI_fitness']
    #         plt.text(x_coord, y_coord, f'Fitness: {fitnessVal}', fontsize=11)
    #         y_coord -= decrement
    #         #measure frequency of peaks
    #         IBIs = np.diff(burstPeakTimes) #
    #         meanIBI = np.mean(IBIs)
    #         plt.text(x_coord, y_coord, f'mean IBI: {meanIBI}', fontsize=11)

    #         plt.title('Network Activity - Peak Frequency and IBI Fit', fontsize=title_font)
    #         #plt.show()
    #         figname = 'peakFreqIBIFit.png'
    #     elif fitplot == 'burst_peak' or fitplot == 'baseline':
            
    #         #measure frequency of peaks
    #         peak_amp = np.mean(burstPeakValues)
    #         peak_amp_target = plotting_params['targets']['pops']['burts_peak_targets']['target']
    #         #adjust ylim as needed to get peak_amp_target in view
    #         if peak_amp_target > max(firingRate)*yhigh100:
    #             plt.ylim([min(firingRate)*ylow100, peak_amp_target*1.05])
    #         elif peak_amp_target < min(firingRate)*ylow100:
    #             plt.ylim([peak_amp_target*0.95, max(firingRate)*yhigh100])
    #         #baseline
    #         baseline = np.mean(firingRate)
    #         baseline_target = plotting_params['targets']['pops']['baseline_targets']['target']
    #         #adjust ylim as needed to get baseline_target in view
    #         if baseline_target > max(firingRate)*yhigh100:
    #             plt.ylim([min(firingRate)*ylow100, baseline_target*1.05])
    #         elif baseline_target < min(firingRate)*ylow100:
    #             plt.ylim([baseline_target*0.95, max(firingRate)*yhigh100])
    #         yhigh100 = plotting_params['yhigh100']
    #         x_min, x_max = plt.xlim()
    #         y_min, y_max = plt.ylim()
    #         x_coord = x_min*1.05
    #         y_coord = y_max*0.980
    #         decrement = 0.025*(y_max-y_min)
    #         targets = plotting_params['targets']['pops']['burts_peak_targets']
    #         for key, target in targets.items():
    #             plt.text(x_coord, y_coord, f'{key}: {target}', fontsize=11)
    #             y_coord -= decrement  # Adjust this value to change the spacing between lines
    #         fitnessVal = plotting_params['fitnessVals']['burstAmp_Fitness']
    #         plt.text(x_coord, y_coord, f'Fitness: {fitnessVal}', fontsize=11)
    #         y_coord -= decrement
    #         plt.text(x_coord, y_coord, f'Mean Peak Amplitude: {peak_amp}', fontsize=11)
    #         y_coord -= decrement
    #         y_coord -= decrement
    #         targets = plotting_params['targets']['pops']['baseline_targets']
    #         for key, target in targets.items():
    #             plt.text(x_coord, y_coord, f'{key}: {target}', fontsize=11)
    #             y_coord -= decrement  # Adjust this value to change the spacing between lines
    #         fitnessVal = plotting_params['fitnessVals']['baselineFitness']
    #         plt.text(x_coord, y_coord, f'Fitness: {fitnessVal}', fontsize=11)
    #         y_coord -= decrement
    #         plt.text(x_coord, y_coord, f'Baseline (signal mean): {baseline}', fontsize=11)
    #         plt.title('Network Activity - Peak Amplitude Fit', fontsize=title_font)
    #         #plot horizontal line at target
    #         plt.axhline(peak_amp_target, color='r', linestyle='--')
    #         plt.axhline(baseline_target, color='b', linestyle='--')
    #         #plt.show()
    #         figname = 'peakAmpBaselineFit.png'
    #     #elif fitplot == 'burst_freq_rms':
    # except:
    #     pass

    if plotting_params['fitplot'] is not None:
        name = 'Network Activity - Fitness'
        #net_activity_metrics = plotting_params['net_activity_metrics']
        targets = plotting_params['targets']
        if targets is not None:
            peak_amp_target = targets['pops']['burts_peak_targets']['target']
            baseline_target = targets['pops']['baseline_targets']['target']
            plt.axhline(peak_amp_target, color='r', linestyle='--', label='Peak Amplitude Target')
            plt.axhline(baseline_target, color='b', linestyle='--', label='Baseline Target')
            plt.legend()
    
    saveFig = plotting_params['saveFig']
    if saveFig:
        assert saveFig, 'saveFig should be set to a relative path written as a string' #e.g. 'NERSC/plots/'
        batch_saveFolder = plotting_params['batch_saveFolder']
        assert batch_saveFolder, 'batch_saveFolder should be set to a relative path written as a string'
        simLabel = plotting_params['simLabel']
        assert simLabel, 'simLabel should be a string'
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
    else:
        plt.show()