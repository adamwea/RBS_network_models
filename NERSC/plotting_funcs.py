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

    saveFig = plotting_params['saveFig']
    if saveFig:
        assert saveFig, 'saveFig should be set to a relative path written as a string' #e.g. 'NERSC/plots/'
        batch_saveFolder = plotting_params['batch_saveFolder']
        assert batch_saveFolder, 'batch_saveFolder should be set to a relative path written as a string'
        simLabel = plotting_params['simLabel']
        assert simLabel, 'simLabel should be a string'
        job_name = os.path.basename(os.path.dirname(batch_saveFolder))
        gen_folder = simLabel.split('_cand')[0]
        fig_path = os.path.join(saveFig, f'{job_name}/{gen_folder}/{simLabel}_NetworkActivity.png')
        fig_dir = os.path.dirname(fig_path)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        if 'output' in fig_path:
            print('')
        plt.savefig(fig_path, bbox_inches='tight')
    else:
        plt.show()