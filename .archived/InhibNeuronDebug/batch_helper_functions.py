'''
Collection of useful functions written for batch processing of the HD-MEA simulations using NetPyNE
'''

#imports
import os
from netpyne import sim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks
from scipy.stats import norm
#import svgutils.transform as sg
from scipy.signal import butter, filtfilt
from scipy import stats
#import cairosvg

### Batch Functions
import os
import shutil

# (I dont understand how this works, figure it out later)
import inspect

from plotting_funcs import *

#from USER_INPUTS import *

def get_walltime_per_sim(USER_walltime_per_gen, USER_pop_size, USER_nodes):
    USER_walltime_per_gen_hours = int(USER_walltime_per_gen.split(':')[0])
    USER_walltime_per_gen_hours = USER_walltime_per_gen_hours / USER_nodes
    USER_walltime_per_gen_minutes = int(USER_walltime_per_gen.split(':')[1])
    USER_walltime_per_gen_minutes = USER_walltime_per_gen_minutes / USER_nodes
    USER_walltime_per_gen_seconds = int(USER_walltime_per_gen.split(':')[2])
    USER_walltime_per_gen_seconds = USER_walltime_per_gen_seconds / USER_nodes
    
    USER_walltime_per_gen_seconds += USER_walltime_per_gen_minutes*60
    USER_walltime_per_gen_seconds += USER_walltime_per_gen_hours*3600
    USER_walltime_per_sim_seconds = USER_walltime_per_gen_seconds/USER_pop_size

    # Convert back to hh:mm:ss
    hours, remainder = divmod(USER_walltime_per_sim_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

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

def move_btr_files():
    try:
        # List all files in the current working directory
        files_in_cwd = os.listdir('.')
        # Filter out the .btr files
        btr_files = [file for file in files_in_cwd if file.endswith('.btr')]
        
        # Check if there are any .btr files to move
        if not btr_files:
            print("No .btr files found.")
            return
        
        # Create a subfolder for .btr files if it doesn't exist
        btr_folder = 'btr files'
        if not os.path.exists(btr_folder):
            os.makedirs(btr_folder)
        
        # Move each .btr file to the subfolder
        for btr_file in btr_files:
            shutil.move(btr_file, os.path.join(btr_folder, btr_file))
        
        print(".btr files successfully moved.")
        
    except Exception as e:
        print("Failed to move files:", e)

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
    burstPeakTimes, burstPeakValues, burstPeakStarts, burstPeakEnds = burstPeakQualityControl(burstPeakTimes, burstPeakValues)

    ##Adjust lengths of timeVector and firingRate to start with the latest start before earliest peak and earliest end after latest peak
    #get the latest start before the earliest pe
    # Find the differences between the first peak time and all start times
    differences = burstPeakTimes[0] - burstPeakStarts
    # Only consider start times that are before the first peak time
    valid_starts = np.where(differences > 0)
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

    '''
    Optionally Plot
    '''
    if plot: 
        assert plotting_params is not None, 'plotting_params must be specified'
        plot_network_activity(plotting_params, timeVector, firingRate, burstPeakTimes, burstPeakValues, thresholdBurst, rmsFiringRate)
        
    '''
    Network Metric Outputs
    '''
    # Calculate Baseline
    baseline = np.mean(firingRate)
    #measure frequency of peaks
    peak_freq = len(burstPeakTimes) / (timeVector[-1] / 1000) #convert to seconds
    # Calculate peak variance
    peak_variance = np.var(burstPeakValues)
    # Calculate the range of burstPeakValues
    value_range = np.max(burstPeakValues) - np.min(burstPeakValues)
    # Calculate the maximum possible variance
    max_possible_variance = value_range**2
    # Normalize the variance to a 0 to 1 scale
    normalized_peak_variance = peak_variance / max_possible_variance if max_possible_variance != 0 else 0
    #measure IBI
    burstPeakTimes = np.array(burstPeakTimes)
    IBIs = np.diff(burstPeakTimes) #
    #measure baseline diff
    height = thresholdBurst * rmsFiringRate
    baseline_diff = height - baseline    
    #IBI = IBI / 1000 #convert to seconds
    sustained_osci100 = (len(timeVector)/og_timeVector_len)*100
    measurements = {
        'burstPeakValues': burstPeakValues-(thresholdBurst * rmsFiringRate),
        'burstPeakTimes': burstPeakTimes,
        'IBIs': IBIs,
        'firingRate': firingRate,
        'timeVector': timeVector,
        'baseline': baseline,
        'baseline_diff': baseline_diff,
        'peak_freq': peak_freq,
        'normalized_peak_variance': normalized_peak_variance,
        'threshold': thresholdBurst * rmsFiringRate,
        'sustain': sustained_osci100,
        #'base_locs': base_locs,
    }    
    return measurements 

def get_batchrun_info(root, file):
    file_path = os.path.join(root, file)
    batchrun_folder = os.path.basename(os.path.normpath(root))    
    batch_key = file.split(batchrun_folder)[1].split('data')[0]
    #hot fix for evol_batchruns
    if "gen_" in batchrun_folder and batch_key == '/':
        batch_key = file.split(f"{batchrun_folder}/")[1].split('data')[0]
    return file_path, batchrun_folder, batch_key

#deprecated
# def plot_network_activity(rasterData, min_peak_distance = 1.0, binSize=0.02*1000, gaussianSigma=0.16*1000, thresholdBurst=1.2, figSize=(10, 6), saveFig=False, timeRange = None, figName='NetworkActivity.png'):
#     SpikeTimes = rasterData['spkTimes']
#     relativeSpikeTimes = SpikeTimes
#     relativeSpikeTimes = np.array(relativeSpikeTimes)
#     relativeSpikeTimes = relativeSpikeTimes - relativeSpikeTimes[0]  # Set the first spike time to 0

#     # Step 1: Bin all spike times into small time windows
#     timeVector = np.arange(0, max(relativeSpikeTimes) + binSize, binSize)  # Time vector for binning
#     binnedTimes, _ = np.histogram(relativeSpikeTimes, bins=timeVector)  # Bin spike times
#     binnedTimes = np.append(binnedTimes, 0)  # Append 0 to match MATLAB's binnedTimes length

#     # Step 2: Smooth the binned spike times with a Gaussian kernel
#     kernelRange = np.arange(-3*gaussianSigma, 3*gaussianSigma + binSize, binSize)  # Range for Gaussian kernel
#     kernel = norm.pdf(kernelRange, 0, gaussianSigma)  # Gaussian kernel
#     kernel *= binSize  # Normalize kernel by bin size
#     firingRate = convolve(binnedTimes, kernel, mode='same') / binSize  # Convolve and normalize by bin size

#     # Create a new figure with a specified size (width, height)
#     plt.figure(figsize=figSize)
#     #margin_width = 200
#     assert timeRange is not None, 'timeRange must be specified'
#     #margin_width = timeRange[0]
#     # Find the indices of timeVector that correspond to the first and last 100 ms
#     #start_index = np.where(timeVector >= margin_width)[0][0]
#     #end_index = np.where(timeVector <= max(timeVector) - margin_width)[0][-1]
#     #end_index = np.where(timeVector <= max(timeVector))[0][-1]

#     debinned_timeRange = [int(timeRange[0]/binSize), int(timeRange[1]/binSize)]
#     timeVector = timeVector[debinned_timeRange[0]:debinned_timeRange[1]]
#     firingRate = firingRate[debinned_timeRange[0]:debinned_timeRange[1]]

#     # Plot the smoothed network activity
#     plt.subplot(1, 1, 1)
#     plt.plot(timeVector, firingRate, color='black')
#     plt.xlim([timeVector[0], timeVector[-1]])  # Restrict the plot to the first and last 100 ms
#     #plt.xlim([timeVector[start_index], timeVector[end_index]])  # Restrict the plot to the first and last 100 ms
#     #plt.ylim([min(firingRate[start_index:end_index])*0.95, max(firingRate[start_index:end_index])*1.05])  # Set y-axis limits to min and max of firingRate
    
#     # restrict y range 1.5 to 5
#     plt.ylim([1, 5.25])  # Set y-axis limits to min and max of firingRate
    
#     plt.ylabel('Firing Rate [Hz]')
#     plt.xlabel('Time [ms]')
#     plt.title('Network Activity', fontsize=11)

#     # Step 3: Peak detection on the smoothed firing rate curve
#     rmsFiringRate = np.sqrt(np.mean(firingRate**2))  # Calculate RMS of the firing rate
#     peaks, properties = find_peaks(firingRate, height=thresholdBurst * rmsFiringRate, distance=min_peak_distance)  # Find peaks above the threshold
#     burstPeakTimes = timeVector[peaks]  # Convert peak indices to times
#     burstPeakValues = properties['peak_heights']  # Get the peak values

#     # Get the indices of the valid peaks
#     valid_peak_indices = np.where((burstPeakTimes > timeVector[0]) & (burstPeakTimes < timeVector[-1]))[0]

#     # Filter burstPeakTimes and burstPeakValues using the valid peak indices
#     burstPeakTimes = burstPeakTimes[valid_peak_indices]
#     burstPeakValues = burstPeakValues[valid_peak_indices]

#     # Plot the threshold line and burst peaks
#     plt.plot(np.arange(timeVector[-1]), thresholdBurst * rmsFiringRate * np.ones(np.ceil(timeVector[-1]).astype(int)), color='gray')
#     plt.plot(burstPeakTimes, burstPeakValues, 'or')  # Plot burst peaks as red circles    

#     if saveFig:
#         if isinstance(saveFig, str):
#             # plt.savefig(saveFig, dpi=600, bbox_inches='tight')
#             # plt.savefig(saveFig.replace('.png', '.svg'), bbox_inches='tight')
#             plt.savefig(saveFig, bbox_inches='tight')
#         else:
#             # plt.savefig(figName, dpi=600, bbox_inches='tight')
#             # plt.savefig(figName.replace('.png', '.svg'), bbox_inches='tight')
#             plt.savefig(figName, bbox_inches='tight')
#     #plt.show()
            
# def generate_all_figures(fresh_figs = False, batchLabel = 'batch', batch_path = None, size = 6, net_activity_params = {'binSize': 3, 'gaussianSigma': 12, 'thresholdBurst': 1.0}, simLabel = None, raster_activity_timeRange = None):
        
#     #function globals    
#     raster_activity_timeRange = raster_activity_timeRange

#     #relative figure sizes in inches    
#     raster_height = size
#     raster_width = size
#     sample_trace_height = size
#     sample_trace_width = size
#     network_activity_height = raster_height
#     network_activity_width = raster_width
#     locations_height = raster_height
#     locations_width = locations_height*2
#     conn_mat_height = raster_height
#     conn_mat_width = conn_mat_height
    
#     def generate_primary_figures(root, file, size = size, raster_activity_timeRange = raster_activity_timeRange, net_activity_params = {'binSize': 3, 'gaussianSigma': 12, 'thresholdBurst': 1.0}):

#         ## Data
#         #load the simulation data
#         #get file info
#         file_path, batchrun_folder, batch_key = get_batchrun_info(root, file)     
#         sim.loadAll(file_path)

#         # Get total time range
#         total_time = sim.allSimData['t'][-1]

#         #Calculate raster and network activity time range
#         if raster_activity_timeRange is None:
#             #raster_activity_timeRange = [0, total_time]
#             crop = total_time/15
#             raster_activity_timeRange = [crop, total_time-crop]

#         # Calculate the middlemost 1 second
#         start_time = total_time / 2 - 500
#         end_time = total_time / 2 + 500
        
#         #get net activity params
#         binSize = net_activity_params['binSize']
#         gaussianSigma = net_activity_params['gaussianSigma']
#         thresholdBurst = net_activity_params['thresholdBurst']      


#         #prepare destination folder
#         NetworkBurstingFolder = 'plots'
#         if simLabel is not None:
#             NetworkBurstingFolder = f'{simLabel}'
#         if not os.path.exists(f'{root}/{NetworkBurstingFolder}'):
#             os.makedirs(f'{root}/{NetworkBurstingFolder}')
#             print(f'Created folder: {root}/{NetworkBurstingFolder}')
#         NetBurst_path = f'{root}/{NetworkBurstingFolder}'    
            
#         # Define the paths to the raster plot and network activity images
#         # raster_plot_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_raster.png'
#         # network_activity_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_network_activity_def.png'
#         # TWODnet_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_plot_2Dnet.png'
#         # locs_path = f'{root}/{NetworkBurstingFolder}/locs.png'
#         # conn_mat_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_conn_matrix.png'
#         # sample_trace_path_E = f'{root}/{NetworkBurstingFolder}/{batch_key}_sample_trace_E.png'
#         # sample_trace_path_I = f'{root}/{NetworkBurstingFolder}/{batch_key}_sample_trace_I.png'
#         raster_plot_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_raster.svg'
#         network_activity_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_network_activity_def.svg'
#         TWODnet_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_plot_2Dnet.svg'
#         locs_path = f'{root}/{NetworkBurstingFolder}/locs.svg'
#         conn_mat_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_conn_matrix.svg'
#         sample_trace_path_E = f'{root}/{NetworkBurstingFolder}/{batch_key}_sample_trace_E.svg'
#         sample_trace_path_I = f'{root}/{NetworkBurstingFolder}/{batch_key}_sample_trace_I.svg'   

#         # Try Loading Images if they Already Exist
#         try:
#             assert fresh_figs == False
#             assert os.path.exists(raster_plot_path)
#             assert os.path.exists(network_activity_path)
#             assert os.path.exists(TWODnet_path)
#             assert os.path.exists(locs_path)
#             assert os.path.exists(conn_mat_path)
#             assert os.path.exists(sample_trace_path_E)
#             assert os.path.exists(sample_trace_path_I)

#             # raster_plot = mpimg.imread(raster_plot_path)
#             # network_activity = mpimg.imread(network_activity_path)
#             # TWODnet = mpimg.imread(TWODnet_path)
#             # locs = mpimg.imread(locs_path)
#             # conn_mat = mpimg.imread(conn_mat_path)
#             # # Confirm that figure aspect ratios are consistent
#             # assert raster_width / raster_height == raster_plot.shape[0] / raster_plot.shape[1]
#             # assert network_activity_width / network_activity_height == network_activity.shape[0] / network_activity.shape[1]
#             # assert locations_width / locations_height == TWODnet.shape[0] / TWODnet.shape[1]
#             # assert locations_width / locations_height == locs.shape[0] / locs.shape[1]
#             # assert raster_width / raster_height == conn_mat.shape[0] / conn_mat.shape[1]
#         except:
#             # if any exception, regenerate the images
#             print(f'Error loading existing images at: {NetBurst_path}')
#             pass       

#         # Attempt to generate sample trace for an excitatory example neuron
#         try:
#             try:
#                 assert fresh_figs == False
#                 # temp hack
#                 #assert fresh_figs == True
#                 assert os.path.exists(sample_trace_path_E)
#             except:
#                 # Prepare the sample trace
#                 sample_trace_E = sim.analysis.plotTraces(
#                     include=[('E', 0), ('I', 0)],
#                     overlay=True,
#                     oneFigPer='trace',
#                     title='Middlemost 1 second of simulation',
#                     timeRange=[start_time, end_time],
#                     #saveFig=sample_trace_path_E,
#                     showFig=False,
#                     figSize=(sample_trace_width, sample_trace_height)
#                 )
#                 # Add title to the figure
#                 fig = sample_trace_E[0]['_trace_soma_voltage']
#                 fig.suptitle('Middlemost 1 second of simulation')
#                 #move title all the way to the left
#                 fig.tight_layout(rect=[0, 0.03, 1, 1])
#                 # Save the figure with the title
#                 fig.savefig(sample_trace_path_E)
#                 #fig.suptitle('Middlemost 1 second of simulation')
#                 # Save the figure with the title
#                 #fig.savefig(sample_trace_path_E)
#                 # redo as png
#                 cairosvg.svg2png(url=sample_trace_path_E, write_to=sample_trace_path_E.replace('.svg', '.png'))
#         except:
#             print(f'Error generating sample trace plot from Data at: {file_path}')
#             sample_trace_path_E = None
#             pass
        


#         # Attempt to generate the raster plot
#         rasterData = sim.analysis.prepareRaster()
#         network_metrics = measure_network_activity(rasterData, binSize=binSize, gaussianSigma=gaussianSigma, thresholdBurst=thresholdBurst, crop = None)
#         base_locs = network_metrics['base_locs']
#         base_locs_time_range = [base_locs[0]*binSize, base_locs[-1]*binSize]
#         try:
#             try: 
#                 assert fresh_figs == False
#                 assert os.path.exists(raster_plot_path) 
#                 # raster_plot = mpimg.imread(raster_plot_path)
#                 # assert raster_width / raster_height == raster_plot.shape[1] / raster_plot.shape[0]
#             except: 
#                 sim.analysis.plotRaster(saveFig=raster_plot_path, 
#                                         #timeRange = raster_activity_timeRange,
#                                         timeRange = base_locs_time_range,
#                                         showFig=False,
#                                         labels = None, 
#                                         figSize=(raster_width, raster_height))#, dpi=600)
#                 #redo as png
#                 cairosvg.svg2png(url=raster_plot_path, write_to=raster_plot_path.replace('.svg', '.png'))
#         except:
#             print(f'Error generating raster plot from Data at: {file_path}')
#             raster_plot_path = None
#             pass
        
#         # # Attempt to generate the neuron locs plot
#         # try:
#         #     try: 
#         #         assert fresh_figs == False
#         #         assert os.path.exists(locs_path)
#         #         # locs = mpimg.imread(locs_path)
#         #         # assert locations_width / locations_height == locs.shape[1] / locs.shape[0]
#         #     except: 
#         #         sim.analysis.plot2Dnet(saveFig=locs_path, showFig=False, showConns=False, figSize=(locations_width, locations_height))
#         #         #redo as png
#         #         cairosvg.svg2png(url=locs_path, write_to=locs_path.replace('.svg', '.png'))
#         # except: 
#         #     print(f'Error generating neuron locs plot from Data at: {file_path}')
#         #     locs_path = None
#         #     pass

#         # Attempt to generate the network connections plot
#         try:
#             try: 
#                 assert fresh_figs == False
#                 assert os.path.exists(TWODnet_path)
#                 # locs = mpimg.imread(TWODnet_path)
#                 # assert locations_width / locations_height == TWODnet.shape[1] / TWODnet.shape[0]
#             except: 
#                 sim.analysis.plot2Dnet(saveFig=TWODnet_path, showFig=False, showConns=True, figSize=(locations_width, locations_height))
#                 #redo as png
#                 cairosvg.svg2png(url=TWODnet_path, write_to=TWODnet_path.replace('.svg', '.png'))
#         except: 
#             print(f'Error generating 2Dnet_conns from Data at: {file_path}')
#             TWODnet_path = None
#             pass

#         # Attempt to generate the connectivity matrix plot
#         try:
#             try: 
#                 assert fresh_figs == False
#                 assert os.path.exists(conn_mat_path)
#                 # locs = mpimg.imread(conn_mat_path)
#                 # assert conn_mat_width / conn_mat_height == conn_mat.shape[0] / conn_mat.shape[1]
#             except: 
#                 sim.analysis.plotConn(saveFig=conn_mat_path, showFig=False, figSize=(raster_width, raster_height))
#                 #redo as png
#                 cairosvg.svg2png(url=conn_mat_path, write_to=conn_mat_path.replace('.svg', '.png'))
#         except: 
#             print(f'Error generating connectivity matrix from Data at: {file_path}')
#             conn_mat_path = None
#             pass


#         # Attempt to generate the network activity plot
#         try:
#             try:
#                 assert fresh_figs == False
#                 assert os.path.exists(network_activity_path)
#                 # network_activity = mpimg.imread(network_activity_path)
#                 # assert network_activity_width / network_activity_height == network_activity.shape[1] / network_activity.shape[0]
#             except:
#                 #prepare raster data
#                 # rasterData = sim.analysis.prepareRaster()
#                 # network_metrics = measure_network_activity(rasterData, binSize=binSize, gaussianSigma=gaussianSigma, thresholdBurst=thresholdBurst, crop = None)
#                 # base_locs = network_metrics['base_locs']
#                 # base_locs_time_range = [base_locs[0], base_locs[-1]]

#                 # Generate the network activity plot with a size of (10, 5)
#                 plot_network_activity(
#                     rasterData, 
#                     binSize=binSize, 
#                     gaussianSigma=gaussianSigma, 
#                     thresholdBurst=thresholdBurst, 
#                     figSize=(network_activity_width, network_activity_height),
#                     #timeRange = raster_activity_timeRange, 
#                     timeRange = base_locs_time_range,
#                     saveFig=network_activity_path
#                 )
#                 #redo as png
#                 cairosvg.svg2png(url=network_activity_path, write_to=network_activity_path.replace('.svg', '.png'))
#         except:
#             print(f'Error generating network activity plot from Data at: {file_path}')
#             network_activity_path = None
#             pass

#         return raster_plot_path, sample_trace_path_I, sample_trace_path_E, network_activity_path, TWODnet_path, locs_path, conn_mat_path, NetBurst_path 

#     def generate_net_activity_raster(raster_plot_path, network_activity_path, NetBurst_path, batchrun_folder, batch_key):
        
#         net_activity_raster_path = f'{NetBurst_path}/{batch_key}_net_activity_raster.svg'
#         # height = raster_height + network_activity_height
#         # width = raster_width      
#         try: 
#             assert fresh_figs == False
#             #temp hack
#             assert fresh_figs == True
#             # net_activity_raster = mpimg.imread(net_activity_raster_path)
#             # assert height / width == net_activity_raster.shape[1] / net_activity_raster.shape[0]
#             assert os.path.exists(net_activity_raster_path)
#             return net_activity_raster_path
#         except:            
#             try: 
#                 # Load the images
#                 # Load the SVG files
#                 raster_plot = sg.fromfile(raster_plot_path)
#                 network_activity = sg.fromfile(network_activity_path)
#                 # raster_plot = mpimg.imread(raster_plot_path)
#                 # network_activity = mpimg.imread(network_activity_path)
#                 #TWODnet = mpimg.imread(TWODnet_path)
                
#                 # Create a new figure
#                 # Create a new SVG figure
#                 #fig = sg.SVGFigure(width, height)

#                 # Get the root of each SVG file
#                 plot1 = raster_plot.getroot()
#                 plot2 = network_activity.getroot()
#                 #plot1.scale(1.125)
                
#                 # Get the size of the first SVG
#                 svg1_width = float(raster_plot.width.rstrip('pt'))
#                 svg1_height = float(raster_plot.height.rstrip('pt'))

#                 # Get the size of the second SVG
#                 svg2_width = float(network_activity.width.rstrip('pt'))
#                 svg2_height = float(network_activity.height.rstrip('pt'))

#                 # Scale the second SVG to match the width of the first SVG
#                 # scale image widths so that axes line up
#                 scale_factor = svg1_width / svg2_width
#                 plot2.scale(scale_factor)
#                 #plot2.scale(.75, 1)

#                 #plot2.scale(scale_factor)

#                 # Move the second plot to the right by the width of the first plot plus a small margin
#                 plot2.moveto(0, svg1_height + 10)  # Adjust the margin as needed

#                 # Create a new SVG figure with the exact size needed to accommodate the figures
#                 fig_width = max(svg1_width, svg2_width*scale_factor)
#                 fig_height = svg1_height + svg2_height*scale_factor + 10  # Add the margin
#                 # #move both plots down a little to make space for title.
#                 # plot1.moveto(0, 10)
#                 # plot2.moveto(0, 10)  # Adjust the margin as needed
#                 fig = sg.SVGFigure(fig_width, fig_height)

#                 # Set the width and height attributes of the SVGFigure
#                 fig.set_size((str(fig._width), str(fig._height)))

#                 # Add the plots to the figure
#                 fig.append([plot1, plot2])

#                 # Adjust layout to fit subplots tightly
#                 #plt.tight_layout()
#                 #fig.suptitle(f'{batchrun_folder}{batch_key}', fontsize=16)
#                 # Create a title
#                 #title = f'{batchrun_folder}{batch_key}'
#                 # Calculate the center of the figure
#                 #center_x = svg1_width / 2

#                 # Create a title
#                 #title = f'{batchrun_folder}{batch_key}'
#                 #title_text = sg.TextElement(center_x, 20, title, size=10, weight="bold", anchor="middle")  # Adjust the position and style as needed

#                 # Add the title to the figure
#                 #fig.append(title_text)
#                 #net_activity_raster_path = f'{NetBurst_path}/{batch_key}_net_activity_raster.png'      
#                 # plt.savefig(net_activity_raster_path, dpi=600, bbox_inches='tight')
#                 # plt.savefig(net_activity_raster_path.replace('.png', '.svg'), bbox_inches='tight')
#                 #plt.savefig(net_activity_raster_path, bbox_inches='tight')
#                 fig.save(net_activity_raster_path)
#                 # Convert SVG to PNG
#                 png_path = net_activity_raster_path.replace('.svg', '.png')
#                 cairosvg.svg2png(url=net_activity_raster_path, write_to=png_path)
#                 return net_activity_raster_path

#             except:
#                 print(f'Error generating net_activity_raster figure at: {NetBurst_path}')
#                 return None
#             #pass

#     def generate_conn_summarry_fig(sample_trace_path_E, conn_mat_path, TWODnet_path, NetBurst_path, batchrun_folder, batch_key):
    
#         conn_summary_fig_path = f'{NetBurst_path}/{batch_key}_conn_summary_fig.svg' 
#         height = raster_height + locations_height
#         width = locations_width      
#         try: 
#             assert fresh_figs == False
#             assert os.path.exists(conn_summary_fig_path)
#             return conn_summary_fig_path
#         except:            
#             try: 
#                 # Load the SVG files
#                 conn_mat = sg.fromfile(conn_mat_path)
#                 TWODnet = sg.fromfile(TWODnet_path)
#                 activity_trace = sg.fromfile(sample_trace_path_E)

#                 # Get the root of each SVG file
#                 plot1 = TWODnet.getroot()
#                 plot2 = conn_mat.getroot()
#                 plot3 = activity_trace.getroot()

#                 # Measure the height and width of the SVGs
#                 svg1_height = float(TWODnet.height.rstrip('pt'))
#                 svg1_width = float(TWODnet.width.rstrip('pt'))
#                 svg2_height = float(conn_mat.height.rstrip('pt'))
#                 svg2_width = float(conn_mat.width.rstrip('pt'))
#                 svg3_height = float(activity_trace.height.rstrip('pt'))
#                 svg3_width = float(activity_trace.width.rstrip('pt'))

#                 # Calculate the scale factors
#                 scale_factor_height = max(svg2_height, svg3_height) / svg1_height
#                 scale_factor_width = (svg2_width + svg3_width + 5) / svg1_width  # 5 is the space between the plots

#                 # Scale the top plot
#                 plot1.scale(scale_factor_width, scale_factor_height)

#                 # Move the second plot down by the height of the first plot plus a small margin
#                 plot2.moveto(0, svg1_height * scale_factor_height + 10)  # Adjust the margin as needed

#                 # Move the third plot down by the height of the first plot plus a small margin, and to the right by the width of the second plot
#                 plot3.moveto(svg2_width + 5, svg1_height * scale_factor_height + 10)  # Adjust the margin as needed

#                 # Create a new SVG figure
#                 fig = sg.SVGFigure(svg1_width * scale_factor_width, svg1_height * scale_factor_height + max(svg2_height, svg3_height) + 10)

#                 # Set the width and height attributes of the SVGFigure
#                 fig.set_size((str(fig._width), str(fig._height)))

#                 # Add the plots to the figure
#                 fig.append([plot1, plot2, plot3])

#                 # Save the figure
#                 fig.save(conn_summary_fig_path)
#                 # Convert SVG to PNG
#                 png_path = conn_summary_fig_path.replace('.svg', '.png')
#                 cairosvg.svg2png(url=conn_summary_fig_path, write_to=png_path)
#                 return conn_summary_fig_path

#             except:
#                 print(f'Error generating conn_summary figure at: {NetBurst_path}')
#             return None
#             #pass

#     def generate_param_summary_fig(conn_summary_path, net_activity_raster_path, NetBurst_path, batchrun_folder, batch_key):
#         param_summary_fig_path = f'{NetBurst_path}/{batch_key}_param_summary_fig.svg'
#         width  = raster_width + locations_width
#         height = conn_mat_height + locations_height 
#         try: 
#             assert fresh_figs == False
#             assert fresh_figs == True
#             assert os.path.exists(param_summary_fig_path)
#             return param_summary_fig_path
#         except:            
#             try: 
#                 # Load the SVG files
#                 conn_summary = sg.fromfile(conn_summary_path)
#                 net_activity_raster = sg.fromfile(net_activity_raster_path)

#                 # Get the root of each SVG file
#                 plot1 = conn_summary.getroot()
#                 plot2 = net_activity_raster.getroot()

#                 # Set the width and height of the first SVG
#                 # svg1_width = 1000.0  # Replace with your desired width
#                 # svg1_height = 500.0  # Replace with your desired height
#                 # conn_summary.set_size((str(svg1_width), str(svg1_height)))
#                 svg1_width = float(conn_summary.width.rstrip('pt')) 
#                 svg1_height = float(conn_summary.height.rstrip('pt'))

#                 # Set the width and height of the second SVG
#                 # svg2_width = svg1_width / 2  # Half the width of the first SVG
#                 # svg2_height = svg1_height  # Same height as the first SVG
#                 # net_activity_raster.set_size((str(svg2_width), str(svg2_height)))
#                 svg2_width = float(net_activity_raster.width.rstrip('pt'))
#                 svg2_height = float(net_activity_raster.height.rstrip('pt'))

#                 # Scale the second SVG to equal height                
                
#                 # Move the second plot to the right by the width of the first plot plus a small margin
#                 plot2.moveto(svg1_width+10, 0)  # Adjust the margin as needed

#                 # Create a new SVG figure
#                 #fig = sg.SVGFigure(width, height)
#                 fig = sg.SVGFigure(svg1_width + svg2_width + 5, svg1_height)

#                 # Set the width and height attributes of the SVGFigure
#                 fig.set_size((str(fig._width), str(fig._height)))

#                 # Add the plots to the figure
#                 fig.append([plot1, plot2])

#                 # Create a title
#                 #title = f'{batchrun_folder}{batch_key}'
#                 #title_text = sg.TextElement(5, 20, title, size=12, weight="bold")  # Adjust the position and style as needed

#                 # Add the title to the figure
#                 #fig.append(title_text)

#                 # Save the figure
#                 fig.save(param_summary_fig_path)
#                 # Convert SVG to PNG
#                 png_path = param_summary_fig_path.replace('.svg', '.png')
#                 cairosvg.svg2png(url=param_summary_fig_path, write_to=png_path)
                
#                 return param_summary_fig_path
          

#             except:
#                 print(f'Error generating param_summary figure at: {NetBurst_path}')
#                 return None

#     def generate_param_summary_row(file_path, TWODnet_path, conn_mat_path, sample_trace_path_E, raster_plot_path, network_activity_path, NetBurst_path, batch_key):
#         #imports
#         import json

#         ## functions
#         def load_params():
#             ##add text            
#             params = None
#             grandparent_folder = os.path.dirname(os.path.dirname(NetBurst_path))    
#             try:
#                 with open(f'{grandparent_folder}/param_space.json') as f:
#                     params = json.load(f)
#             except:
#                 with open(f'{grandparent_folder}/params.json') as f:
#                     params = json.load(f)
#             return params

#         #Code
#         param_summary_row_path = f'{NetBurst_path}/{batch_key}_param_summary_row.svg'
#         #param_summary_row_path = f'{output_folder}/{output_key}_param_summary_row.svg'
#         #if not os.path.exists(param_summary_row_path):
#         if not fresh_figs:
#             param_summary_row_png_path = param_summary_row_path.replace('.svg', '.png')
#             if os.path.exists(param_summary_row_png_path) and os.path.exists(param_summary_row_path):
#                 print("Param Summary row already exists for this candidate")
#                 return param_summary_row_path
#         try: 
#             print('Generating Param Summary Row...')
#             # Load the SVG files
#             TwoDNet_plot = sg.fromfile(TWODnet_path)
#             conn_mat = sg.fromfile(conn_mat_path)
#             soma_voltage_trace = sg.fromfile(sample_trace_path_E)
#             raster_plot = sg.fromfile(raster_plot_path)
#             net_activity_plot = sg.fromfile(network_activity_path)

#             # Get the root of each SVG file
#             plot1 = TwoDNet_plot.getroot()
#             plot2 = conn_mat.getroot()
#             plot3 = soma_voltage_trace.getroot()
#             plot4 = raster_plot.getroot()
#             plot5 = net_activity_plot.getroot()

#             # Measure the height and width of the SVGs
#             svg_width = float(raster_plot.width.rstrip('pt'))
#             svg_height = float(raster_plot.height.rstrip('pt'))

#             # Scale the net_activity_plot to match the height of the raster_plot
#             net_activity_height = float(net_activity_plot.height.rstrip('pt'))
#             scale_factor = svg_height / net_activity_height
#             plot5.scale(1, scale_factor)

#             # Move the plots to the right by the width of the previous plot plus a small margin
#             plot1.moveto(svg_width*2 + 10, 0)  # Adjust the margin as needed
#             plot2.moveto(4 * svg_width + 10, 0)  # Adjust the margin as needed
#             plot3.moveto(5 * svg_width + 10, 0)  # Adjust the margin as needed
#             plot4.moveto(6 * svg_width + 10, 0)  # Adjust the margin as needed
#             plot5.moveto(7 * svg_width + 10, 0)  # Adjust the margin as needed

#             # Create a new SVG figure
#             fig = sg.SVGFigure(8 * svg_width + 60, svg_height*2)  # 60 is the total margin

#             # Set the width and height attributes of the SVGFigure
#             fig.set_size((str(fig._width), str(fig._height)))

#             # Add the plots to the figure
#             fig.append([plot1, plot2, plot3, plot4, plot5])

#             # Load params.json
#             params = load_params()

#             # Load batch config file
#             cfg_file = file_path.replace('_data.json', '_cfg.json')
#             with open(cfg_file) as f:
#                 cfg = json.load(f)
#             cfg = cfg['simConfig']

#             # Calculate the height of each row
#             row_height = svg_height / len(params)
#             # Create a new SVG figure
#             param_fig = sg.SVGFigure(svg_width*2, row_height * len(params)*2)  # Increase the height of the figure

#             from svgwrite import Drawing

#             # Create a new SVG file with a white rectangle as the background
#             dwg = Drawing(f'{NetBurst_path}/background.svg', profile='tiny')
#             dwg.add(dwg.rect(insert=(0, 0), size=(svg_width*2, svg_height*2), fill='white'))  # Increase the height of the background
#             dwg.save()

#             # Load the background SVG file
#             background = sg.fromfile(f'{NetBurst_path}/background.svg').getroot()
#             os.remove(f'{NetBurst_path}/background.svg')

#             # Add the background to the SVG figure
#             fig.append(background)
#             # For each key in params
#             # For each key in params
#             for i, key in enumerate(params.keys()):
#                 # Prepare a row and a title
#                 row_y = i * row_height * 2  # Increase the space between rows
#                 title = sg.TextElement(5, row_y + 20, key, size=10, weight="bold")  # Reduce the font size

#                 # Write the range in the figure
#                 range_pos = (svg_width / 4)*1.2
#                 range_y_pos = row_y + 20
#                 range_text = sg.TextElement(range_pos, range_y_pos, f"Range: {params[key]}", size=10)  # Move to the second column

#                 # Find a matching value in cfg and write it in the figure
#                 if key in cfg:
#                     value_pos = range_pos
#                     value_y_pos = range_y_pos + 15
#                     value_text = sg.TextElement(value_pos, value_y_pos, f"Value: {cfg[key]}", size=10)  # Move to the third column

#                 # Plot a 1D plot of the value within each respective range
#                 plots_pos = value_pos + (svg_width / 4)*1.5
#                 param_fig, ax = plt.subplots(figsize=(svg_width / 80, row_height / 55))  # Reduce the width of the plot

#                 # Calculate the thickness of the line
#                 range_width = params[key][1] - params[key][0]
                 
#                 # value = cfg[key]
#                 # frange_width = (params[key][1] - value) / 100
#                 if range_width == 0:
#                     #full line thickness
#                     near_zero = 0.000001
#                     line_thickness = 1/near_zero
#                 else:
#                     assert range_width != 0
#                     assert range_width > 0
#                     # max_line_thickness = range_width/2
#                     # range_width_frac = 0.01*max_line_thickness
#                     # line_thickness = 1/normalized_range_frac
#                     line_thickness = (1 / range_width)*0.1 if (1 / range_width)*0.1>1 else 1

#                 #adjust x axes limits
#                 ax.set_xlim(params[key][0], params[key][1])

#                 # Draw a vertical line at the value
#                 ax.axvline(x=cfg[key], linewidth=line_thickness, color='r')

#                 # # Set the direction of the y-ticks to out
#                 #x.tick_params(axis='y', direction='out')                

#                 # Display the range on the left and right of the plot
#                 # ax.text(params[key][0], 0.125, str(params[key][0]), ha='left', va='center', transform=ax.get_yaxis_transform())
#                 # ax.text(params[key][1], 0.125, str(params[key][1]), ha='right', va='center', transform=ax.get_yaxis_transform())

#                 #exclude y axes numbers, include x axes numbers
#                 ax.set_yticklabels([])
#                 #ax.set_xticklabels([params[key][0], params[key][1]])
#                 # Reset the x-ticks to the default
#                 ax.locator_params(axis='x', nbins='auto')

#                 # Save the plot
#                 plt.savefig(f"{NetBurst_path}/temp_plot.svg")
#                 plot = sg.fromfile(f"{NetBurst_path}/temp_plot.svg").getroot()
#                 os.remove(f"{NetBurst_path}/temp_plot.svg")
#                 plot.moveto(plots_pos, row_y)  # Move to the fourth column

#                 # Add the elements to the figure
#                 fig.append([title, range_text, value_text, plot])

#             #for each key in params, prepare a row and a title
#             #for each key in params define, there is a list [min, max]
#             #for each key in params, write the range in the figure
#             #for each key in params, find a matching value in cfg and write it in the figure
#             #for each key in params, plot a 1D plot of the value within each respective range. 
#             #fit each plot into it's respective row
#             #make sure all of this fits wihtin 1 raster heightxwidth

#             # Load batch fitness data
#             fitness_file = file_path.replace('_data.json', '_Fitness.json')
#             with open(fitness_file) as f:
#                 fitness = json.load(f)

#             # Print each key and value in fitness
#             for key, value in fitness.items():
#                 print(f"Key: {key}, Value: {value}")

#             # Respect plot x position. move a bit to the right.
#             plots_pos += svg_width / 4

#             # Print from top
#             i = 0
#             fit_pos = plots_pos + (svg_width / 4)*2.5
#             # Handle "maxFitness" first
#             if 'maxFitness' in fitness:
#                 value = fitness['maxFitness']
#                 row_y = i * row_height * 2  # Increase the space between rows
#                 i += 1
#                 text = sg.TextElement(fit_pos, row_y, f"maxFitness: {value:.3f}", size=12)
#                 fig.append(text)
#                 row_y += row_height
#                 i += 1

#             # Then handle the rest of the keys
#             for key, value in fitness.items():
#                 if key == 'maxFitness' or key == 'fitness':
#                     continue  # Skip "maxFitness" because we've already handled it
#                 row_y = i * row_height * 2  # Increase the space between rows
#                 i += 1
#                 # Create a text element for the key and value
#                 text = sg.TextElement(fit_pos, row_y, f"{key}: {value:.3f}", size=12)
#                 # Add the text element to the figure
#                 fig.append(text)
#                 # Move to the next row
#                 row_y += row_height

#             #make sure fitness goes last
#             if 'fitness' in fitness:
#                 row_y = i * row_height * 2  # Increase the space between rows
#                 value = fitness['fitness']
#                 text = sg.TextElement(fit_pos, row_y, f"fitness: {value:.3f}", size=12, weight="bold")
#                 fig.append(text)

#             # if 'Fitness' not in fitness:
#             #     #calculate fitness by averaging values in fitness
#             #     fitness['Fitness'] = sum(fitness.values()) / len(fitness)

#             print(f'Saving... {param_summary_row_path}')
#             # Save the figure
#             fig.save(param_summary_row_path)
#             # Convert SVG to PNG
#             png_path = param_summary_row_path.replace('.svg', '.png')
#             cairosvg.svg2png(url=param_summary_row_path, write_to=png_path) 
            
#             return param_summary_row_path

#         except Exception as e:
#             print(f'Error generating param_summary_row figure at: {NetBurst_path}')
#             print(e)
#             return None

    def main():
        #get file info
        file_path, batchrun_folder, batch_key = get_batchrun_info(root, file)
        
        raster_plot_path, sample_trace_path_I, sample_trace_path_E, network_activity_path, TWODnet_path, locs_path, conn_mat_path, NetBurst_path = generate_primary_figures(
            root, file,  size = size, raster_activity_timeRange = raster_activity_timeRange,
            net_activity_params = net_activity_params)
        # net_activity_raster_path = generate_net_activity_raster(
        #     raster_plot_path, network_activity_path, NetBurst_path, batchrun_folder, batch_key) 
        # conn_summary_fig_path = generate_conn_summarry_fig(
        #     sample_trace_path_E,
        #     conn_mat_path, TWODnet_path, NetBurst_path, batchrun_folder, batch_key)
        # param_summary_fig_path = generate_param_summary_fig(
        #     conn_summary_fig_path, net_activity_raster_path, NetBurst_path, batchrun_folder, batch_key)
        param_summary_row_path = generate_param_summary_row(
            file_path, TWODnet_path, conn_mat_path, sample_trace_path_E, raster_plot_path, network_activity_path, NetBurst_path, batch_key)
    
    #Simulation plots will be saved in the batch folder as png files
    #Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.dirname(script_dir)
    output_path = f'{output_path}/output' 
    if batch_path is None:
        batch_path = f'{output_path}/{batchLabel}'
    else:
        batch_path = batch_path    
    
    #for each file named *_cfg.json in the batch folder, get raster data
    #for file in os.listdir(batch_path):
    #walk through the batch folder and subfolders
    # if batchpath is directory
    if os.path.isdir(batch_path):
        for root, dirs, files in os.walk(batch_path):
            for file in files:
                if file.endswith("_data.json"):
                    main()
    #if batchpath is file
    elif batch_path.endswith("_data.json"):
        file = batch_path
        root = os.path.dirname(batch_path)
        main()
    #if batchpath is not a valid directory or file
    else:
        print(f'Error: {batch_path} is not a valid directory or file')

