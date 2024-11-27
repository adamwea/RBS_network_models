import numpy as np
import statistics as stats
from workspace.RBS_network_simulations._archive.temp_user_args import *
from scipy.signal import convolve, find_peaks, butter, filtfilt
from scipy.stats import linregress, norm, stats, gaussian_kde
from helper_functions import load_clean_sim_object
import netpyne
#from plotting_functions import *

'''optimizing functions'''
def analyze_network_activity(rasterData, conv_params = None):
    '''subfunctions'''
    def bimodality_metric(values, bandwidth='scott'):
        # Estimate the density using KDE
        kde = gaussian_kde(values, bw_method=bandwidth)
        x = np.linspace(min(values), max(values), 1000)
        density = kde(x)
        
        # Find local maxima (modes) in the density estimate
        peaks, _ = find_peaks(density)
        
        # Ensure there are at least two peaks (modes)
        if len(peaks) < 2:
            return None, None  # Not bimodal
        
        # Identify the local minimum between the two highest modes
        peak_values = density[peaks]
        highest_peaks = peaks[np.argsort(peak_values)[-2:]]  # Get indices of the two highest peaks
        min_index = min(highest_peaks)
        max_index = max(highest_peaks)
        
        # Find the local minimum between the two highest peaks
        local_min = np.argmin(density[min_index:max_index]) + min_index
        
        # Calculate the bimodality metric (difference between the heights of the modes and the local minimum)
        bimodality_metric = (density[highest_peaks[0]] + density[highest_peaks[1]]) / (2 * density[local_min])
        
        return bimodality_metric, x[local_min]

        # # Example usage:
        # values = np.random.normal(loc=0, scale=1, size=500).tolist() + np.random.normal(loc=5, scale=1, size=500).tolist()
        # metric, threshold = bimodality_metric(values)
        # print(f"Bimodality Metric: {metric}")
        # print(f"Intersection Threshold: {threshold}")
    def apply_min_distance_between_peaks():
        try:
            assert len(burstPeakValues) > 1, 'There must be at least 2 peaks to adjust minimum peak distance'
            if min_peak_distance is None: min_peak_distance = 1 #seconds
            else: min_peak_distance = min_peak_distance #seconds
            #min_peak_distance /= 1000 #convert to seconds
            #avg_IBI = np.mean(np.diff(burstPeakTimes))
            #print(f'Average IBI before 10s correction: {avg_IBI}')
            #avg_IBI = np.mean(np.diff(burstPeakTimes))
            #print(f'Average IBI before 10s correction: {avg_IBI}')
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
            #avg_IBI = np.mean(np.diff(burstPeakTimes))
            #print(f'Average IBI after 10s correction: {avg_IBI}')
        except Exception as e: 
            print(e)
            pass
    
    ''' Init'''
    assert rasterData is not None, 'rasterData must be specified'

    '''convolution parameters'''    
    default_params = {'binSize': .1, 'gaussianSigma': .15, 'thresholdBurst': 1.0, 'min_peak_distance': 1,} #seconds
    if conv_params is None: conv_params = default_params
    assert 'binSize' in conv_params, 'binSize must be specified'
    assert 'gaussianSigma' in conv_params, 'gaussianSigma must be specified'
    assert 'thresholdBurst' in conv_params, 'thresholdBurst must be specified'
    #min_peak_distance = None, binSize=None, gaussianSigma=None, 
    #thresholdBurst = None, plot=False, plotting_params = None, crop = None): # thresholdBurst=1.2, crop = None): #, figSize=(10, 6), saveFig=False, figName='NetworkActivity.png'):

    '''Get relative spike time data'''
    assert 'spkt' in rasterData or 'spkTimes' in rasterData, 'Error: No spike times found in rasterData'
    if 'spkt' in rasterData: SpikeTimes = rasterData['spkt']
    elif 'spkTimes' in rasterData: SpikeTimes = rasterData['spkTimes']
    else: return None
    relativeSpikeTimes = (np.array(SpikeTimes) - SpikeTimes[0])#/1000  # Set the first spike time to 0 and convert to seconds

    ''' Convolve Spike Data '''    
    # Step 1: Bin all spike times into small time windows
    binSize = conv_params['binSize']
    timeVector = np.arange(0, max(relativeSpikeTimes) + binSize, binSize)  # Time vector for binning
    binnedTimes, _ = np.histogram(relativeSpikeTimes, bins=timeVector)  # Bin spike times
    binnedTimes = np.append(binnedTimes, 0)  # Append 0 to match MATLAB's binnedTimes length

    # Step 2: Smooth the binned spike times with a Gaussian kernel
    gaussianSigma = conv_params['gaussianSigma']
    kernelRange = np.arange(-3*gaussianSigma, 3*gaussianSigma + binSize, binSize)  # Range for Gaussian kernel
    kernel = norm.pdf(kernelRange, 0, gaussianSigma)  # Gaussian kernel
    kernel *= binSize  # Normalize kernel by bin size
    firingRate = convolve(binnedTimes, kernel, mode='same') / binSize  # Convolve and normalize by bin size

    # Step 4: Peak detection on the smoothed and cropped firing rate curve
    min_peak_distance = conv_params['min_peak_distance']
    thresholdBurst = conv_params['thresholdBurst']
    rmsFiringRate = np.sqrt(np.mean(firingRate**2))  # Calculate RMS of the firing rate
    min_distance_samples = min_peak_distance / binSize
    peaks, properties = find_peaks(firingRate, height=thresholdBurst * rmsFiringRate, distance=min_distance_samples)  # Find peaks above the threshold
    #convert peak indices to times
    burstPeakTimes = timeVector[peaks]  # Convert peak indices to times
    burstPeakValues = properties['peak_heights']  # Get the peak values

    # Step 5: Apply minimum peak distance
    #apply_min_distance_between_peaks()
        
    '''
    Network Metric Outputs
    '''
    #fireing_rate
    firing_rate = firingRate

    # Calculate Baseline
    try: baseline = np.mean(firing_rate)
    except: baseline = None

    #time_vector
    time_vector = timeVector

    #measure frequency of peaks
    try: 
        assert len(burstPeakTimes) > 0, 'Error: No burst peaks found. peak_freq set to None.'
        assert len(burstPeakTimes) > 1, 'Error: Only one burst peak found. peak_freq set to None.'
        burst_freq = len(burstPeakTimes) / (time_vector[-1]) #now in seconds
    except: burst_freq = None

    #burst_times
    try: burst_times = np.array(burstPeakTimes)
    except: burst_times = None

    #burst_vals
    try: burst_vals = np.array(burstPeakValues)
    except: burst_vals = None

    #find biomdality in burst_vals
    try: bimodality_val, bimodality_threshold = bimodality_metric(burst_vals)
    except: bimodality_val, bimodality_threshold = None, None

    #big_burst_vals
    try: big_burst_vals = burst_vals[burst_vals > bimodality_threshold]
    except: big_burst_vals = None

    #big_burst_times
    try: big_burst_times = burst_times[burst_vals > bimodality_threshold]
    except: big_burst_times = None

    #small_burst_vals
    try: small_burst_vals = burst_vals[burst_vals < bimodality_threshold]
    except: small_burst_vals = None

    #small_burst_times
    try: small_burst_times = burst_times[burst_vals < bimodality_threshold]
    except: small_burst_times = None
    
    #measure IBI
    try: IBIs = np.diff(burst_times) 
    except: IBIs = None

    #burst_threshold
    burst_threshold = thresholdBurst * rmsFiringRate

    '''return measurements'''
    measurements = {
        'burst_vals': burst_vals, #-(thresholdBurst * rmsFiringRate),
        'burst_times': burst_times,
        'IBIs': IBIs,
        'firing_rate': firing_rate,
        'time_vector': time_vector,
        'baseline': baseline,
        'burst_freq': burst_freq,
        'burst_threshold': burst_threshold,
        'bimodal_val': bimodality_val,
        'bimodal_threshold': bimodality_threshold,
        'big_burst_vals': big_burst_vals,
        'big_burst_times': big_burst_times,
        'small_burst_vals': small_burst_vals,
        'small_burst_times': small_burst_times,
    }    
    return measurements 


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
            from plotting_functions import plot_network_activity
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
def get_network_activity_metrics(simData, exp_mode = False):
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
        assert len(rasterData['spkt']) > 0, 'Error: rasterData has no elements. burstPeak, baseline, slopeFitness, and IBI fitness set to 1000.'                       
        # Generate the network activity plot with a size of (10, 5)
        plotting_params = None
        plot = False
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
def get_individual_neuron_metrics(data_file_path, exp_mode = False):
    '''subfunctions'''
    def get_spike_data_from_simulated_data():
        load_clean_sim_object(data_file_path)
        spike_data = netpyne.sim.analysis.prepareSpikeData()
        E_neurons = [ind for ind in spike_data['cellGids'] if spike_data['cellPops'][ind] == 'E']
        I_neurons = [ind for ind in spike_data['cellGids'] if spike_data['cellPops'][ind] == 'I']
        return spike_data, E_neurons, I_neurons
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

        return spike_data, firing_rates, E_neurons, I_neurons
    '''main function'''
    print('Calculating individual firing rates and average ISIs...')
    try:        
        if not exp_mode: spike_data, E_neurons, I_neurons = get_spike_data_from_simulated_data()            
        elif exp_mode: spike_data, firing_rates, E_neurons, I_neurons = get_spike_data_from_experimental_data()          
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
            'I_average_ISIs': I_average_ISIs,
            'E_neurons': E_neurons,
            'I_neurons': I_neurons,
        }
        print('Individual firing rates and ISIs calculated.')
        return neuron_metrics
    except Exception as e:
        print(f'Error calculating individual firing rates and average ISIs.')
        print(f'Error: {e}')
        return None