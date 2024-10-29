import numpy as np
from scipy.signal import convolve, find_peaks
from scipy.stats import norm
import matplotlib.pyplot as plt

def init_convolution_params(binSize=None, gaussianSigma=None, thresholdBurst=None, min_peak_distance=None):
    try:
        from simulate._config_files.convolution_params import conv_params
        conv_params_temp = conv_params
        conv_params_temp['binSize'] = binSize if binSize is not None else conv_params['binSize']
        conv_params_temp['gaussianSigma'] = gaussianSigma if gaussianSigma is not None else conv_params['gaussianSigma']
        conv_params_temp['thresholdBurst'] = thresholdBurst if thresholdBurst is not None else conv_params['thresholdBurst']
        conv_params_temp['min_peak_distance'] = min_peak_distance if min_peak_distance is not None else conv_params['min_peak_distance']
    except:
        conv_params_temp = {
            'binSize': 0.1 if binSize is None else binSize,
            'gaussianSigma': 0.15 if gaussianSigma is None else gaussianSigma,
            'thresholdBurst': 1.0 if thresholdBurst is None else thresholdBurst,
            'min_peak_distance':1 if min_peak_distance is None else min_peak_distance,
        }
    
    conv_params = conv_params_temp
    return conv_params

def init_burst_analysis_params(isi_threshold=None):
    try:
        from simulate._config_files.burst_analysis_params import burst_analysis_params
        burst_analysis_params_temp = burst_analysis_params
        burst_analysis_params_temp['isi_threshold'] = isi_threshold if isi_threshold is not None else burst_analysis_params['isi_threshold']
    except:
        burst_analysis_params_temp = {
            'isi_threshold': 0.1 if isi_threshold is None else isi_threshold,
        }
    
    burst_analysis_params = burst_analysis_params_temp
    return burst_analysis_params

def convolve_network_activity(rasterData, binSize=None, gaussianSigma=None, thresholdBurst=None, min_peak_distance=None):

    conv_params = init_convolution_params(binSize, gaussianSigma, thresholdBurst, min_peak_distance)
    binSize = conv_params['binSize']
    gaussianSigma = conv_params['gaussianSigma']
    thresholdBurst = conv_params['thresholdBurst']
    min_peak_distance = conv_params['min_peak_distance']
    
    relativeSpikeTimes = np.array(rasterData['spkTimes'])
    relativeSpikeTimes -= relativeSpikeTimes[0]
    
    timeVector = np.arange(0, max(relativeSpikeTimes) + binSize, binSize)
    binnedTimes, _ = np.histogram(relativeSpikeTimes, bins=timeVector)
    binnedTimes = np.append(binnedTimes, 0)

    kernelRange = np.arange(-3*gaussianSigma, 3*gaussianSigma + binSize, binSize)
    kernel = norm.pdf(kernelRange, 0, gaussianSigma)
    kernel *= binSize
    spike_counts_at_time = convolve(binnedTimes, kernel, mode='same') / binSize

    rmsSpikeCounts = np.sqrt(np.mean(spike_counts_at_time**2))
    peaks, properties = find_peaks(spike_counts_at_time, height=thresholdBurst * rmsSpikeCounts, distance=min_peak_distance)

    burstPeakTimes = timeVector[peaks]
    burstPeakValues = properties['peak_heights']

    return burstPeakTimes, burstPeakValues, spike_counts_at_time, timeVector, rmsSpikeCounts


''' Use Functions from MEA Analysis Repository'''
import submodules.MEA_Analysis.IPNAnalysis.helper_functions as helper

def extract_number_bursts(spike_times, isi_threshold=0.1):
    burst_statistics = helper.detect_bursts_statistics(spike_times, isi_threshold)
    return sum(len(unit_stats['bursts']) for unit_stats in burst_statistics.values())

def extract_mean_ibi(spike_times, isi_threshold=0.1):
    burst_statistics = helper.detect_bursts_statistics(spike_times, isi_threshold)
    all_isis = np.concatenate([stats['isis_all'] for stats in burst_statistics.values() if stats['isis_all'].size > 0])
    return np.mean(all_isis) if all_isis.size > 0 else np.nan

def extract_cov_ibi(spike_times, isi_threshold=0.1):
    burst_statistics = helper.detect_bursts_statistics(spike_times, isi_threshold)
    all_isis = np.concatenate([stats['isis_all'] for stats in burst_statistics.values() if stats['isis_all'].size > 0])
    return np.cov(all_isis) if all_isis.size > 0 else np.nan

def extract_mean_burst_peak(network_data):
    return network_data['mean_Burst_Peak']

def extract_cov_burst_peak(network_data):
    return network_data['cov_Burst_Peak']

def extract_fano_factor(network_data):
    return network_data['fano_factor']

def extract_mean_within_burst_isi(network_data):
    return network_data['MeanWithinBurstISI']

def extract_cov_within_burst_isi(network_data):
    return network_data['CoVWithinBurstISI']

def extract_mean_outside_burst_isi(network_data):
    return network_data['MeanOutsideBurstISI']

def extract_cov_outside_burst_isi(network_data):
    return network_data['CoVOutsideBurstISI']

def extract_mean_network_isi(network_data):
    return network_data['MeanNetworkISI']

def extract_cov_network_isi(network_data):
    return network_data['CoVNetworkISI']

def extract_num_units(network_data):
    return network_data['NumUnits']

def extract_all_metrics(spike_times, network_data, isi_threshold=0.1):
    metrics = {
        'Number_Bursts': extract_number_bursts(spike_times, isi_threshold),
        'mean_IBI': extract_mean_ibi(spike_times, isi_threshold),
        'cov_IBI': extract_cov_ibi(spike_times, isi_threshold),
        'mean_Burst_Peak': extract_mean_burst_peak(network_data),
        'cov_Burst_Peak': extract_cov_burst_peak(network_data),
        'fano_factor': extract_fano_factor(network_data),
        #'MeanWithinBurstISI': extract_mean_within_burst_isi(network_data),
        #'CoVWithinBurstISI': extract_cov_within_burst_isi(network_data),
        #'MeanOutsideBurstISI': extract_mean_outside_burst_isi(network_data),
        #'CoVOutsideBurstISI': extract_cov_outside_burst_isi(network_data),
        #'MeanNetworkISI': extract_mean_network_isi(network_data),
        #'CoVNetworkISI': extract_cov_network_isi(network_data),
        #'NumUnits': extract_num_units(network_data)
    }
    return metrics


'''Main logic of the measure_network_activity function'''

def analyze_burst_activity(spike_times):
    
    #init convolution parameters
    conv_params = init_convolution_params()
    binSize = conv_params['binSize']
    gaussianSigma = conv_params['gaussianSigma']
    thresholdBurst = conv_params['thresholdBurst']
    min_peak_distance = conv_params['min_peak_distance']
    
    #init burst analysis parameters
    burst_analysis_params = init_burst_analysis_params()
    isi_threshold = burst_analysis_params['isi_threshold']
    
    #network analysis
    fig, ax = plt.subplots()
    ax, network_data = helper.plot_network_activity(ax, spike_times, min_peak_distance=min_peak_distance, binSize=binSize, gaussianSigma=gaussianSigma, thresholdBurst=thresholdBurst)
    
    #burst analysis - slightly modified code from mea_analysis_pipeline.py
    fig, axs = plt.subplots(2, 1, figsize=(8, 8),sharex=True)

    burst_statistics = helper.detect_bursts_statistics(spike_times, isi_threshold=isi_threshold)
    bursts = [unit_stats['bursts'] for unit_stats in burst_statistics.values()]
    # Extracting ISIs as combined arrays
    all_isis_within_bursts = np.concatenate([stats['isis_within_bursts'] for stats in burst_statistics.values() if stats['isis_within_bursts'].size > 0])
    all_isis_outside_bursts = np.concatenate([stats['isis_outside_bursts'] for stats in burst_statistics.values() if stats['isis_outside_bursts'].size > 0])
    all_isis = np.concatenate([stats['isis_all'] for stats in burst_statistics.values() if stats['isis_all'].size > 0])

    # Calculate combined statistics
    mean_isi_within_combined = np.mean(all_isis_within_bursts) if all_isis_within_bursts.size > 0 else np.nan
    cov_isi_within_combined = np.cov(all_isis_within_bursts) if all_isis_within_bursts.size > 0 else np.nan

    mean_isi_outside_combined = np.mean(all_isis_outside_bursts) if all_isis_outside_bursts.size > 0 else np.nan
    cov_isi_outside_combined = np.cov(all_isis_outside_bursts) if all_isis_outside_bursts.size > 0 else np.nan

    mean_isi_all_combined = np.mean(all_isis) if all_isis.size > 0 else np.nan
    cov_isi_all_combined = np.cov(all_isis) if all_isis.size > 0 else np.nan

    # Calculate spike counts for each unit
    spike_counts = {unit: len(times) for unit, times in spike_times.items()}

    # Sort units by ascending spike counts
    sorted_units = sorted(spike_counts, key=spike_counts.get)

    axs[0]= helper.plot_raster_with_bursts(axs[0],spike_times, bursts,sorted_units=sorted_units, title_suffix="(Sorted Raster Order)")

    # Call the plot_network_activity function and pass the SpikeTimes dictionary
    axs[1],network_data= helper.plot_network_activity(axs[1],spike_times, figSize=(8, 4),binSize=0.1, gaussianSigma=0.2,min_peak_distance=10, thresholdBurst=2)

    network_data['MeanWithinBurstISI'] = mean_isi_within_combined
    network_data['CoVWithinBurstISI'] = cov_isi_within_combined
    network_data['MeanOutsideBurstISI'] = mean_isi_outside_combined   
    network_data['CoVOutsideBurstISI'] = cov_isi_outside_combined
    network_data['MeanNetworkISI'] = mean_isi_all_combined
    network_data['CoVNetworkISI'] = cov_isi_all_combined
    network_data['NumUnits'] = len(spike_times) # Number of units in the network
    #network_data["fileName"]=f"{desired_pattern}/{stream_id}"

    # Extract all metrics
    metrics = extract_all_metrics(spike_times, network_data, isi_threshold)
    return metrics