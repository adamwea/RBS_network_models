import numpy as np
from scipy.signal import convolve, find_peaks
from scipy.stats import norm, gaussian_kde

def analyze_network_activity(rasterData, conv_params=None):
    def bimodality_metric(values, bandwidth='scott'):
        kde = gaussian_kde(values, bw_method=bandwidth)
        x = np.linspace(min(values), max(values), 1000)
        density = kde(x)
        peaks, _ = find_peaks(density)

        if len(peaks) < 2:
            return None, None  # Not bimodal

        peak_values = density[peaks]
        highest_peaks = peaks[np.argsort(peak_values)[-2:]]
        min_index = min(highest_peaks)
        max_index = max(highest_peaks)
        local_min = np.argmin(density[min_index:max_index]) + min_index
        bimodality_metric = (density[highest_peaks[0]] + density[highest_peaks[1]]) / (2 * density[local_min])
        
        return bimodality_metric, x[local_min]

    default_params = {'binSize': 0.1, 'gaussianSigma': 0.15, 'thresholdBurst': 1.0, 'min_peak_distance': 1}
    if conv_params is None:
        conv_params = default_params

    relativeSpikeTimes = np.array(rasterData['spkt']) - rasterData['spkt'][0]
    binSize = conv_params['binSize']
    timeVector = np.arange(0, max(relativeSpikeTimes) + binSize, binSize)
    binnedTimes, _ = np.histogram(relativeSpikeTimes, bins=timeVector)
    binnedTimes = np.append(binnedTimes, 0)

    gaussianSigma = conv_params['gaussianSigma']
    kernelRange = np.arange(-3*gaussianSigma, 3*gaussianSigma + binSize, binSize)
    kernel = norm.pdf(kernelRange, 0, gaussianSigma)
    kernel *= binSize
    firingRate = convolve(binnedTimes, kernel, mode='same') / binSize

    min_peak_distance = conv_params['min_peak_distance']
    thresholdBurst = conv_params['thresholdBurst']
    rmsFiringRate = np.sqrt(np.mean(firingRate**2))
    min_distance_samples = min_peak_distance / binSize
    peaks, properties = find_peaks(firingRate, height=thresholdBurst * rmsFiringRate, distance=min_distance_samples)
    
    burstPeakTimes = timeVector[peaks]
    burstPeakValues = properties['peak_heights']

    try:
        bimodality_val, bimodality_threshold = bimodality_metric(burstPeakValues)
    except:
        bimodality_val, bimodality_threshold = None, None

    big_burst_vals = burstPeakValues[burstPeakValues > bimodality_threshold] if bimodality_threshold else None
    small_burst_vals = burstPeakValues[burstPeakValues < bimodality_threshold] if bimodality_threshold else None

    IBIs = np.diff(burstPeakTimes) if len(burstPeakTimes) > 1 else None

    measurements = {
        'burst_vals': burstPeakValues,
        'burst_times': burstPeakTimes,
        'IBIs': IBIs,
        'firing_rate': firingRate,
        'baseline': np.mean(firingRate) if firingRate.size else None,
        'burst_freq': len(burstPeakTimes) / (timeVector[-1]) if len(burstPeakTimes) > 1 else None,
        'bimodal_val': bimodality_val,
        'bimodal_threshold': bimodality_threshold,
        'big_burst_vals': big_burst_vals,
        'small_burst_vals': small_burst_vals
    }
    return measurements