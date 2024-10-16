import numpy as np
from scipy.signal import convolve, find_peaks
from scipy.stats import norm

def measure_network_activity(rasterData, binSize, gaussianSigma, thresholdBurst, min_peak_distance):
    relativeSpikeTimes = np.array(rasterData['spkTimes'])
    relativeSpikeTimes -= relativeSpikeTimes[0]
    
    timeVector = np.arange(0, max(relativeSpikeTimes) + binSize, binSize)
    binnedTimes, _ = np.histogram(relativeSpikeTimes, bins=timeVector)
    binnedTimes = np.append(binnedTimes, 0)

    kernelRange = np.arange(-3*gaussianSigma, 3*gaussianSigma + binSize, binSize)
    kernel = norm.pdf(kernelRange, 0, gaussianSigma)
    kernel *= binSize
    firingRate = convolve(binnedTimes, kernel, mode='same') / binSize

    rmsFiringRate = np.sqrt(np.mean(firingRate**2))
    peaks, properties = find_peaks(firingRate, height=thresholdBurst * rmsFiringRate, distance=min_peak_distance)

    burstPeakTimes = timeVector[peaks]
    burstPeakValues = properties['peak_heights']

    return burstPeakTimes, burstPeakValues, firingRate, timeVector, rmsFiringRate