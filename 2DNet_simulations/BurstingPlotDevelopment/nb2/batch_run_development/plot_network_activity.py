import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks
from scipy.stats import norm

def plot_network_activity(rasterData, min_peak_distance = 1.0, binSize=0.02*1000, gaussianSigma=0.16*1000, thresholdBurst=1.2, figSize=(10, 6), saveFig=False, figName='NetworkActivity.png'):
    SpikeTimes = rasterData['spkTimes']
    relativeSpikeTimes = SpikeTimes
    relativeSpikeTimes = np.array(relativeSpikeTimes)
    relativeSpikeTimes = relativeSpikeTimes - relativeSpikeTimes[0]  # Set the first spike time to 0

    # Step 1: Bin all spike times into small time windows
    timeVector = np.arange(0, max(relativeSpikeTimes) + binSize, binSize)  # Time vector for binning
    binnedTimes, _ = np.histogram(relativeSpikeTimes, bins=timeVector)  # Bin spike times
    binnedTimes = np.append(binnedTimes, 0)  # Append 0 to match MATLAB's binnedTimes length

    # Step 2: Smooth the binned spike times with a Gaussian kernel
    kernelRange = np.arange(-3*gaussianSigma, 3*gaussianSigma + binSize, binSize)  # Range for Gaussian kernel
    kernel = norm.pdf(kernelRange, 0, gaussianSigma)  # Gaussian kernel
    kernel *= binSize  # Normalize kernel by bin size
    firingRate = convolve(binnedTimes, kernel, mode='same') / binSize  # Convolve and normalize by bin size

    # Create a new figure with a specified size (width, height)
    plt.figure(figsize=figSize)
    margin_width = 100
    # Find the indices of timeVector that correspond to the first and last 100 ms
    start_index = np.where(timeVector >= margin_width)[0][0]
    end_index = np.where(timeVector <= max(timeVector) - margin_width)[0][-1]

    # Plot the smoothed network activity
    plt.subplot(1, 1, 1)
    plt.plot(timeVector[start_index:end_index], firingRate[start_index:end_index], color='black')
    plt.xlim([timeVector[start_index], timeVector[end_index]])  # Restrict the plot to the first and last 100 ms
    plt.ylim([min(firingRate[start_index:end_index])*0.8, max(firingRate[start_index:end_index])*1.2])  # Set y-axis limits to min and max of firingRate
    plt.ylabel('Firing Rate [Hz]')
    plt.xlabel('Time [ms]')
    plt.title('Network Activity', fontsize=11)

    # Step 3: Peak detection on the smoothed firing rate curve
    rmsFiringRate = np.sqrt(np.mean(firingRate**2))  # Calculate RMS of the firing rate
    peaks, properties = find_peaks(firingRate, height=thresholdBurst * rmsFiringRate, distance=min_peak_distance)  # Find peaks above the threshold
    burstPeakTimes = timeVector[peaks]  # Convert peak indices to times
    burstPeakValues = properties['peak_heights']  # Get the peak values

    # Plot the threshold line and burst peaks
    plt.plot(np.arange(timeVector[-1]), thresholdBurst * rmsFiringRate * np.ones(np.ceil(timeVector[-1]).astype(int)), color='gray')
    plt.plot(burstPeakTimes, burstPeakValues, 'or')  # Plot burst peaks as red circles    

    if saveFig:
        if isinstance(saveFig, str):
            plt.savefig(saveFig, dpi=300)
        else:
            plt.savefig(figName, dpi=300)

    #plt.show()