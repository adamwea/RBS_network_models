#debugging new fitness thing

import numpy as np
from scipy.signal import argrelmax

# Generate some sample data
burstPeakTimes = np.linspace(0, 1, 1000)  # 1000 points from 0 to 1 second
burstPeakValues = np.sin(2 * np.pi * 5 * burstPeakTimes)  # 5 Hz sine wave
print("Burst peak times:", burstPeakTimes)
print("Burst peak values:", burstPeakValues)

# Convert min_peak_distance from ms to seconds
min_peak_distance = 10  # ms
min_peak_distance /= 1000

# Calculate the minimum number of samples apart a peak must be from its neighbors
distance = int(min_peak_distance / np.median(np.diff(burstPeakTimes)))

# Find the indices of the burst peak times that are at least 'distance' samples apart
indices = argrelmax(burstPeakValues, order=distance)

# Use the indices to get the burst peak times and values that are more than 10 ms apart
filtered_burstPeakTimes = burstPeakTimes[indices]
filtered_burstPeakValues = burstPeakValues[indices]

# Print the results
print("Filtered burst peak times:", filtered_burstPeakTimes)
print("Filtered burst peak values:", filtered_burstPeakValues)