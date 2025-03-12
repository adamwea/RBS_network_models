from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

# aw 2025-03-05 11:16:01 - copied from fastdtw.py
def __reduce_by_half(x):
    return [(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]

def event_times_dtw_distance(spike_times1, spike_times2):
    # reduce until the length of both spike times are < 5000 points. Speeds up the dtw calculation.
    # original_len1 = len(spike_times1)
    # original_len2 = len(spike_times2)
    reduction_count = 0
    shrinked_st1 = spike_times1
    shrinked_st2 = spike_times2
    #print(f'Original lengths: {len(spike_times1)}')
    while len(shrinked_st1) > 2000 and len(shrinked_st2) > 2000:
        shrinked_st1 = __reduce_by_half(shrinked_st1)
        shrinked_st2 = __reduce_by_half(shrinked_st2)
        reduction_count += 1
        #print(f'Reduced lengths: {len(shrinked_st1)}')
        
    #dtw time-series comparison
    shrinked_st1 = np.array(shrinked_st1)
    shrinked_st2 = np.array(shrinked_st2)
    distance, path = dtw_distance(shrinked_st1, shrinked_st2)
    return distance

def dtw_distance(s1, s2):
    distance, path = fastdtw(s1, s2, dist=euclidean)
    return distance, path