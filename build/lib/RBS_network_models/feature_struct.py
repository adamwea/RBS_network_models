from RBS_network_models.network_analysis import get_min, get_max
import numpy as np
from copy import deepcopy
#import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.utils import to_time_series_dataset
#from tslearn.clustering import TimeSeriesKMedoids
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from dtwParallel import dtw_functions
from scipy.spatial import distance
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import multiprocessing
import time

# =============================================================================

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import multiprocessing
from functools import partial
from scipy.signal import resample


''' newer funcs 
^
|
|
'''

def compute_dtw_batch_3(pair_batch, sequence_stacks, lock, counter, total, effective_rate, start_time):
    """Computes DTW distances for a batch of pairs (vectorized approach)."""
    results = []
    
    # loop through pairs
    for i, j in pair_batch:
        # seq_i = np.array(time_sequence_dict[i])
        # seq_j = np.array(time_sequence_dict[j])
        
        # # Downsample for speed (adjust this if accuracy is a concern)
        # aw 2025-02-22 22:39:34
        # apparently fastdtw already downsamples by half internally. That's why it's so fast lol.
        # accuracy is... vaugely concerning, but not really since we're just trying to decipher a 
        # general trend to tell simulations to target.
        # Just going to try downsampling by factor of 4 to see if it speeds things up.
        
        #distance_test, _ = fastdtw(seq_i, seq_j, dist=euclidean)  # Compute DTW

        
        # seq_i = downsample_sequence(seq_i, downsample_factor=4)
        # seq_j = downsample_sequence(seq_j, downsample_factor=4)
        
        seq_i = sequence_stacks[i]
        seq_j = sequence_stacks[j]

        distance, _ = fastdtw(seq_i, seq_j, dist=euclidean)  # Compute DTW
        
        results.append((i, j, distance))

        with lock:  # Safely update progress counter
            counter.value += 1
            effective_rate.value = counter.value / (time.time() - start_time)
            # if counter.value % 25 == 0 or counter.value == total:
            #     #print(f"Progress: {counter.value}/{total} distances computed")
            #     print(f"EFFECTIVE RATE: {effective_rate.value} distances computed")
            #     estimated_time_remaining = (total - counter.value) / effective_rate.value
            #     # convert to minutes
            #     estimated_time_remaining = estimated_time_remaining / 60
            #     print(f"Estimated time remaining: {estimated_time_remaining} minutes")
            if counter.value % 100 == 0 or counter.value == total:  
                #print(f"Progress: {counter.value}/{total} distances computed")
                #put all messages together on one line
                estimated_time_remaining = (total - counter.value) / effective_rate.value
                # convert to HH:MM format
                estimated_time_remaining = time.strftime('%H:%M', time.gmtime(estimated_time_remaining))
                print(f"Progress: {counter.value}/{total} distances computed | "
                      f"Effective Rate: {effective_rate.value} computations/s | "
                      f"Estimated time remaining: {estimated_time_remaining}")
    
    return results

def dtw_analysis_v3(time_sequence_dict, cat_sequence_dict, sequence_stacks, bursting_data):
    
    ## =================== ##
    ## save test data for easier debugging
    # path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/tests/dtw_test_data/'
    # time_sequence_dict = {i: time_sequence_mat[i] for i in range(len(time_sequence_mat))}
    # cat_sequence_dict = {i: cat_sequence_mat[i] for i in range(len(cat_sequence_mat))}
    # np.save(path + 'time_sequence_dict.npy', time_sequence_dict)
    # np.save(path + 'cat_sequence_dict.npy', cat_sequence_dict)
    # np.save(path + 'sequence_stacks.npy', sequence_stacks)
    # np.save(path + 'bursting_data.npy', bursting_data)
    
    ## =================== ##
    ## DTW Analysis    
    
    # print('Length of cat_sequence_mat:', len(cat_sequence_mat))
    # print('Length of time_sequence_mat:', len(time_sequence_mat))
    print('Length of cat_sequence_dict:', len(cat_sequence_dict))
    print('Length of time_sequence_dict:', len(time_sequence_dict))

    #num_sequences = len(time_sequence_mat)
    num_sequences = len(time_sequence_dict)
    distance_matrix = np.zeros((num_sequences, num_sequences))  # Initialize matrix

    # Create a list of unique index pairs (i, j)
    pairs = [(i, j) for i in range(num_sequences) for j in range(i + 1, num_sequences)]
    total_distances = len(pairs)
    print('total_distances to compute:', total_distances)
    
    # Setup multiprocessing resources
    counter = multiprocessing.Value('i', 0)  # Shared counter for tracking progress
    lock = multiprocessing.Lock()  # Lock for updating counter safely
    start_time = time.time()
    effective_rate = multiprocessing.Value('f', 0.0)
    
    # # one dtw at a time for now
    # for i in range(num_sequences):
    #     for j in range(i + 1, num_sequences):
    #         # seq_i = np.array(time_sequence_dict[i])
    #         # seq_j = np.array(time_sequence_dict[j])
            
    #         # correct way to do it - sensitive to spike times and i/e categorization
    #         seq_i = sequence_stacks[i]
    #         seq_j = sequence_stacks[j]
    #         distance, _ = fastdtw(seq_i, seq_j, dist=euclidean)
    #         print(f'Shape of seq_i: {seq_i.shape}')
    #         print(f'Shape of seq_j: {seq_j.shape}')
    #         print(f'DTW distance between {i} and {j}: {distance}')
            
    #         # test - time data alone
    #         # # seq_i_test = time_sequence_dict[i]
    #         # # seq_j_test = time_sequence_dict[j]
    #         # seq_i_test = np.array(time_sequence_dict[i])
    #         # seq_j_test = np.array(time_sequence_dict[j])
    #         # distance_test, _ = fastdtw(seq_i_test, seq_j_test, dist=euclidean)
    #         # print(f'Shape of seq_i_test: {seq_i_test.shape}')
    #         # print(f'Shape of seq_j_test: {seq_j_test.shape}')
    #         # print(f'DTW distance between {i} and {j}: {distance_test}')
            
    #         distance_matrix[i, j] = distance
    #         distance_matrix[j, i] = distance  # Symmetric property
    #         with lock:  # Safely update progress counter
    #             counter.value += 1
    #             if counter.value % 100 == 0 or counter.value == total_distances:
    #                 print(f"Progress: {counter.value}/{total_distances} distances computed")
                    
    # parallelized dtw and batched
    # Batch the pairs to reduce overhead
    #batch_size = 500  # Adjust this for better performance (100-500 is a good range)
    #batch_size = 1000
    
    #num_workers = 50
    # use all possible cpus. 1 worker per cpu
    num_workers = multiprocessing.cpu_count()
    print(f'Number of cpus: {multiprocessing.cpu_count()}')
    print(f'Number of workers: {num_workers}')
    
    # split pairs into 1 batch per worker
    # split batches squarely in to num_workers-1 batches
    # then use the last worker-batch to fill in the rest
    batch_size = len(pairs)//(num_workers)
    #pair_batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    
    # randomize pairs for better approximation of time
    np.random.shuffle(pairs)
    pair_batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]    

    #num_workers = min(multiprocessing.cpu_count(), len(pair_batches))  # Limit workers to CPU count
    #num_workers = 50
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use partial function to pass additional arguments
        results_batches = [compute_dtw_batch_3(pair_batch, sequence_stacks, 
                                               lock, counter, total_distances, 
                                               effective_rate, start_time) for pair_batch in pair_batches]
        #results_batches = [func, pair_batch for pair_batch in pair_batches]
        
    # try multi threading instead
    
    
    
        
    # Flatten results and store them in the distance matrix
    for results in results_batches:
        for i, j, distance in results:
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric property

    print("DTW distance matrix computation complete!")
    return distance_matrix

def downsample_sequence(seq, target_length=None, downsample_factor=None):
    # assert that either target_length or downsample_factor is provided
    assert target_length is not None or downsample_factor is not None, "Either target_length or downsample_factor must be provided."
    
    """Resamples a sequence to a fixed length for faster DTW computation."""
    if target_length is not None and len(seq) > target_length:
        if len(seq) > target_length:
            return resample(seq, target_length)
        return seq
    elif downsample_factor is not None:
        start_length = len(seq)
        target_length = int(start_length / downsample_factor)
        return resample(seq, target_length)
    else:
        return seq

def compute_dtw_batch_v2(pairs_batch, time_sequence_mat, lock, counter, total):
    """Computes DTW distances for a batch of pairs (vectorized approach)."""
    results = []
    
    #options
    loop = True
    stack = False # aw 2025-02-22 22:23:36 nvm this doesn't work
    
    # loop through pairs
    if loop:
        for i, j in pairs_batch:
            seq_i = np.array(time_sequence_mat[i])
            seq_j = np.array(time_sequence_mat[j])
            
            # # Downsample for speed (adjust this if accuracy is a concern)
            # aw 2025-02-22 22:39:34
            # apparently fastdtw already downsamples by half internally. That's why it's so fast lol.
            # accuracy is... vaugely concerning, but not really since we're just trying to decipher a 
            # general trend to tell simulations to target.
            # Just going to try downsampling by factor of 4 to see if it speeds things up.
            
            #distance_test, _ = fastdtw(seq_i, seq_j, dist=euclidean)  # Compute DTW

            
            # seq_i = downsample_sequence(seq_i, downsample_factor=4)
            # seq_j = downsample_sequence(seq_j, downsample_factor=4)

            distance, _ = fastdtw(seq_i, seq_j, dist=euclidean)  # Compute DTW
            
            results.append((i, j, distance))

            with lock:  # Safely update progress counter
                counter.value += 1
                if counter.value % 100 == 0 or counter.value == total:  
                    print(f"Progress: {counter.value}/{total} distances computed")
    # stack pairs
    # elif stack:
    #     stack_i = []
    #     stack_j = []
    #     for i, j in pairs_batch:
    #         stack_i.append(time_sequence_mat[i])
    #         stack_j.append(time_sequence_mat[j])
        
    #     # pad stacks with np.nan as needed
    #     max_length_i = max([len(x) for x in stack_i])
    #     max_length_j = max([len(x) for x in stack_j])
    #     max_length = max(max_length_i, max_length_j)
    #     for i in range(len(stack_i)):
    #         if len(stack_i[i]) < max_length:
    #             stack_i[i] = np.pad(stack_i[i], (0, max_length - len(stack_i[i])), 'constant', constant_values=(0))
    #     for i in range(len(stack_j)):
    #         if len(stack_j[i]) < max_length:
    #             stack_j[i] = np.pad(stack_j[i], (0, max_length - len(stack_j[i])), 'constant', constant_values=(0))
        
        
    #     stack_i = np.array(stack_i)
    #     stack_j = np.array(stack_j)
    #     print('stack_i:', stack_i.shape)
    #     print('stack_j:', stack_j.shape)
    #     print('stack_i:', stack_i)
    #     print('stack_j:', stack_j)
        
    #     # Downsample for speed (adjust this if accuracy is a concern)
    #     #seq_i = downsample_sequence(seq_i)
    #     #seq_j = downsample_sequence(seq_j)
        
    #     distance, _ = fastdtw(stack_i, stack_j, dist=euclidean)  # Compute DTW
        
    #     results.append((i, j, distance))
        
        
        
        
    
    return results

def dtw_analysis_v2(bursting_data, sequence_stacks, time_sequence_mat, cat_sequence_mat):
    
    ## =================== ##
    ## save test data for easier debugging
    # path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/tests/dtw_test_data/'
    # time_sequence_dict = {i: time_sequence_mat[i] for i in range(len(time_sequence_mat))}
    # cat_sequence_dict = {i: cat_sequence_mat[i] for i in range(len(cat_sequence_mat))}
    # np.save(path + 'time_sequence_dict.npy', time_sequence_dict)
    # np.save(path + 'cat_sequence_dict.npy', cat_sequence_dict)
    # np.save(path + 'sequence_stacks.npy', sequence_stacks)
    # np.save(path + 'bursting_data.npy', bursting_data)
    
    ## =================== ##
    ## DTW Analysis    
    
    print('Length of cat_sequence_mat:', len(cat_sequence_mat))
    print('Length of time_sequence_mat:', len(time_sequence_mat))

    num_sequences = len(time_sequence_mat)
    distance_matrix = np.zeros((num_sequences, num_sequences))  # Initialize matrix

    # Create a list of unique index pairs (i, j)
    pairs = [(i, j) for i in range(num_sequences) for j in range(i + 1, num_sequences)]
    total_distances = len(pairs)
    print('total_distances to compute:', total_distances)
    
    # Setup multiprocessing resources
    counter = multiprocessing.Value('i', 0)  # Shared counter for tracking progress
    lock = multiprocessing.Lock()  # Lock for updating counter safely

    # Batch the pairs to reduce overhead
    batch_size = 500  # Adjust this for better performance (100-500 is a good range)
    pair_batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]

    #num_workers = min(multiprocessing.cpu_count(), len(pair_batches))  # Limit workers to CPU count
    num_workers = 25
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use partial function to pass additional arguments
        func = partial(compute_dtw_batch_v2, time_sequence_mat=time_sequence_mat, lock=lock, counter=counter, total=total_distances)
        #results_batches = pool.map(func, pair_batches)
        results_batches = [compute_dtw_batch_v2(pair_batch, time_sequence_mat, lock, counter, total_distances) for pair_batch in pair_batches]
        #results_batches = [func, pair_batch for pair_batch in pair_batches]
        
    # Flatten results and store them in the distance matrix
    for results in results_batches:
        for i, j, distance in results:
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric property

    print("DTW distance matrix computation complete!")
    return distance_matrix

def compute_dtw_distance(pair, lock, counter, total, **args):
    """Computes DTW distance for a given pair of indices."""
    i, j, time_sequence_mat = pair
    seq_i = np.array(time_sequence_mat[i])
    seq_j = np.array(time_sequence_mat[j])
    distance, _ = fastdtw(seq_i, seq_j, dist=euclidean)  # Compute DTW distance
    
    with lock:  # Safely update progress counter
        counter.value += 1
        #if counter.value % 100 == 0 or counter.value == total:  # Show progress every 100 calculations
        print(f"Progress: {counter.value}/{total} distances computed")

    
    #print(f"DTW distance between {i} and {j}: {distance}")
    return (i, j, distance)

def dtw_analysis(bursting_data, sequence_stacks, time_sequence_mat, cat_sequence_mat):
    print('Length of cat_sequence_mat:', len(cat_sequence_mat))
    print('Length of time_sequence_mat:', len(time_sequence_mat))

    num_sequences = len(time_sequence_mat)
    distance_matrix = np.zeros((num_sequences, num_sequences))  # Initialize matrix

    # Create a list of unique index pairs (i, j)
    pairs = [(i, j, time_sequence_mat) for i in range(num_sequences) for j in range(i + 1, num_sequences)]
    
    # counter
    # Setup multiprocessing resources
    counter = multiprocessing.Value('i', 0)  # Shared counter for tracking progress
    lock = multiprocessing.Lock()  # Lock for updating counter safely

    # Use multiprocessing to parallelize DTW computations
    #num_workers = min(multiprocessing.cpu_count(), len(pairs))  # Limit workers to CPU count
    num_workers = 50
    total = len(pairs)
    with multiprocessing.Pool(processes=num_workers) as pool:
        #results = pool.map(compute_dtw_distance, pairs)
        results = [compute_dtw_distance(pair, lock, counter, total) for pair in pairs]

    # Store results in distance matrix
    for i, j, distance in results:
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance  # Symmetric property

    return distance_matrix

def build_network_metric_targets_dict_v2(network_metrics):
    network_metric_targets = deepcopy(network_metrics)
    
    #remove simulation data if it exists
    if 'simulated_data' in network_metric_targets:
        network_metric_targets.pop('simulated_data')
    
    # remove source if it exists
    if 'source' in network_metric_targets:
        network_metric_targets.pop('source')
    
    # remove timeVector if it exists
    if 'timeVector' in network_metric_targets:
        network_metric_targets.pop('timeVector')
    
    # remove recording, sorting and waveform output paths if they exist
    if 'recording_path' in network_metric_targets:
        network_metric_targets.pop('recording_path')
    if 'sorting_path' in network_metric_targets:
        network_metric_targets.pop('sorting_path')
    if 'waveform_path' in network_metric_targets:
        network_metric_targets.pop('waveform_path')
        
    ## ** Spiking Data ** ##
    # remove spiking_data['spike_times'] and spiking_data['spiking_times_by_unit'] if they exist
    if 'spike_times' in network_metric_targets['spiking_data']:
        network_metric_targets['spiking_data'].pop('spike_times')
    if 'spiking_times_by_unit' in network_metric_targets['spiking_data']:
        network_metric_targets['spiking_data'].pop('spiking_times_by_unit')
    if 'spiking_data_by_unit' in network_metric_targets['spiking_data']:
        network_metric_targets['spiking_data'].pop('spiking_data_by_unit')
    if 'spiking_metrics_by_unit' in network_metric_targets['spiking_data']:
        for unit in network_metric_targets['spiking_data']['spiking_metrics_by_unit']:
            # remove num_spikes
            network_metric_targets['spiking_data']['spiking_metrics_by_unit'][unit].pop('num_spikes')
            
            # remove wf_metrics
            network_metric_targets['spiking_data']['spiking_metrics_by_unit'][unit].pop('wf_metrics') # TOOD: Bring this back when we start doing waveform analysis in simulations
            
            # remove individual spike_times 
            network_metric_targets['spiking_data']['spiking_metrics_by_unit'][unit].pop('spike_times')
            
            # replace 'data' in isi with max and min
            data = network_metric_targets['spiking_data']['spiking_metrics_by_unit'][unit]['isi']['data']
            try: network_metric_targets['spiking_data']['spiking_metrics_by_unit'][unit]['isi']['min'] = np.min(data)
            except: network_metric_targets['spiking_data']['spiking_metrics_by_unit'][unit]['isi']['min'] = None
            try: network_metric_targets['spiking_data']['spiking_metrics_by_unit'][unit]['isi']['max'] = np.max(data)
            except: network_metric_targets['spiking_data']['spiking_metrics_by_unit'][unit]['isi']['max'] = None
            network_metric_targets['spiking_data']['spiking_metrics_by_unit'][unit]['isi'].pop('data')
            
            # replace fr, which should be a float/int, with 'target' as the same value
            fr = network_metric_targets['spiking_data']['spiking_metrics_by_unit'][unit]['fr']
            network_metric_targets['spiking_data']['spiking_metrics_by_unit'][unit]['fr'] = {'target': fr}
            
    ## ** Bursting Data ** ##
    ### Unit Metrics ###
    bursting_data = network_metric_targets['bursting_data']
    if 'ax' in bursting_data:
        bursting_data.pop('ax')
        
    if 'convolved_data' in bursting_data:
        convolved_data = bursting_data['convolved_data'].copy() # pull this out for later use
        bursting_data.pop('convolved_data')
    else:
        convolved_data = None
    
    #
    unit_metrics = bursting_data['unit_metrics']
    for unit in unit_metrics:
        
        # remove burst_id if it exists
        if 'burst_id' in unit_metrics[unit]:
            unit_metrics[unit].pop('burst_id')
        
        # remove quiet_id if it exists
        if 'quiet_id' in unit_metrics[unit]:
            unit_metrics[unit].pop('quiet_id')
            
        # remove burst_durations
        if 'burst_durations' in unit_metrics[unit]:
            unit_metrics[unit].pop('burst_durations')
            
        # remove quiet_durations if it exists
        if 'quiet_durations' in unit_metrics[unit]:
            unit_metrics[unit].pop('quiet_durations')
        
        # for burst_part_rate, quiet_part_rate, and burst_part_perc - there should be floats/integers in each of these fields
        # replace them with 'target' as the same value
        for key in ['burst_part_rate', 'quiet_part_rate', 'burst_part_perc']:
            if key in unit_metrics[unit]:
                unit_metrics[unit][key] = {'target': unit_metrics[unit][key]}
                
        fr = unit_metrics[unit]['fr']
        # in_burst, if 'data' exists, replace it with 'min' and 'max'
        data = fr['in_burst']['data']
        try: unit_metrics[unit]['fr']['in_burst']['min'] = np.min(data)
        except: unit_metrics[unit]['fr']['in_burst']['min'] = None
        try: unit_metrics[unit]['fr']['in_burst']['max'] = np.max(data)
        except: unit_metrics[unit]['fr']['in_burst']['max'] = None
        unit_metrics[unit]['fr']['in_burst'].pop('data')
        
        # out_burst, if 'data' exists, replace it with 'min' and 'max'
        data = fr['out_burst']['data']
        try: unit_metrics[unit]['fr']['out_burst']['min'] = np.min(data)
        except: unit_metrics[unit]['fr']['out_burst']['min'] = None
        try: unit_metrics[unit]['fr']['out_burst']['max'] = np.max(data)
        except: unit_metrics[unit]['fr']['out_burst']['max'] = None
        unit_metrics[unit]['fr']['out_burst'].pop('data')
        
        isi = unit_metrics[unit]['isi']
        # in_burst, if 'data' exists, replace it with 'min' and 'max'
        data = isi['in_burst']['data']
        try: unit_metrics[unit]['isi']['in_burst']['min'] = np.min(data)
        except: unit_metrics[unit]['isi']['in_burst']['min'] = None
        try: unit_metrics[unit]['isi']['in_burst']['max'] = np.max(data)
        except: unit_metrics[unit]['isi']['in_burst']['max'] = None
        unit_metrics[unit]['isi']['in_burst'].pop('data')
        
        # out_burst, if 'data' exists, replace it with 'min' and 'max'
        data = isi['out_burst']['data']
        try: unit_metrics[unit]['isi']['out_burst']['min'] = np.min(data)
        except: unit_metrics[unit]['isi']['out_burst']['min'] = None
        try: unit_metrics[unit]['isi']['out_burst']['max'] = np.max(data)
        except: unit_metrics[unit]['isi']['out_burst']['max'] = None
        unit_metrics[unit]['isi']['out_burst'].pop('data')
        
        spike_counts = unit_metrics[unit]['spike_counts']
        # in_burst, if 'data' exists, replace it with 'min' and 'max'
        data = spike_counts['in_burst']['data']
        try: unit_metrics[unit]['spike_counts']['in_burst']['min'] = np.min(data)
        except: unit_metrics[unit]['spike_counts']['in_burst']['min'] = None
        try: unit_metrics[unit]['spike_counts']['in_burst']['max'] = np.max(data)
        except: unit_metrics[unit]['spike_counts']['in_burst']['max'] = None
        unit_metrics[unit]['spike_counts']['in_burst'].pop('data')
        
        # out_burst, if 'data' exists, replace it with 'min' and 'max'
        data = spike_counts['out_burst']['data']
        try: unit_metrics[unit]['spike_counts']['out_burst']['min'] = np.min(data)
        except: unit_metrics[unit]['spike_counts']['out_burst']['min'] = None
        try: unit_metrics[unit]['spike_counts']['out_burst']['max'] = np.max(data)
        except: unit_metrics[unit]['spike_counts']['out_burst']['max'] = None
        unit_metrics[unit]['spike_counts']['out_burst'].pop('data')
        
        fano_factor = unit_metrics[unit]['fano_factor']
        # in_burst, this should be a float/int, replace it with 'target' as the same value
        in_burst = fano_factor['in_burst']
        unit_metrics[unit]['fano_factor']['in_burst'] = {'target': in_burst}
        
        # out_burst, this should be a float/int, replace it with 'target' as the same value
        out_burst = fano_factor['out_burst']
        unit_metrics[unit]['fano_factor']['out_burst'] = {'target': out_burst}
    
    ### Burst Metrics ###
    burst_metrics = bursting_data['burst_metrics']
    
    #remove num_bursts if it exists
    if 'num_bursts' in burst_metrics:
        burst_metrics.pop('num_bursts')
    
    # burst_rate, should be a float/int, replace it with 'target' as the same value
    burst_rate = burst_metrics['burst_rate']
    bursting_data['burst_metrics']['burst_rate'] = {'target': burst_rate}
    
    # remove burst_ids if they exist
    if 'burst_ids' in burst_metrics:
        burst_metrics.pop('burst_ids')
        
    # ibi, if 'data' exists, replace it with 'min' and 'max'
    data = burst_metrics['ibi']['data']
    try: bursting_data['burst_metrics']['ibi']['min'] = np.min(data)
    except: bursting_data['burst_metrics']['ibi']['min'] = None
    try: bursting_data['burst_metrics']['ibi']['max'] = np.max(data)
    except: bursting_data['burst_metrics']['ibi']['max'] = None
    bursting_data['burst_metrics']['ibi'].pop('data')
    
    # burst_amp, if 'data' exists, replace it with 'min' and 'max'
    data = burst_metrics['burst_amp']['data']
    try: bursting_data['burst_metrics']['burst_amp']['min'] = np.min(data)
    except: bursting_data['burst_metrics']['burst_amp']['min'] = None
    try: bursting_data['burst_metrics']['burst_amp']['max'] = np.max(data)
    except: bursting_data['burst_metrics']['burst_amp']['max'] = None
    bursting_data['burst_metrics']['burst_amp'].pop('data')
    
    # burst_durations, if 'data' exists, replace it with 'min' and 'max'
    burst_durations = burst_metrics['burst_duration'].copy() # pull this out for later use
    data = burst_metrics['burst_duration']['data']
    try: bursting_data['burst_metrics']['burst_duration']['min'] = np.min(data)
    except: bursting_data['burst_metrics']['burst_duration']['min'] = None
    try: bursting_data['burst_metrics']['burst_duration']['max'] = np.max(data)
    except: bursting_data['burst_metrics']['burst_duration']['max'] = None
    bursting_data['burst_metrics']['burst_duration'].pop('data')
    
    #### Burst Participation Metrics ####
    burst_parts = burst_metrics['burst_parts']
    sequence_stacks = {}
    time_sequence_mat = []
    cat_sequence_mat = []
    for burst_id in burst_parts:
        
        #
        burst = burst_parts[burst_id]
        try: unit_sequence = burst['unit_sequence']
        except: unit_sequence = burst['unit_seqeunce'] #TODO #stupid typo in analysis code - fix and rerun
        time_sequence = burst['time_sequence']
        relative_time_sequence = burst['relative_time_sequence']
        
        #classification output
        #cross reference participating units with classification data to get excitatory and inhibitory units
        #participating_units = burst_parts[burst_id]['participating_units']
        classified_units = network_metric_targets['classification_output']['classified_units']
        I_units = [unit for unit in classified_units if classified_units[unit]['desc'] == 'inhib']
        E_units = [unit for unit in classified_units if classified_units[unit]['desc'] == 'excit']
        classified_sequence = []
        for unit in unit_sequence:
            if unit in I_units:
                classified_sequence.append('I')
            elif unit in E_units:
                classified_sequence.append('E')
            else:
                classified_sequence.append('U')
        burst_parts[burst_id]['classified_sequence'] = classified_sequence
        #print('classified_sequence:', classified_sequence)
        
        #stack sequences for each unit
        cat_sequence = [0 if x == 'I' else 1 if x == 'E' else 2 for x in classified_sequence]
        sequence_stack = np.column_stack((
            #unit_sequence, 
            #time_sequence, 
            cat_sequence,
            relative_time_sequence,)).astype(float)
            #cat_sequence)).astype(float)
            
        cat_sequence_mat.append(cat_sequence)
        time_sequence_mat.append(relative_time_sequence)            
        
        #store for DTW analysis
        sequence_stacks[burst_id] = np.array(sequence_stack)
        
        #store for DTW analysis
        burst_parts[burst_id]['sequence_stack'] = sequence_stack
        
        # remove burst_start_time if it exists
        if 'burst_start_time' in burst:
            burst.pop('burst_start_time')
            
        # remove burst_end_time if it exists
        if 'burst_end_time' in burst:
            burst.pop('burst_end_time')

        # remove spike_counts if it exists
        if 'spike_counts' in burst:
            burst.pop('spike_counts')
            
        # remove participating units if it exists
        if 'participating_units' in burst:
            burst.pop('participating_units')
        
        # remove spike_counts_by_unit if it exists
        if 'spike_counts_by_unit' in burst:
            burst.pop('spike_counts_by_unit')
            
        # remove spike_times_by_unit if it exists
        if 'spike_times_by_unit' in burst:
            burst.pop('spike_times_by_unit')
            
        isi = burst['isi']
        # if data exists, replace it with 'min' and 'max'
        data = isi['data']
        try: burst['isi']['min'] = np.min(data)
        except: burst['isi']['min'] = None
        try: burst['isi']['max'] = np.max(data)
        except: burst['isi']['max'] = None
        burst['isi'].pop('data')
        
    #update burst_metrics
    bursting_data['burst_metrics']['burst_parts'] = burst_parts
    
    #num_units_per_burst, if 'data' exists, replace it with 'min' and 'max'
    data = burst_metrics['num_units_per_burst']['data']
    try: bursting_data['burst_metrics']['num_units_per_burst']['min'] = np.min(data)
    except: bursting_data['burst_metrics']['num_units_per_burst']['min'] = None
    try: bursting_data['burst_metrics']['num_units_per_burst']['max'] = np.max(data)
    except: bursting_data['burst_metrics']['num_units_per_burst']['max'] = None
    bursting_data['burst_metrics']['num_units_per_burst'].pop('data')
    
    #in_burst_fr, if 'data' exists, replace it with 'min' and 'max'
    data = burst_metrics['in_burst_fr']['data']
    try: bursting_data['burst_metrics']['in_burst_fr']['min'] = np.min(data)
    except: bursting_data['burst_metrics']['in_burst_fr']['min'] = None
    try: bursting_data['burst_metrics']['in_burst_fr']['max'] = np.max(data)
    except: bursting_data['burst_metrics']['in_burst_fr']['max'] = None
    bursting_data['burst_metrics']['in_burst_fr'].pop('data')
    
    ### do DTW analysis ###
    
    #
    #prepare_sequences(time_sequence_mat, cat_sequence_mat)
    
    #dtw_analysis(bursting_data, sequence_stacks, time_sequence_mat, cat_sequence_mat)
    dtw_analysis_v3(bursting_data, sequence_stacks, time_sequence_mat, cat_sequence_mat)
    
    print('targets built')    
    
    
    
    
        
    
        
    
            
            

    
    
    network_metrics_targets = {
        #Spiking Data
        #Bursting Data
        #Mega Bursting Data
    }
    return network_metric_targets
'''
|
|
v


older funcs '''
# =============================================================================
# =============================================================================


def get_target_E_FR(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_I_FR(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_E_FR(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_I_FR(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_E_FR(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_I_FR(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_E_CoV_FR(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fr_CoV'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_I_CoV_FR(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fr_CoV'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_E_CoV_FR(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fr_CoV'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_I_CoV_FR(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fr_CoV'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_E_CoV_FR(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fr_CoV'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_I_CoV_FR(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fr_CoV'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_E_ISI(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_I_ISI(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_E_ISI(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_I_ISI(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_E_ISI(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_I_ISI(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_E_CoV_ISI(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['isi_CoV'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_I_CoV_ISI(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['isi_CoV'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_E_CoV_ISI(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['isi_CoV'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_I_CoV_ISI(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['isi_CoV'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_E_CoV_ISI(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['isi_CoV'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_I_CoV_ISI(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['isi_CoV'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_fano_factor_E_target(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fano_factor'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_fano_factor_I_target(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fano_factor'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_fano_factor_E_min(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fano_factor'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_fano_factor_I_min(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fano_factor'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_fano_factor_E_max(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fano_factor'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_fano_factor_I_max(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fano_factor'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_withinBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_within'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_withinBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_within'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_withinBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_within'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_withinBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_within'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_withinBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_within'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_withinBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_within'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_CoVWithinBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_within'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_CoVWithinBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_within'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_CoVWithinBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_within'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_CoVWithinBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_within'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_CoVWithinBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_within'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_CoVWithinBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_within'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_outsideBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_outside'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_outsideBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_outside'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_outsideBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_outside'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_outsideBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_outside'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_outsideBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_outside'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_outsideBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_outside'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_CoVOutsideBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_outside'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_CoVOutsideBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_outside'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_CoVOutsideBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_outside'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_CoVOutsideBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_outside'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_CoVOutsideBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_outside'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_CoVOutsideBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_outside'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def build_network_metric_targets_dict(network_metrics):
    
    #initialize network_metric_targets
    bursting_data = network_metrics['bursting_data']
    mega_bursting_data = network_metrics['mega_bursting_data']
    duration_seconds = network_metrics['timeVector'][-1]
    
    network_metric_targets = {
        #General Data
        #'source': f'{network_metrics['source']}', # 'simulated' or 'experimental'
        'source': 'experimental',
        'duration_seconds': duration_seconds,
        'number_of_units': network_metrics['classification_data']['no. of units'],
        'inhib_units': network_metrics['classification_data']['inhibitory_neurons'],
        'excit_units': network_metrics['classification_data']['excitatory_neurons'],
        'number_of_inhib_units': len(network_metrics['classification_data']['inhibitory_neurons']),
        'number_of_excit_units': len(network_metrics['classification_data']['excitatory_neurons']),
        'unit_locations': network_metrics['classification_data']['unit_locations'],
        #'timeVector': network_metrics['timeVector'],
        
        # Spiking Data
        'spiking_data': {
            #'spike_times': network_metrics['spiking_data']['spike_times'],
            #'spiking_times_by_unit': network_metrics['spiking_data']['spiking_times_by_unit'],
            #'spiking_data_by_unit': network_metrics['spiking_data']['spiking_data_by_unit'],
            'spiking_summary_data': {
                'MeanFireRate': {
                    'target': network_metrics['spiking_data']['spiking_summary_data']['MeanFireRate'],
                    'min': get_min(network_metrics['spiking_data']['spiking_data_by_unit'], 'FireRate'),
                    'max': get_max(network_metrics['spiking_data']['spiking_data_by_unit'], 'FireRate'),      

                    'target_E': get_target_E_FR(network_metrics),
                    'min_E': get_min_E_FR(network_metrics),
                    'max_E': get_max_E_FR(network_metrics),
                    
                    'target_I': get_target_I_FR(network_metrics),
                    'min_I': get_min_I_FR(network_metrics),
                    'max_I': get_max_I_FR(network_metrics),                 
                                        
                    'weight': 1, # TODO: update these with Nfactors
                },
                'CoVFireRate': {
                    'target': network_metrics['spiking_data']['spiking_summary_data']['CoVFireRate'],
                    'min': get_min(network_metrics['spiking_data']['spiking_data_by_unit'], 'fr_CoV'),
                    'max': get_max(network_metrics['spiking_data']['spiking_data_by_unit'], 'fr_CoV'),
                    
                    'target_E': get_target_E_CoV_FR(network_metrics),
                    'min_E': get_min_E_CoV_FR(network_metrics),
                    'max_E': get_max_E_CoV_FR(network_metrics),
                    
                    'target_I': get_target_I_CoV_FR(network_metrics),
                    'min_I': get_min_I_CoV_FR(network_metrics),
                    'max_I': get_max_I_CoV_FR(network_metrics),
                    
                    'weight': 1, # TODO: update these with Nfactors
                },
                'MeanISI': {
                    'target': network_metrics['spiking_data']['spiking_summary_data']['MeanISI'],
                    'min': get_min(network_metrics['spiking_data']['spiking_data_by_unit'], 'meanISI'),
                    'max': get_max(network_metrics['spiking_data']['spiking_data_by_unit'], 'meanISI'),
                    
                    'target_E': get_target_E_ISI(network_metrics),
                    'min_E': get_min_E_ISI(network_metrics),
                    'max_E': get_max_E_ISI(network_metrics),
                    
                    'target_I': get_target_I_ISI(network_metrics),
                    'min_I': get_min_I_ISI(network_metrics),
                    'max_I': get_max_I_ISI(network_metrics),
                    
                    'weight': 1, # TODO: update these with Nfactors
                },
                'CoV_ISI': {
                    'target': network_metrics['spiking_data']['spiking_summary_data']['CoV_ISI'],
                    'min': get_min(network_metrics['spiking_data']['spiking_data_by_unit'], 'isi_CoV'),
                    'max': get_max(network_metrics['spiking_data']['spiking_data_by_unit'], 'isi_CoV'),
                    
                    'target_E': get_target_E_CoV_ISI(network_metrics),
                    'min_E': get_min_E_CoV_ISI(network_metrics),
                    'max_E': get_max_E_CoV_ISI(network_metrics),
                    
                    'target_I': get_target_I_CoV_ISI(network_metrics),
                    'min_I': get_min_I_CoV_ISI(network_metrics),
                    'max_I': get_max_I_CoV_ISI(network_metrics),
                    
                    #'target_E': get_target_E_CoV_ISI(network_metrics),                    
                    'weight': 1, # TODO: update these with Nfactors
                },
                'fano_factor': {
                    'target_E': get_fano_factor_E_target(network_metrics),
                    'min_E': get_fano_factor_E_min(network_metrics),
                    'max_E': get_fano_factor_E_max(network_metrics),
                    
                    'target_I': get_fano_factor_I_target(network_metrics),
                    'min_I': get_fano_factor_I_min(network_metrics),
                    'max_I': get_fano_factor_I_max(network_metrics),
                    
                    'weight': 1, # TODO: update these with Nfactors
                }
            },
        },
        
        
        #Bursting Data
        'bursting_data': {
            'bursting_summary_data': {
                'MeanBurstRate': {
                    'target': bursting_data['bursting_summary_data'].get('mean_Burst_Rate'),
                    # 'min': get_min_burst(bursting_data['bursting_data_by_unit'], 'bursts', duration_seconds), # NOTE: Calculated as individual unit burst participation rate.
                    # 'max': get_max_burst(bursting_data['bursting_data_by_unit'], 'bursts', duration_seconds), # NOTE: Calculated as individual unit burst participation rate.
                    'min': 0,
                    'max': None,
                    'weight': 1,
                },                
                'MeanWithinBurstISI': {
                    'target': bursting_data['bursting_summary_data'].get('MeanWithinBurstISI'),
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_isi_within'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_isi_within'),
                    # 'target_E': get_target_withinBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_withinBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_withinBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_withinBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_withinBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_withinBurst_ISI_I(network_metrics),
                    # 'min': 0,
                    # 'max': None,
                    'weight': 1,
                },
                'CovWithinBurstISI': {
                    'target': bursting_data['bursting_summary_data'].get('CoVWithinBurstISI'),
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_within'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_within'),
                    # 'target_E': get_target_CoVWithinBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_CoVWithinBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_CoVWithinBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_CoVWithinBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_CoVWithinBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_CoVWithinBurst_ISI_I(network_metrics),
                    
                    'weight': 1,
                },
                'MeanOutsideBurstISI': {
                    'target': bursting_data['bursting_summary_data'].get('MeanOutsideBurstISI'),
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_isi_outside'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_isi_outside'),
                    # 'target_E': get_target_outsideBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_outsideBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_outsideBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_outsideBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_outsideBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_outsideBurst_ISI_I(network_metrics),
                    
                    # 'min': 0,
                    # 'max': None,
                    'weight': 1, 
                },
                'CoVOutsideBurstISI': {
                    'target': bursting_data['bursting_summary_data'].get('CoVOutsideBurstISI'),
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    # 'target_E': get_target_CoVOutsideBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_CoVOutsideBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_CoVOutsideBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_CoVOutsideBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_CoVOutsideBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_CoVOutsideBurst_ISI_I(network_metrics),
                    
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    'weight': 1,
                },
                # 'MeanNetworkISI': {
                #     'target': bursting_data['bursting_summary_data'].get('MeanNetworkISI'),
                #     'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     # 'target_E': get_target_meanNetwork_ISI_E(network_metrics),
                #     # 'target_I': get_target_meanNetwork_ISI_I(network_metrics),
                #     # 'min_E': get_min_meanNetwork_ISI_E(network_metrics),
                #     # 'min_I': get_min_meanNetwork_ISI_I(network_metrics),
                #     # 'max_E': get_max_meanNetwork_ISI_E(network_metrics),
                #     # 'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     # 'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     # 'min': 0,
                #     # 'max': None,
                #     'weight': 1,
                # },
                # 'CoVNetworkISI': {
                #     'target': bursting_data['bursting_summary_data'].get('CoVNetworkISI'),
                #     'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     # 'target_E': get_target_CoVNetwork_ISI_E(network_metrics),
                #     # 'target_I': get_target_CoVNetwork_ISI_I(network_metrics),
                #     # 'min_E': get_min_CoVNetwork_ISI_E(network_metrics),
                #     # 'min_I': get_min_CoVNetwork_ISI_I(network_metrics),
                #     # 'max_E': get_max_CoVNetwork_ISI_E(network_metrics),
                #     # 'max_I': get_max_CoVNetwork_ISI_I(network_metrics),
                    
                #     # 'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     # 'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     'weight': 1,
                # },
                'NumUnits': {
                    # 'target': bursting_data['bursting_summary_data'].get('NumUnits'),
                    # 'min': 1,
                    # 'max': None,
                    # 'weight': 1,
                    'target': network_metrics['classification_data']['no. of units'],
                    'target_I': len(network_metrics['classification_data']['inhibitory_neurons']),
                    'target_E': len(network_metrics['classification_data']['excitatory_neurons']),
                },
                'Number_Bursts': {
                    'target': bursting_data['bursting_summary_data'].get('Number_Bursts'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'mean_IBI': {
                    'target': bursting_data['bursting_summary_data'].get('mean_IBI'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'cov_IBI': {
                    'target': bursting_data['bursting_summary_data'].get('cov_IBI'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'mean_Burst_Peak': {
                    'target': bursting_data['bursting_summary_data'].get('mean_Burst_Peak'),
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_burst_peak'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_burst_peak'),
                    # 'min': None,
                    # 'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'cov_Burst_Peak': {
                    'target': bursting_data['bursting_summary_data'].get('cov_Burst_Peak'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'fano_factor': {
                    'target': bursting_data['bursting_summary_data'].get('fano_factor'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'baseline': {
                    'target': bursting_data['bursting_summary_data'].get('baseline'),
                    'min': None,
                    'max': None,
                    'weight': 1,
                },
            },
        },
        
        #Mega Bursting Data
        'mega_bursting_data': {
            'bursting_summary_data': {
                'MeanBurstRate': {
                    'target': mega_bursting_data['bursting_summary_data'].get('mean_Burst_Rate'),
                    # 'min': get_min_burst(bursting_data['bursting_data_by_unit'], 'bursts', duration_seconds), # NOTE: Calculated as individual unit burst participation rate.
                    # 'max': get_max_burst(bursting_data['bursting_data_by_unit'], 'bursts', duration_seconds), # NOTE: Calculated as individual unit burst participation rate.
                    'min': 0,
                    'max': None,
                    'weight': 1,
                },                
                'MeanWithinBurstISI': {
                    'target': mega_bursting_data['bursting_summary_data'].get('MeanWithinBurstISI'),
                    # 'min': get_min(mega_bursting_data['bursting_data_by_unit'], 'mean_isi_within'),
                    # 'max': get_max(mega_bursting_data['bursting_data_by_unit'], 'mean_isi_within'),
                    # 'target_E': get_target_withinBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_withinBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_withinBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_withinBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_withinBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_withinBurst_ISI_I(network_metrics),
                    # 'min': 0,
                    # 'max': None,
                    'weight': 1,
                },
                'CovWithinBurstISI': {
                    'target': mega_bursting_data['bursting_summary_data'].get('CoVWithinBurstISI'),
                    # 'min': get_min(mega_bursting_data['bursting_data_by_unit'], 'cov_isi_within'),
                    # 'max': get_max(mega_bursting_data['bursting_data_by_unit'], 'cov_isi_within'),
                    # 'target_E': get_target_CoVWithinBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_CoVWithinBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_CoVWithinBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_CoVWithinBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_CoVWithinBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_CoVWithinBurst_ISI_I(network_metrics),
                    
                    'weight': 1,
                },
                'MeanOutsideBurstISI': {
                    'target': mega_bursting_data['bursting_summary_data'].get('MeanOutsideBurstISI'),
                    # 'min': get_min(mega_bursting_data['bursting_data_by_unit'], 'mean_isi_outside'),
                    # 'max': get_max(mega_bursting_data['bursting_data_by_unit'], 'mean_isi_outside'),
                    # 'target_E': get_target_outsideBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_outsideBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_outsideBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_outsideBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_outsideBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_outsideBurst_ISI_I(network_metrics),
                    
                    # 'min': 0,
                    # 'max': None,
                    'weight': 1, 
                },
                'CoVOutsideBurstISI': {
                    'target': mega_bursting_data['bursting_summary_data'].get('CoVOutsideBurstISI'),
                    # 'min': get_min(mega_bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    # 'max': get_max(mega_bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    # 'target_E': get_target_CoVOutsideBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_CoVOutsideBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_CoVOutsideBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_CoVOutsideBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_CoVOutsideBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_CoVOutsideBurst_ISI_I(network_metrics),
                    
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    'weight': 1,
                },
                # 'MeanNetworkISI': {
                #     'target': bursting_data['bursting_summary_data'].get('MeanNetworkISI'),
                #     'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     # 'target_E': get_target_meanNetwork_ISI_E(network_metrics),
                #     # 'target_I': get_target_meanNetwork_ISI_I(network_metrics),
                #     # 'min_E': get_min_meanNetwork_ISI_E(network_metrics),
                #     # 'min_I': get_min_meanNetwork_ISI_I(network_metrics),
                #     # 'max_E': get_max_meanNetwork_ISI_E(network_metrics),
                #     # 'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     # 'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     # 'min': 0,
                #     # 'max': None,
                #     'weight': 1,
                # },
                # 'CoVNetworkISI': {
                #     'target': bursting_data['bursting_summary_data'].get('CoVNetworkISI'),
                #     'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     # 'target_E': get_target_CoVNetwork_ISI_E(network_metrics),
                #     # 'target_I': get_target_CoVNetwork_ISI_I(network_metrics),
                #     # 'min_E': get_min_CoVNetwork_ISI_E(network_metrics),
                #     # 'min_I': get_min_CoVNetwork_ISI_I(network_metrics),
                #     # 'max_E': get_max_CoVNetwork_ISI_E(network_metrics),
                #     # 'max_I': get_max_CoVNetwork_ISI_I(network_metrics),
                    
                #     # 'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     # 'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     'weight': 1,
                # },
                'NumUnits': {
                    # 'target': bursting_data['bursting_summary_data'].get('NumUnits'),
                    # 'min': 1,
                    # 'max': None,
                    # 'weight': 1,
                    'target': network_metrics['classification_data']['no. of units'],
                    'target_I': len(network_metrics['classification_data']['inhibitory_neurons']),
                    'target_E': len(network_metrics['classification_data']['excitatory_neurons']),
                },
                'Number_Bursts': {
                    'target': mega_bursting_data['bursting_summary_data'].get('Number_Bursts'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'mean_IBI': {
                    'target': mega_bursting_data['bursting_summary_data'].get('mean_IBI'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'cov_IBI': {
                    'target': mega_bursting_data['bursting_summary_data'].get('cov_IBI'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'mean_Burst_Peak': {
                    'target': mega_bursting_data['bursting_summary_data'].get('mean_Burst_Peak'),
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_burst_peak'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_burst_peak'),
                    # 'min': None,
                    # 'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'cov_Burst_Peak': {
                    'target': mega_bursting_data['bursting_summary_data'].get('cov_Burst_Peak'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'fano_factor': {
                    'target': mega_bursting_data['bursting_summary_data'].get('fano_factor'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'baseline': {
                    'target': mega_bursting_data['bursting_summary_data'].get('baseline'),
                    'min': None,
                    'max': None,
                    'weight': 1,
                },
            },
        },
    }
    return network_metric_targets

'''

'''