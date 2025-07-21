# imports
import os
import numpy as np
import traceback
import json
from MEA_Analysis.MEAProcessingLibrary.mea_processing_library import get_data_obj_groups


# subfuncs ===================================================================
def compute_partial_network_metrics(
        recording_object, 
        sorting_object,
        #wf_extractor, 
        sorting_analyzer,
        stream_num, 
        conv_params,
        mega_params,
        #convolution_params_path, 
        output_path,
        save_path=None,
        bursting_plot_path=None, bursting_fig_path=None, 
        plot=False, 
        limit_seconds = None,
        debug_mode = False,
        overwrite=False,
        **kwargs):
    
    # import
    from MEA_Analysis.NetworkAnalysis_aw.compute_network_metrics import compute_network_metrics
    
    # Subfunctions ======================================
    def get_metrics(sorting_object, recording_object, sorting_analyzer, conv_params, mega_params, debug_mode=False, **kwargs):
        # get network metrics
        well_id = f'well{str(0).zfill(2)}{stream_num}'
        well_recording_segment = recording_object 
        
        # define paths based on wf_extractor
        analyzer_folder = sorting_analyzer.folder._str
        #wf_extractor_info = wf_extractor._save_data()
        #wf_folder = wf_extractor.folder._str
        #print(f"wf_folder: {wf_folder}")
        
        # replace 'waveforms' with 'dtw'
        dtw_folder = analyzer_folder.replace('analyzer', 'dtw')
        #dtw_folder = wf_folder.replace('waveforms', 'dtw')
        #dtw_output = os.path.join(dtw_folder, 'dtw_output')
        dtw_temp = os.path.join(dtw_folder, 'dtw_temp')
        #mega_dtw_output = os.path.join(dtw_folder, 'mega_dtw_output')
        # print(f"dtw_output: {dtw_output}")
        # print(f"dtw_temp: {dtw_temp}")
        # print(f"mega_dtw: {mega_dtw_output}")
        
        try:
            source = 'experimental'
            netkwargs = {
                'debug_mode': debug_mode,
                'well_id': well_id,
                'stream_num': stream_num,
                'recording_object': recording_object,
                'sorting_object': sorting_object,
                #'wf_extractor': wf_extractor,
                'sorting_analyzer': sorting_analyzer,
                'run_parallel': True,
                'max_workers': kwargs['max_workers'],
                #'max_workers': 32,
                #'max_workers': 16,
                #'max_workers' : 256,
                #'plot_wfs': True,
                
                #'plot_wfs': False,
                'plot_wfs': kwargs['plot_wfs'],
                'burst_sequencing': True,
                #'burst_sequencing': False,
                
                # debug - move to run script later #HACK
                # fitness_save_path = kwargs['fitness_save_path']
                # basename = os.path.basename(fitness_save_path)
                # sa_dir = os.path.dirname(fitness_save_path)
                # # remove .json from basename
                # basename = basename.replace('.json', '')
                # dtw_dir = os.path.join(sa_dir, basename, 'dtw_temp')    
                # kwargs['dtw_temp'] = dtw_dir
                
                'dtw_temp': dtw_temp,
                # 'dtw_output': dtw_output,
                # 'mega_dtw': mega_dtw_output,
                
                #analysis options
                'compute_spike_metrics': True, # compute spike metrics
                'compute_burst_metrics': True, # compute burst metrics
                'classify_units': False, # classify units
                #'locate_units': False, # locate units
                'compute_dynamic_time_warping': False, # compute dynamic time warping
                'compute_summary_metrics': False,                
                
            }
            network_metrics = compute_network_metrics(conv_params, mega_params, source, **netkwargs)
            #network_metrics = get_experimental_network_metrics_v3(sorting_object, well_recording_segment, wf_extractor, conv_params, mega_params, debug_mode=debug_mode, **kwargs)
            return network_metrics
        except Exception as e:
            print(f'Error: Could not get network metrics for {well_id}')
            traceback.print_exc()
            return (e, traceback.format_exc())
        
    # Main ======================================
    # assertions
    assert sorting_object is not None, f"Error: sorting_object is None"
    
    # create output path
    output_dir = output_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # prepare save path
    bin_size = mega_params['binSize']
    gaussian_sigma = mega_params['gaussianSigma']
    save_path = os.path.join(output_dir, f"metrics_{bin_size}_{gaussian_sigma}.npy")
    
    # check if save path already exists and overwrite is False
    if os.path.exists(save_path) and not overwrite:
        print(f"Skipping computation, file already exists: {save_path}")
        return None, save_path
        
    # get network metrics
    network_metrics = get_metrics(sorting_object, recording_object, sorting_analyzer, conv_params, mega_params, debug_mode=debug_mode, **kwargs)
        
    # get recording details
    print("Saving network metrics as numpy...")
    # recording_details = kwargs['details']
    # projectName = recording_details['projectName']
    # date = recording_details['date']
    # chipID = recording_details['chipID']
    # scanType = recording_details['scanType']
    # runID = recording_details['runID']
    
    # # create output path
    # output_dir = output_path
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #save_path = os.path.join(output_dir, f"metrics.npy")
    
    # include bin_size and gaussian_sigma in the save path for clarity
    # bin_size = mega_params['binSize']
    # gaussian_sigma = mega_params['gaussianSigma']
    # save_path = os.path.join(output_dir, f"metrics_{bin_size}_{gaussian_sigma}.npy")
    #print(f"Saving network metrics to {save_path}")
    np.save(save_path, network_metrics)

    # return network metrics and save path
    print(f"Saved network metrics to {save_path}")
    return network_metrics, save_path


# Typical usage function to run the analysis ================
def run_analysis(
    #raw_data_paths, 
    sorted_data_dirs = None, 
    output_dirs = None, 
    stream_select=None, 
    plot=True, 
    #conv_params=None,
    #mega_params=None,
    bin_size_range=None,
    gaussian_sigma_range=None,
    limit_seconds=None,
    plot_wfs=False,
    max_workers = 4, # safe for all computers - if not specified, will use all available cores
    debug_mode = False,
    **kwargs,
    ):
    
    ## subfunctions =================================================================
    def initialize_output_dir(output_dirs):
        assert output_dirs is not None, f"Error: output_dirs is None"
        for output_dir in output_dirs:
            assert output_dir is not None, f"Error: output_dir is None"
            output_dir = os.path.abspath(output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        return output_dirs
    
    def initialize_h5_paths(sorted_data_dirs):
        assert sorted_data_dirs is not None, f"Error: sorted_data_dirs is None"
        h5_paths = []
        for sorted_data_dir in sorted_data_dirs:
            # look for a file called spikeinterface_recording.json in the directory
            if not os.path.exists(sorted_data_dir):
                print(f"Error: sorted_data_dir does not exist. {sorted_data_dir}")
                raise FileNotFoundError(f"Error: sorted_data_dir does not exist. {sorted_data_dir}")
                # continue
            json_path = os.path.join(sorted_data_dir, 'spikeinterface_recording.json')
            if not os.path.exists(json_path):
                print(f"Error: sorted_data_dir does not contain spikeinterface_recording.json. {sorted_data_dir}")
                raise FileNotFoundError(f"Error: sorted_data_dir does not contain spikeinterface_recording.json. {sorted_data_dir}")
                #continue
            # load json file
            with open(json_path, 'r') as f:
                recording_details = json.load(f)
            h5_path = recording_details['kwargs']['file_path']
            
            #HACK stupid patch to fix old sorting path
            if 'zinputs' in h5_path:
                h5_path = h5_path.replace('zinputs', 'z_raw_data')
                
            h5_paths.append(h5_path)
        return h5_paths
    
    def initialize_sorted_output_dirs(sorted_data_dirs):
        assert sorted_data_dirs is not None, f"Error: sorted_data_dir is None"  
        #sorted_data_dir = os.path.abspath(sorted_data_dir)
        
        # iterate through sorter_output_dir and get all well output directories
        sorted_output_folders = []
        for sorted_data_dir in sorted_data_dirs:
            for root, dirs, files in os.walk(sorted_data_dir):
                if root.endswith('sorter_output'):
                    if not os.path.exists(root):
                        print(f"Error: sorted_output_folder does not exist. {root}")
                        #continue
                        raise FileNotFoundError(f"Error: sorted_output_folder does not exist. {root}")
                    sorted_output_folders.append(os.path.join(root))
        return sorted_output_folders
    
    def validate_three_objects(well_data):
        #init skip flag
        skip = False
        
        recording_segments, sort_obj, wf_extractor = well_data
        
        objs = [recording_segments, sort_obj, wf_extractor]
        for obj in objs:
            try:
                for i, item in enumerate(obj):
                    if isinstance(item, Exception):
                        skip = True
                        return objs, skip
            except:
                if isinstance(obj, Exception):
                    skip = True
                    return objs, skip
        return objs, skip
    
    #init paths
    output_dirs = initialize_output_dir(output_dirs)
    h5_paths = initialize_h5_paths(sorted_data_dirs)
    sorted_output_folders = initialize_sorted_output_dirs(sorted_data_dirs)
    
    # get convolution and mega params from bin_size_range and gaussian_sigma_range
    if bin_size_range is None or gaussian_sigma_range is None:
        raise ValueError("bin_size_range and gaussian_sigma_range must be specified")
    if len(bin_size_range) != 3 or len(gaussian_sigma_range) != 3:
        raise ValueError("bin_size_range and gaussian_sigma_range must be of length 3")
    
    # bin_size_range and gaussian_sigma_range are lists of [min, max, step], build map of params
    param_map = {
        'binSize': np.arange(bin_size_range[0], bin_size_range[1], bin_size_range[2]).tolist(),
        'gaussianSigma': np.arange(gaussian_sigma_range[0], gaussian_sigma_range[1], gaussian_sigma_range[2]).tolist(),
        'thresholdBurst': None,
        'min_peak_distance': None,
        'prominence': 1,
    }
     
    # get paired data objects - network analysis requires both recording and sorting objects
    well_data_list, recording_details, path_pairs = get_data_obj_groups(h5_paths, sorted_output_folders)
    
    # iterate through data_obj_list and get network metrics for each pair
    for i, well_data in enumerate(well_data_list):
        
        # choose to skip or not by validating objects
        objs, skip = validate_three_objects(well_data)
        if skip: continue
        #recording_segments, sort_obj, wf_extractor = objs
        recording_segments, sort_obj, sorting_analyzer = objs
        recording_segment = recording_segments[0] # HACK: this function is really only going to be used for network scans... but if I try to use it for multiple segments, I'll need to update this.
        
        # init kwargs         
        stream_id = recording_segment.stream_id
        stream_num = int(stream_id.split('well')[1][:3])
        kwargs = recording_details.copy()
        kwargs['plot_wfs'] = plot_wfs
        kwargs['max_workers'] = max_workers
        
        # init print statements
        print(f"Analyzing network data collected in well{str(0).zfill(2)}{stream_num}...")
        
        # get output directory for this well        
        output_dir = output_dirs[i]
        
        bin_size_list = param_map['binSize']
        gaussian_sigma_list = param_map['gaussianSigma']
        for bin_size in bin_size_list:
            for gaussian_sigma in gaussian_sigma_list:
                # create convolution params
                conv_params = {
                    'binSize': bin_size / 10, # 2025-06-05 15:00:48 new thinking, regular bursting just needs to be smaller bins than mega bursting
                    'gaussianSigma': gaussian_sigma,
                    'thresholdBurst': param_map['thresholdBurst'],
                    'min_peak_distance': param_map['min_peak_distance'],
                    'prominence': param_map['prominence'],
                }
                
                # create mega params
                mega_params = {
                    'binSize': bin_size,
                    'gaussianSigma': gaussian_sigma,
                    'thresholdBurst': param_map['thresholdBurst'],
                    'min_peak_distance': param_map['min_peak_distance'],
                    'prominence': param_map['prominence'],
                }        
        
                # run analysis
                kwargs['recording_object'] = recording_segment
                kwargs['sorting_object'] = sort_obj
                kwargs['sorting_analyzer'] = sorting_analyzer
                kwargs['stream_num'] = stream_num
                kwargs['conv_params'] = conv_params
                kwargs['mega_params'] = mega_params
                kwargs['output_path'] = output_dir
                #kwargs['plot'] = plot
                kwargs['details'] = recording_details
                kwargs['limit_seconds'] = limit_seconds
                kwargs['debug_mode'] = debug_mode
                kwargs['overwrite'] = False # aw 2025-01-25 12:16:01 - don't overwrite existing files, skip if they exist
                network_metrics, save_path = compute_partial_network_metrics(**kwargs)
                #print(f"Network metrics saved to {save_path}") 
    print('done')
    return

# batch optimized version of run_analysis ================
# Rewritten batch_optimized_run_analysis with pickling-safe task dispatch

import os
import json
import numpy as np
import time
from multiprocessing import Process, Queue, Value, Lock, cpu_count
from threading import Thread
from rich.console import Console
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, TimeRemainingColumn
from pathlib import Path
#from MEA_Analysis.MEAProcessingLibrary.mea_processing_library import get_data_obj_groups
#from MEA_Analysis.NetworkAnalysis_aw.compute_network_metrics_sensitivity import compute_partial_network_metrics

console = Console()

# def child_worker(task, completed_tasks, lock):
#     try:
#         compute_partial_network_metrics(**task)
#     except Exception as e:
#         console.print(f"[red]Error processing task: {e}")
#     with lock:
#         completed_tasks.value += 1

def worker(task_queue, completed_tasks, lock, child_max_workers):
    local_tasks = []
    while not task_queue.empty(): #and len(local_tasks) < child_max_workers:
        try:
            task = task_queue.get_nowait()
            h5_path = task['recording_path']
            sorting_path = task['sorting_path']

            # Load objects (recording, sorting, analyzer)
            well_data_list, recording_details, _ = get_data_obj_groups([h5_path], [sorting_path])
            if not well_data_list or isinstance(well_data_list[0], Exception):
                raise RuntimeError(f"Failed to load data objects from {h5_path} and {sorting_path}")

            recording_segments, sort_obj, sorting_analyzer = well_data_list[0]
            recording_segment = recording_segments[0]
            stream_num = int(recording_segment.stream_id.split('well')[1][:3])

            resolved_task = {
                'recording_object': recording_segment,
                'sorting_object': sort_obj,
                'sorting_analyzer': sorting_analyzer,
                'stream_num': stream_num,
                **task['params'],
            }
            
            # HACK: redefine childworkers as workers here
            resolved_task['max_workers'] = child_max_workers
            if 'child_max_workers' in resolved_task:
                del resolved_task['child_max_workers']             # remove 'child_max_workers' from params

            #local_tasks.append(resolved_task)      
            compute_partial_network_metrics(**resolved_task)
        except Exception as e:
            console.print(f"[red]Parent worker failed to prepare task: {e}")
            continue


    
    # processes = [Process(target=child_worker, args=(task, completed_tasks, lock)) for task in local_tasks]
    # for p in processes:
    #     p.start()
    # for p in processes:
    #     p.join()

def monitor_progress(total_tasks, completed_tasks, lock):
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing", total=total_tasks)
        while not progress.finished:
            time.sleep(5)
            with lock:
                done = completed_tasks.value
            progress.update(task, completed=done)
            if done >= total_tasks:
                break

def run_manual_process_pool(task_queue, total_tasks, completed_tasks, lock, child_max_workers, max_workers):
    if total_tasks == 0:
        console.print("[bold red]No tasks to process.")
        return

    monitor_thread = Thread(target=monitor_progress, args=(total_tasks, completed_tasks, lock))
    monitor_thread.start()

    ##num_parents = min(cpu_count() // child_max_workers, 8)
    ##num_parents = kwargs.get('max_workers', cpu_count() // child_max_workers)
    num_parents = min(max_workers, cpu_count() // child_max_workers)
    console.print(f"[green]Launching {num_parents} parent processes with up to {child_max_workers} children each.")

    processes = [Process(target=worker, args=(task_queue, completed_tasks, lock, child_max_workers)) for _ in range(num_parents)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    monitor_thread.join()
    console.print("[bold green]Batch analysis complete.")

def batch_optimized_run_analysis(
    sorted_data_dirs, output_dirs, bin_size_range, gaussian_sigma_range,
    limit_seconds=None, plot_wfs=False, max_workers=4, child_max_workers=8,
    debug_mode=False, **kwargs):

    def get_params_grid():
        bins = np.arange(*bin_size_range).tolist()
        sigmas = np.arange(*gaussian_sigma_range).tolist()
        for b in bins:
            for s in sigmas:
                yield {
                    'binSize': b,
                    'gaussianSigma': s,
                    'thresholdBurst': None,
                    'min_peak_distance': None,
                    'prominence': 1,
                }

    def resolve_json_path(sorted_dir):
        json_path = os.path.join(sorted_dir, 'spikeinterface_recording.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} not found")
        with open(json_path, 'r') as f:
            data = json.load(f)
        h5_path = data['kwargs']['file_path']
        if 'zinputs' in h5_path:
            h5_path = h5_path.replace('zinputs', 'z_raw_data')
        return h5_path

    def find_sorter_output_dirs(base_dirs):
        output_dirs = []
        for d in base_dirs:
            for root, dirs, files in os.walk(d):
                if root.endswith('sorter_output'):
                    output_dirs.append(root)
        return output_dirs

    # Ensure output dirs exist
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)

    h5_paths = [resolve_json_path(d) for d in sorted_data_dirs]
    sorter_output_dirs = find_sorter_output_dirs(sorted_data_dirs)
    task_queue = Queue()

    for i, (h5_path, sorting_path, output_dir) in enumerate(zip(h5_paths, sorter_output_dirs, output_dirs)):
        for param in get_params_grid():
            conv_params = param.copy()
            conv_params['binSize'] /= 10
            mega_params = param.copy()

            task_queue.put({
                'recording_path': h5_path,
                'sorting_path': sorting_path,
                'params': {
                    'conv_params': conv_params,
                    'mega_params': mega_params,
                    'output_path': output_dir,
                    'plot_wfs': plot_wfs,
                    
                    # workers
                    'max_workers': max_workers,
                    'child_max_workers': child_max_workers,
                    
                    'limit_seconds': limit_seconds,
                    'debug_mode': debug_mode,
                    'overwrite': False,
                }
            })

    total_tasks = task_queue.qsize()
    completed_tasks = Value('i', 0)
    lock = Lock()

    run_manual_process_pool(task_queue, total_tasks, completed_tasks, lock, child_max_workers, max_workers)
    console.print("[bold green]Batch analysis complete.")
