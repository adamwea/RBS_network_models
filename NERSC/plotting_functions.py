from USER_INPUTS import *
import numpy as np
#from analysis_functions import *
from analysis_functions import measure_network_activity
from matplotlib import pyplot as plt
import netpyne
import json
# import numpy as np
# import matplotlib.pyplot as plt

'''Report Plotting'''
def plot_params(elite_paths_cull, cfg_data, params, cgf_file_path):
    
    # Nested function to plot each configuration
    def plot_each_cfg(cfg_data, color='k', markersize=10, markerfacecolor='none'):
        # Find common keys between cfg_data and params
        param_dict = {}
        for key in cfg_data.keys():
            if key in params.keys():
                param_dict[key] = [cfg_data[key], params[key]]

        # Normalize values and ranges to 0-1
        for key, value in param_dict.items():
            try:
                assert len(value[1]) == 2, "Parameter without a range. Skipping."
            except:
                continue
            val = value.copy()
            val[0] = (val[0] - val[1][0]) / (val[1][1] - val[1][0])
            val[1] = [0, 1]
            #exempted_print(f"{key}: {val}")
            #assert 0 <= val[0] <= 1, f"Error: {key} has an invalid value."
            param_dict[key] = val

        # Separate constant and variable items
        constant_items = {key: value for key, value in param_dict.items() if isinstance(value[1], (int, float))}
        variable_items = {key: value for key, value in param_dict.items() if isinstance(value[1], list) and len(value[1]) == 2}
        erroneous_items = {key: value for key, value in param_dict.items() if isinstance(value[1], list) and len(value[1]) != 2}
        if len(erroneous_items) > 0:
            raise Exception(f"Error: {erroneous_items} has an invalid range.")
        param_dict = {**constant_items, **variable_items}

        # Plot each parameter
        for i, (key, value) in enumerate(param_dict.items()):
            try:
                assert len(value[1]) == 2, "Parameter without a range. Solid Red Line."
            except:
                ax.hlines(i, 0, 1, colors='r')
                continue
            ax.plot(value[0], i, marker='o', color=color, markersize=markersize, markerfacecolor=markerfacecolor)
            ax.plot(value[1], [i, i], 'k--')

        # Set y-axis labels
        ax.set_yticks(range(len(param_dict)))
        ax.set_yticklabels(param_dict.keys())

    # Create figure and axis
    #plt = plt.figure()
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 20)

    # Plot the main configuration data
    main_cfg_data = cfg_data.copy()
    main_cfg_path = cgf_file_path

    # Plot other configuration data for comparison
    print(f"Generating parameter comparison plot...")
    elite_cfg_paths = [f[1]['data_file_path'].replace('_data', '_cfg') for f in elite_paths_cull.items()]
    for file in elite_cfg_paths:
        if '.json' in file and 'cfg' in file and file != main_cfg_path and os.path.exists(file):
            cfg_data = json.load(open(file))['simConfig']
            plot_each_cfg(cfg_data, color='k', markersize=8, markerfacecolor='none')

    # Plot the main configuration data last so it is on top
    plot_each_cfg(main_cfg_data, color='r', markersize=10, markerfacecolor='red')

    # Set x-axis label and show plot
    ax.set_xlabel('Normalized Param Value')
    plt.tight_layout()
    return plt
'''Simulation Plotting Functions'''
def plot_network_activity(plotting_params, timeVector, firingRate, burstPeakTimes, burstPeakValues, thresholdBurst, rmsFiringRate, svg_mode = False): #rasterData, min_peak_distance = 1.0, binSize=0.02*1000, gaussianSigma=0.16*1000, thresholdBurst=1.2, figSize=(10, 6), saveFig=False, timeRange = None, figName='NetworkActivity.png'):
    #activate svg mode if specified
    if USER_svg_mode: svg_mode = True; print('SVG mode enabled')
    
    # Create a new figure with a specified size (width, height)
    figsize = plotting_params['figsize']
    assert figsize, 'figsize must be set to a tuple of two integers' #e.g. (10, 10)
    plt.figure(figsize=figsize)

    # Plot
    plt.subplot(1, 1, 1)
    plt.plot(timeVector, firingRate, color='black')
    plt.xlim([timeVector[0], timeVector[-1]])  # Restrict the plot to the first and last 100 ms   

    fig_ylim = plotting_params['ylim']
    if fig_ylim:       
        plt.ylim(fig_ylim)  # Set y-axis limits to min and max of firingRate
    else:
        yhigh100 = plotting_params['yhigh100']
        ylow100 = plotting_params['ylow100'] 
        assert yhigh100, 'USER_Activity_yhigh100 must be set to a float' #e.g. 1.05
        assert ylow100, 'USER_Activity_ylow100 must be set to a float' #e.g. 0.95
        plt.ylim([min(firingRate)*ylow100, max(firingRate)*yhigh100])  # Set y-axis limits to min and max of firingRate
    plt.ylabel('Spike Count')
    plt.xlabel('Time [s]')
    title_font = plotting_params['title_font']
    assert title_font, 'title_font must be set to an interger' #e.g. {'fontsize': 11}
    plt.title('Network Activity', fontsize=title_font)

    # Plot the threshold line and burst peaks
    plt.axhline(thresholdBurst * rmsFiringRate, color='gray', linestyle='--', label='Threshold')
    plt.plot(burstPeakTimes, burstPeakValues, 'or')  # Plot burst peaks as red circles

    if plotting_params['fitplot'] is not None:
        #name = 'Network Activity - Fitness'
        #net_activity_metrics = plotting_params['net_activity_metrics']
        targets = plotting_params['targets']
        if targets is not None:
            peak_amp_target = targets['pops']['big_burst_target']['target']
            baseline_target = targets['pops']['baseline_target']['target']
            plt.axhline(peak_amp_target, color='r', linestyle='--', label='Burst Target')
            plt.axhline(baseline_target, color='b', linestyle='--', label='Baseline Target')
            plt.legend()
    
    saveFig = plotting_params['saveFig']
    if saveFig:
        fig_path = saveFig
        fig_format = fig_path.split('.')[-1]
        plt.savefig(fig_path, bbox_inches='tight', format=fig_format)
        print(f'Network Activity plot saved to {fig_path}')
    else:
        plt.show()
def plot_network_activity_fitness(simData, net_activity_metrics, simLabel, batch_saveFolder, plot_save_path, fitnessVals, exp_mode=False, svg_mode = False, **kwargs):
    #activate svg mode if specified
    if USER_svg_mode: svg_mode = True; print('SVG mode enabled')
    
    #net_activity_metrics = net_activity_metrics
    rasterData = simData.copy()
    if not exp_mode:
        rasterData['spkt'] = np.array(rasterData['spkt'])/1000
        rasterData['t'] = np.array(rasterData['t'])/1000
    assert USER_raster_convolve_params, 'USER_raster_convolve_params needs to be specified in USER_INPUTS.py'
    net_activity_params = USER_raster_convolve_params #{'binSize': .03*1000, 'gaussianSigma': .12*1000, 'thresholdBurst': 1.0}
    binSize = net_activity_params['binSize']
    gaussianSigma = net_activity_params['gaussianSigma']
    thresholdBurst = net_activity_params['thresholdBurst']
    min_peak_distance = net_activity_params['min_peak_distance']

    assert len(rasterData['spkt']) > 0, 'Error: rasterData has no elements. burstPeak, baseline, slopeFitness, and IBI fitness set to 1000.'                       
    # Generate the network activity plot with a size of (10, 5)
    plotting_params = None
    #if plot:
    assert USER_plotting_params is not None, 'USER_plotting_params must be set in USER_INPUTS.py'
    plotting_params = USER_plotting_params['NetworkActivity']
    plotting_params['simLabel'] = simLabel
    plotting_params['batch_saveFolder'] = batch_saveFolder
    print('plotting_params:', plotting_params)
    #sys.exit()
    #prep to plot fitplot
    below_baseline_target = kwargs['pops']['baseline_target']['target'] * 0.95
    above_amplitude_target = kwargs['pops']['big_burst_target']['target']
    max_fire_rate = max(net_activity_metrics['firingRate'])
    min_fire_rate = min(net_activity_metrics['firingRate'])
    yhigh = max(above_amplitude_target, max_fire_rate) * 1.05
    ylow = min(below_baseline_target, min_fire_rate) * 0.95
    plotting_params['ylim'] = [ylow, yhigh]
    #override ylim
    yhigh = 3000
    plotting_params['ylim'] = [ylow, yhigh]
    #
    plotting_params['fitnessVals'] = fitnessVals
    plotting_params['targets'] = kwargs
    plotting_params['fitplot'] = True
    plotting_params['net_activity_metrics'] = net_activity_metrics
    plotting_params['fresh_plots'] = USER_plotting_params['fresh_plots']
    plotting_params['figsize'] = USER_plotting_params['figsize']
    #make rectangle instead of square
    plotting_params['figsize'] = (plotting_params['figsize'][0], plotting_params['figsize'][1]/2)
    #if plot_save_path defined, replace saveFig with plot_save_path
    #plot_save_path = USER_plotting_path
    if plot_save_path is not None: plotting_params['saveFig'] = plot_save_path
    print('plot_save_path:', plot_save_path)

    '''handle savepath'''
    default_name = 'NetworkActivity.png'
    figname = default_name
    saveFig = plotting_params['saveFig']
    assert saveFig, 'saveFig should be set to a relative path written as a string' #e.g. 'NERSC/plots/'
    batch_saveFolder = plotting_params['batch_saveFolder']
    assert batch_saveFolder, 'batch_saveFolder should be set to a relative path written as a string'
    simLabel = plotting_params['simLabel']
    assert simLabel, 'simLabel should be a string'
    #job_name = os.path.basename(batch_saveFolder)
    job_name = os.path.basename(os.path.dirname(batch_saveFolder))
    gen_folder = simLabel.split('_cand')[0]
    fig_path = os.path.join(saveFig, f'{job_name}/{gen_folder}/{simLabel}_{figname}')
    fig_dir = os.path.dirname(fig_path)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if 'output' in fig_path:
        print('')
    USER_fresh_plots = plotting_params['fresh_plots']
    if os.path.exists(fig_path) and USER_fresh_plots: pass
    elif os.path.exists(fig_path) and not USER_fresh_plots: 
        print(f'File {fig_path} already exists and USER_fresh_plots is set to False. Skipping plot.')
        return
    elif os.path.exists(fig_path) is False: pass
    else: raise ValueError(f'Idk how we got here. Logically.')
    #print(f'File {fig_path} already exists')

    if svg_mode: fig_path = fig_path.replace('.png', '.svg') #set svg mode as needed
    print('Save Path set to:', fig_path)
    plotting_params['saveFig'] = fig_path
    plot=True
    net_metrics = measure_network_activity(
        rasterData, 
        binSize=binSize, 
        gaussianSigma=gaussianSigma, 
        thresholdBurst=thresholdBurst,
        min_peak_distance=min_peak_distance,
        plot=plot,
        plotting_params = plotting_params,
        crop = USER_raster_crop
    )
def plot_raster(net_activity_metrics, plot_save_path, batch_saveFolder, simLabel, data_file_path, svg_mode = False):
    #activate svg mode if specified
    if USER_svg_mode: svg_mode = True; print('SVG mode enabled')
    
    #Attempt to generate the raster plot
    figname = 'raster_plot.png'
    timeVector = net_activity_metrics['timeVector']*1000 #convert back to ms
    timeRange = [timeVector[0], timeVector[-1]]
    #raster_plot_path = f'{batch_saveFolder}/{simLabel}_raster_plot.svg'
    job_name = os.path.basename(os.path.dirname(batch_saveFolder))
    #job_name = os.path.basename(batch_saveFolder)
    gen_folder = simLabel.split('_cand')[0]
    saveFig = USER_plotting_params['saveFig']
    #if plot_save_path defined, replace saveFig with plot_save_path
    print('plot_save_path:', plot_save_path)
    if plot_save_path is not None: saveFig = plot_save_path
    print('saveFig:', saveFig) 
    fig_path = os.path.join(saveFig, f'{job_name}/{gen_folder}/{simLabel}_{figname}')
    
    USER_fresh_plots = USER_plotting_params['fresh_plots']
    fig_size = USER_plotting_params['figsize']
    #make rectangle instead of square
    fig_size = (fig_size[0], fig_size[1]/2)
    if os.path.exists(fig_path) and USER_fresh_plots: pass
    elif os.path.exists(fig_path) and not USER_fresh_plots: 
        print(f'File {fig_path} already exists and USER_fresh_plots is set to False. Skipping plot.')
        return
    elif os.path.exists(fig_path) is False: pass
    else: raise ValueError(f'Idk how we got here. Logically.')
    #Apply SVG mode
    if svg_mode: fig_path = fig_path.replace('.png', '.svg')
    print('fig_path:', fig_path)
    try:
        netpyne.sim.analysis.plotRaster(saveFig=fig_path, 
                                    #timeRange = raster_activity_timeRange,
                                    timeRange = timeRange,
                                    showFig=False,
                                    labels = None, 
                                    figSize=fig_size)#, dpi=600)
        # #redo as png
        # cairosvg.svg2png(url=raster_plot_path, write_to=raster_plot_path.replace('.svg', '.png'))
        print('Raster plot saved to:', fig_path)
    except:
        print(f'Error generating raster plot from Data at: {data_file_path}')
        # raster_plot_path = None
        pass
def most_active_time_range(timeVector, sim_obj):
        '''subfunc'''
        def electric_slide(time_points, voltage_trace):
            #print('Getting most active time range...')
            #spike threshold, anything above zero is a spike
            spike_threshold = 0
            
            # Define the window size and step size in milliseconds
            window_size = 1000  # 1 second
            step_size = 1  # 1 millisecond

            # Convert the time points to an array for easier indexing
            time_points = np.array(time_points)

            # Initialize the maximum spike count and the start time of the window with the maximum spike count
            max_spike_count = 0
            max_spike_start_time = None

            # Convert the voltage trace to an array once, outside the loop
            voltage_trace = np.array(voltage_trace)

            # Detect zero-crossings for the entire voltage trace
            zero_crossings = np.where(np.diff(np.sign(voltage_trace)))[0]
            zero_crossing_times = time_points[zero_crossings]

            # Initialize the maximum spike count and start time
            max_spike_count = 0
            max_spike_start_time = None

            # Slide the window over the voltage trace
            for start_time in np.arange(time_points[0], time_points[-1] - window_size + step_size, step_size):
                # Get the end time of the current window
                end_time = start_time + window_size

                # Count the number of zero-crossings in the current window
                spike_count = np.sum((zero_crossing_times >= start_time) & (zero_crossing_times < end_time))

                # If the current window has more spikes than the previous maximum, update the maximum
                if spike_count > max_spike_count:
                    max_spike_count = spike_count
                    max_spike_start_time = start_time
            
            #if no spikes are found, return full time range
            if max_spike_start_time is None:
                return [0, time_points[-1]]                    
            # The time range with the most spiking activity is from max_spike_start_time to max_spike_start_time + window_size
            timeRange = [max_spike_start_time, max_spike_start_time + window_size]
            #if any values are < 0, make them 0
            return timeRange
        '''main'''
        # Get the time range of the most active part of the simulation for each neuron
        # Get the voltage trace for a specific cell
        # Get the keys (GIDs) of the neurons
        neuron_gids = list(sim_obj.allSimData['soma_voltage'].keys())
        time_points = sim_obj.allSimData['t']
        time_ranges = {}

        for gid in neuron_gids:
            # Get the voltage trace for the neuron
            voltage_trace = sim_obj.allSimData['soma_voltage'][gid]
            # Create a zip object from time_points and voltage trace
            pairs = zip(time_points, voltage_trace)
            # Filter voltage trace based on filtered time_points
            voltage_trace = [v for t, v in pairs if t >= timeVector[0] and t <= timeVector[-1]]
            # Filter time_points
            time_points_filtered = [t for t in time_points if t >= timeVector[0] and t <= timeVector[-1]]
            # Get time range
            time_range = electric_slide(time_points_filtered, voltage_trace)
            # Store time range in dictionary
            time_ranges[gid] = time_range

        return time_ranges
def plot_trace_example(neuron_metrics, net_activity_metrics, plot_save_path, batch_saveFolder, simLabel, data_file_path, svg_mode = False):
    #activate svg mode if specified
    if USER_svg_mode: svg_mode = True; print('SVG mode enabled')
    # Attempt to generate sample trace for an excitatory example neuron
    try:
        figname = 'sample_trace'
        job_name = os.path.basename(os.path.dirname(batch_saveFolder))
        gen_folder = simLabel.split('_cand')[0]
        saveFig = USER_plotting_params['saveFig']
        #if plot_save_path defined, replace saveFig with plot_save_path
        if plot_save_path is not None: saveFig = plot_save_path
        fig_path = os.path.join(saveFig, f'{job_name}/{gen_folder}/{simLabel}_{figname}')
        USER_fresh_plots = USER_plotting_params['fresh_plots']
        if os.path.exists(fig_path) and USER_fresh_plots: pass
        elif os.path.exists(fig_path) and not USER_fresh_plots: 
            print(f'File {fig_path} already exists and USER_fresh_plots is set to False. Skipping plot.')
            return
        elif os.path.exists(fig_path) is False: pass
        else: raise ValueError(f'Idk how we got here. Logically.')

        sim_obj = netpyne.sim
        timeVector = np.array(net_activity_metrics['timeVector']*1000) #convert back to ms
        print('Getting most active time ranges for sample traces...')
        timeRanges = most_active_time_range(timeVector, sim_obj)
        num_cells = len(timeRanges)
        fig_height_per_cell = USER_plotting_params['figsize'][1] / num_cells
        titles = []

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(num_cells, 1, figsize=USER_plotting_params['figsize'])

        for i, (gid, timeRange) in enumerate(timeRanges.items()):
            gid_num = int(gid.split('_')[-1])
            if gid_num in neuron_metrics['E_neurons']: 
                title = f'{gid}_excitatory'
                type = 'E'
                num = gid_num
            elif gid_num in neuron_metrics['I_neurons']: 
                title = f'{gid}_inibitory'
                type = 'I'
                num = gid_num - np.nanmax(neuron_metrics['E_neurons'])
            titles.append(title)
            #include =[type, num]
            include = [gid_num]
            sample_trace = sim_obj.analysis.plotTraces(
                include=include,
                overlay=True,
                oneFigPer='trace',
                title=title,
                timeRange=timeRange,
                showFig=False,
                figSize=(USER_plotting_params['figsize'][0], fig_height_per_cell)
            )
            # Add the plot to the subplot
            #axs[i].plot(sample_trace)
            v_key = f'cell_{gid_num}_soma_voltage'
            v = sample_trace[1]['tracesData'][0][v_key]
            t = sample_trace[1]['tracesData'][0]['t']
            if len(v) > len(t): v = v[:len(t)]
            if len(t) > len(v): t = t[:len(v)]
            color = 'gold' if type == 'I' else 'blue'
            axs[i].plot(t, v, color=color)
            axs[i].set_title(title)
            #if inhib, make line yellow, else blue

        fig.tight_layout(rect=[0, 0.03, 1, 1])
        fig_path_combined = f'{fig_path}_combined.svg' if svg_mode else f'{fig_path}_combined.png'
        fig.savefig(fig_path_combined)
        print(f'Sample trace plot saved to {fig_path_combined}')
    except:
        print(f'Error generating sample trace plot from Data at: {data_file_path}')
        pass
def plot_connections(plot_save_path, batch_saveFolder, simLabel, data_file_path, svg_mode = False):
    #activate svg mode if specified
    if USER_svg_mode: svg_mode = True; print('SVG mode enabled')
    
    # Attempt to generate sample trace for an excitatory example neuron
    try:
        print('Generating connections plot...')
        figname = 'connections.png'
        # job_name = os.path.basename(batch_saveFolder)
        job_name = os.path.basename(os.path.dirname(batch_saveFolder))
        gen_folder = simLabel.split('_cand')[0]
        saveFig = USER_plotting_params['saveFig']
        #if plot_save_path defined, replace saveFig with plot_save_path
        if plot_save_path is not None: saveFig = plot_save_path
        fig_path = os.path.join(saveFig, f'{job_name}/{gen_folder}/{simLabel}_{figname}')
        
        #sys.exit()
        USER_fresh_plots = USER_plotting_params['fresh_plots']
        if os.path.exists(fig_path) and USER_fresh_plots: pass
        elif os.path.exists(fig_path) and not USER_fresh_plots: 
            print(f'File {fig_path} already exists and USER_fresh_plots is set to False. Skipping plot.')
            return
        elif os.path.exists(fig_path) is False: pass
        else: raise ValueError(f'Idk how we got here. Logically.')
        #Apply SVG mode
        if svg_mode: fig_path = fig_path.replace('.png', '.svg')
        print('fig_path:', fig_path)

        sim_obj = netpyne.sim
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        # Create individual plots and save as PNG
        sim_obj.analysis.plot2Dnet(saveFig=fig_path, showFig=False, showConns=True, figSize=(USER_plotting_params['figsize'][0], USER_plotting_params['figsize'][1]/2))
        print(f'Connections plot saved to {fig_path}')

    except:
        print(f'Error generating connections from Data at: {data_file_path}')
        #sample_trace_path_E = None
        pass
'''experimental data plotting'''
def plot_experimental_raster(real_spike_data, sampling_rate=10000, xlim_start=30000, xlim_end=300000):
    """
    Plot a raster plot of spike trains with the top 30% of firing neurons in gold and the rest in light blue.
    
    Parameters:
    - real_spike_data: List of lists or numpy array, each inner list/array represents a spike train.
    - sampling_rate: Sampling rate in Hz, default is 10000.
    - xlim_start: Start of the x-axis in milliseconds, default is 30000 (30 seconds).
    - xlim_end: End of the x-axis in milliseconds, default is 300000 (300 seconds).
    """
    
    # Ensure real_spike_data is a numpy array
    real_spike_data = np.array(real_spike_data)
    
    # Sort the spike trains by firing rate, highest to lowest
    sorted_spike_data = sorted(real_spike_data, key=lambda x: np.sum(x), reverse=False)
    
    # Convert the sorted list back to a numpy array for plotting
    sorted_spike_data = np.array(sorted_spike_data)
    
    # Calculate the threshold for the bottom 70% of firing neurons
    num_neurons = len(sorted_spike_data)
    bottom_70_percent_threshold = int(0.7 * num_neurons)
    
    # Plot raster plot of spike trains
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Loop through each spike train and plot the spikes
    for neuron_id, spike_train in enumerate(sorted_spike_data):
        spike_times = np.where(spike_train == 1)[0] / sampling_rate * 1000  # Convert to milliseconds
        color = '#FFD700' if neuron_id > bottom_70_percent_threshold else '#ADD8E6'
        ax.vlines(spike_times, neuron_id + 0.5, neuron_id + 1.5, color=color)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('IPNs')
    ax.set_title('Spike Raster Plot')
    ax.set_ylim(0, len(sorted_spike_data) + 0.5)  # Ensure correct y-axis direction
    ax.set_xlim(xlim_start, xlim_end)  # Set x-axis limits based on parameters
    
    plt.show()

# Example usage
# plot_experimental_data(real_spike_data, xlim_start=30000, xlim_end=300000)
