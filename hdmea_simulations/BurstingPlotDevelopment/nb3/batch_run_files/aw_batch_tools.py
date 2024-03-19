'''
Collection of useful functions written for batch processing of the HD-MEA simulations using NetPyNE
'''

#imports
import os
from netpyne import sim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks
from scipy.stats import norm
import svgutils.transform as sg

### Figure Functions

# from svg.path import parse_path
# from xml.dom.minidom import parse

# def load_svg_paths(svg_file_path):
#     # Parse the SVG file
#     doc = parse(svg_file_path)
#     path_strings = [path.getAttribute('d') for path
#                     in doc.getElementsByTagName('path')]

#     # Convert the paths to matplotlib objects
#     paths = [parse_path(path_string) for path_string in path_strings]

#     return paths

# def load_svgs(image1_path, image2_path):
#     # Load the SVGs
#     image1_paths = load_svg_paths(image1_path)
#     image2_paths = load_svg_paths(image2_path)

#     return image1_paths, image2_paths

def get_batchrun_info(root, file):
    file_path = os.path.join(root, file)
    batchrun_folder = os.path.basename(os.path.normpath(root))  
    batch_key = file.split(batchrun_folder)[1].split('data')[0]
    return file_path, batchrun_folder, batch_key

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
    margin_width = 200
    # Find the indices of timeVector that correspond to the first and last 100 ms
    start_index = np.where(timeVector >= margin_width)[0][0]
    end_index = np.where(timeVector <= max(timeVector) - margin_width)[0][-1]
    #end_index = np.where(timeVector <= max(timeVector))[0][-1]

    # Plot the smoothed network activity
    plt.subplot(1, 1, 1)
    plt.plot(timeVector[start_index:end_index], firingRate[start_index:end_index], color='black')
    plt.xlim([timeVector[0], timeVector[-1]])  # Restrict the plot to the first and last 100 ms
    plt.ylim([min(firingRate[start_index:end_index])*0.95, max(firingRate[start_index:end_index])*1.05])  # Set y-axis limits to min and max of firingRate
    plt.ylabel('Firing Rate [Hz]')
    plt.xlabel('Time [ms]')
    plt.title('Network Activity', fontsize=11)

    # Step 3: Peak detection on the smoothed firing rate curve
    rmsFiringRate = np.sqrt(np.mean(firingRate**2))  # Calculate RMS of the firing rate
    peaks, properties = find_peaks(firingRate, height=thresholdBurst * rmsFiringRate, distance=min_peak_distance)  # Find peaks above the threshold
    burstPeakTimes = timeVector[peaks]  # Convert peak indices to times
    burstPeakValues = properties['peak_heights']  # Get the peak values

    # Get the indices of the valid peaks
    valid_peak_indices = np.where((burstPeakTimes > timeVector[start_index]) & (burstPeakTimes < timeVector[end_index]))[0]

    # Filter burstPeakTimes and burstPeakValues using the valid peak indices
    burstPeakTimes = burstPeakTimes[valid_peak_indices]
    burstPeakValues = burstPeakValues[valid_peak_indices]

    # Plot the threshold line and burst peaks
    plt.plot(np.arange(timeVector[-1]), thresholdBurst * rmsFiringRate * np.ones(np.ceil(timeVector[-1]).astype(int)), color='gray')
    plt.plot(burstPeakTimes, burstPeakValues, 'or')  # Plot burst peaks as red circles    

    if saveFig:
        if isinstance(saveFig, str):
            # plt.savefig(saveFig, dpi=600, bbox_inches='tight')
            # plt.savefig(saveFig.replace('.png', '.svg'), bbox_inches='tight')
            plt.savefig(saveFig, bbox_inches='tight')
        else:
            # plt.savefig(figName, dpi=600, bbox_inches='tight')
            # plt.savefig(figName.replace('.png', '.svg'), bbox_inches='tight')
            plt.savefig(figName, bbox_inches='tight')
    #plt.show()
            
def generate_all_figures(fresh_figs = False, batchLabel = 'batch', batch_path = None, size = 6, net_activity_params = {'binSize': 3, 'gaussianSigma': 12, 'thresholdBurst': 1.0}):
    
    
    #relative figure sizes in inches
    raster_height = size
    raster_width = size
    sample_trace_height = size
    sample_trace_width = size
    network_activity_height = raster_height
    network_activity_width = raster_width
    locations_height = raster_height
    locations_width = locations_height*2
    conn_mat_height = raster_height
    conn_mat_width = conn_mat_height
    
    def generate_primary_figures(root, file, size = size, net_activity_params = {'binSize': 3, 'gaussianSigma': 12, 'thresholdBurst': 1.0}):

        #get net activity params
        binSize = net_activity_params['binSize']
        gaussianSigma = net_activity_params['gaussianSigma']
        thresholdBurst = net_activity_params['thresholdBurst']
        
        #get file info
        file_path, batchrun_folder, batch_key = get_batchrun_info(root, file)

        #prepare destination folder
        NetworkBurstingFolder = 'NetworkBurst_and_Raster_Figs'
        if not os.path.exists(f'{root}/{NetworkBurstingFolder}'):
            os.makedirs(f'{root}/{NetworkBurstingFolder}')
            print(f'Created folder: {root}/{NetworkBurstingFolder}')
        NetBurst_path = f'{root}/{NetworkBurstingFolder}'    
            
        # Define the paths to the raster plot and network activity images
        # raster_plot_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_raster.png'
        # network_activity_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_network_activity_def.png'
        # TWODnet_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_plot_2Dnet.png'
        # locs_path = f'{root}/{NetworkBurstingFolder}/locs.png'
        # conn_mat_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_conn_matrix.png'
        # sample_trace_path_E = f'{root}/{NetworkBurstingFolder}/{batch_key}_sample_trace_E.png'
        # sample_trace_path_I = f'{root}/{NetworkBurstingFolder}/{batch_key}_sample_trace_I.png'
        raster_plot_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_raster.svg'
        network_activity_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_network_activity_def.svg'
        TWODnet_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_plot_2Dnet.svg'
        locs_path = f'{root}/{NetworkBurstingFolder}/locs.svg'
        conn_mat_path = f'{root}/{NetworkBurstingFolder}/{batch_key}_conn_matrix.svg'
        sample_trace_path_E = f'{root}/{NetworkBurstingFolder}/{batch_key}_sample_trace_E.svg'
        sample_trace_path_I = f'{root}/{NetworkBurstingFolder}/{batch_key}_sample_trace_I.svg'   

        # Try Loading Images if they Already Exist
        try:
            assert fresh_figs == False
            assert os.path.exists(raster_plot_path)
            assert os.path.exists(network_activity_path)
            assert os.path.exists(TWODnet_path)
            assert os.path.exists(locs_path)
            assert os.path.exists(conn_mat_path)
            assert os.path.exists(sample_trace_path_E)
            assert os.path.exists(sample_trace_path_I)

            # raster_plot = mpimg.imread(raster_plot_path)
            # network_activity = mpimg.imread(network_activity_path)
            # TWODnet = mpimg.imread(TWODnet_path)
            # locs = mpimg.imread(locs_path)
            # conn_mat = mpimg.imread(conn_mat_path)
            # # Confirm that figure aspect ratios are consistent
            # assert raster_width / raster_height == raster_plot.shape[0] / raster_plot.shape[1]
            # assert network_activity_width / network_activity_height == network_activity.shape[0] / network_activity.shape[1]
            # assert locations_width / locations_height == TWODnet.shape[0] / TWODnet.shape[1]
            # assert locations_width / locations_height == locs.shape[0] / locs.shape[1]
            # assert raster_width / raster_height == conn_mat.shape[0] / conn_mat.shape[1]
        except:
            # if any exception, regenerate the images
            print(f'Error loading existing images at: {NetBurst_path}')
            pass       
        
        ## Data
        #load the simulation data     
        sim.loadAll(file_path)
        
        #Attempt to generate sample trace for an excititory example neuron
        try:
            try:
                assert fresh_figs == False
                #temp hack
                assert fresh_figs == True
                assert os.path.exists(sample_trace_path_E)  
                # sample_trace_E = mpimg.imread(sample_trace_path_E)
                # assert sample_trace_width / sample_trace_height == sample_trace_E.shape[1] / sample_trace_E.shape[0]
            except:
                # Prepare the sample trace
                sample_trace_E = sim.analysis.plotTraces(
                    include = [('E', 0), ('I', 0)], 
                    overlay=True, 
                    oneFigPer='trace', 
                    #timeRange=[0, 2000], 
                    saveFig=sample_trace_path_E, 
                    showFig=False, 
                    figSize=(sample_trace_width, sample_trace_height)
                )
        except:
            print(f'Error generating sample trace plot from Data at: {file_path}')
            sample_trace_path_E = None
            pass
        
        # #Attempt to generate sample trace for an inhib example neuron
        # try:
        #     try:
        #         assert fresh_figs == False
        #         assert os.path.exists(sample_trace_path_I)
        #         # sample_trace_I = mpimg.imread(sample_trace_path_I)
        #         # assert sample_trace_width / sample_trace_height == sample_trace_I.shape[1] / sample_trace_I.shape[0]
        #     except:
        #         # Prepare the sample trace
        #         sample_trace_I = sim.analysis.plotTraces(
        #             include = [('I', 0)], 
        #             #overlay=True, 
        #             oneFigPer='trace', 
        #             timeRange=[0, 1000], 
        #             saveFig=sample_trace_path_I, 
        #             showFig=False, 
        #             figSize=(sample_trace_width, sample_trace_height)
        #         )
        # except:
        #     print(f'Error generating sample trace plot from Data at: {file_path}')
        #     sample_trace_path_I = None
        #     pass

        # Attempt to generate the raster plot
        try:
            try: 
                assert fresh_figs == False
                assert os.path.exists(raster_plot_path) 
                # raster_plot = mpimg.imread(raster_plot_path)
                # assert raster_width / raster_height == raster_plot.shape[1] / raster_plot.shape[0]
            except: sim.analysis.plotRaster(saveFig=raster_plot_path, showFig=False, figSize=(raster_width, raster_height))#, dpi=600)
        except:
            print(f'Error generating raster plot from Data at: {file_path}')
            raster_plot_path = None
            pass
        
        # Attempt to generate the neuron locs plot
        try:
            try: 
                assert fresh_figs == False
                assert os.path.exists(locs_path)
                # locs = mpimg.imread(locs_path)
                # assert locations_width / locations_height == locs.shape[1] / locs.shape[0]
            except: sim.analysis.plot2Dnet(saveFig=locs_path, showFig=False, showConns=False, figSize=(locations_width, locations_height))
        except: 
            print(f'Error generating neuron locs plot from Data at: {file_path}')
            locs_path = None
            pass

        # Attempt to generate the network connections plot
        try:
            try: 
                assert fresh_figs == False
                assert os.path.exists(TWODnet_path)
                # locs = mpimg.imread(TWODnet_path)
                # assert locations_width / locations_height == TWODnet.shape[1] / TWODnet.shape[0]
            except: sim.analysis.plot2Dnet(saveFig=TWODnet_path, showFig=False, showConns=True, figSize=(locations_width, locations_height))
        except: 
            print(f'Error generating 2Dnet_conns from Data at: {file_path}')
            TWODnet_path = None
            pass

        # Attempt to generate the connectivity matrix plot
        try:
            try: 
                assert fresh_figs == False
                assert os.path.exists(conn_mat_path)
                # locs = mpimg.imread(conn_mat_path)
                # assert conn_mat_width / conn_mat_height == conn_mat.shape[0] / conn_mat.shape[1]
            except: sim.analysis.plotConn(saveFig=conn_mat_path, showFig=False, figSize=(raster_width, raster_height))
        except: 
            print(f'Error generating connectivity matrix from Data at: {file_path}')
            conn_mat_path = None
            pass


        # Attempt to generate the network activity plot
        try:
            try:
                assert fresh_figs == False
                assert os.path.exists(network_activity_path)
                # network_activity = mpimg.imread(network_activity_path)
                # assert network_activity_width / network_activity_height == network_activity.shape[1] / network_activity.shape[0]
            except:
                #prepare raster data
                rasterData = sim.analysis.prepareRaster()
                # Generate the network activity plot with a size of (10, 5)
                plot_network_activity(
                    rasterData, 
                    binSize=binSize, 
                    gaussianSigma=gaussianSigma, 
                    thresholdBurst=thresholdBurst, 
                    figSize=(network_activity_width, network_activity_height), 
                    saveFig=network_activity_path
                )
        except:
            print(f'Error generating network activity plot from Data at: {file_path}')
            network_activity_path = None
            pass

        return raster_plot_path, sample_trace_path_I, sample_trace_path_E, network_activity_path, TWODnet_path, locs_path, conn_mat_path, NetBurst_path 

    def generate_net_activity_raster(raster_plot_path, network_activity_path, NetBurst_path, batchrun_folder, batch_key):
        
        net_activity_raster_path = f'{NetBurst_path}/{batch_key}_net_activity_raster.svg'
        # height = raster_height + network_activity_height
        # width = raster_width      
        try: 
            assert fresh_figs == False
            #temp hack
            assert fresh_figs == True
            # net_activity_raster = mpimg.imread(net_activity_raster_path)
            # assert height / width == net_activity_raster.shape[1] / net_activity_raster.shape[0]
            assert os.path.exists(net_activity_raster_path)
            return net_activity_raster_path
        except:            
            try: 
                # Load the images
                # Load the SVG files
                raster_plot = sg.fromfile(raster_plot_path)
                network_activity = sg.fromfile(network_activity_path)
                # raster_plot = mpimg.imread(raster_plot_path)
                # network_activity = mpimg.imread(network_activity_path)
                #TWODnet = mpimg.imread(TWODnet_path)
                
                # Create a new figure
                # Create a new SVG figure
                #fig = sg.SVGFigure(width, height)

                # Get the root of each SVG file
                plot1 = raster_plot.getroot()
                plot2 = network_activity.getroot()
                #plot1.scale(1.125)
                
                # Get the size of the first SVG
                svg1_width = float(raster_plot.width.rstrip('pt'))
                svg1_height = float(raster_plot.height.rstrip('pt'))

                # Get the size of the second SVG
                svg2_width = float(network_activity.width.rstrip('pt'))
                svg2_height = float(network_activity.height.rstrip('pt'))

                # Scale the second SVG to match the width of the first SVG
                # scale image widths so that axes line up
                scale_factor = svg1_width / svg2_width
                plot2.scale(scale_factor)
                #plot2.scale(.75, 1)

                #plot2.scale(scale_factor)

                # Move the second plot to the right by the width of the first plot plus a small margin
                plot2.moveto(0, svg1_height + 10)  # Adjust the margin as needed

                # Create a new SVG figure with the exact size needed to accommodate the figures
                fig_width = max(svg1_width, svg2_width*scale_factor)
                fig_height = svg1_height + svg2_height*scale_factor + 10  # Add the margin
                # #move both plots down a little to make space for title.
                # plot1.moveto(0, 10)
                # plot2.moveto(0, 10)  # Adjust the margin as needed
                fig = sg.SVGFigure(fig_width, fig_height)

                # Set the width and height attributes of the SVGFigure
                fig.set_size((str(fig._width), str(fig._height)))

                # Add the plots to the figure
                fig.append([plot1, plot2])

                # Adjust layout to fit subplots tightly
                #plt.tight_layout()
                #fig.suptitle(f'{batchrun_folder}{batch_key}', fontsize=16)
                # Create a title
                #title = f'{batchrun_folder}{batch_key}'
                # Calculate the center of the figure
                #center_x = svg1_width / 2

                # Create a title
                #title = f'{batchrun_folder}{batch_key}'
                #title_text = sg.TextElement(center_x, 20, title, size=10, weight="bold", anchor="middle")  # Adjust the position and style as needed

                # Add the title to the figure
                #fig.append(title_text)
                #net_activity_raster_path = f'{NetBurst_path}/{batch_key}_net_activity_raster.png'      
                # plt.savefig(net_activity_raster_path, dpi=600, bbox_inches='tight')
                # plt.savefig(net_activity_raster_path.replace('.png', '.svg'), bbox_inches='tight')
                #plt.savefig(net_activity_raster_path, bbox_inches='tight')
                fig.save(net_activity_raster_path)
                return net_activity_raster_path

            except:
                print(f'Error generating net_activity_raster figure at: {NetBurst_path}')
                return None
            #pass

    def generate_conn_summarry_fig(sample_trace_path_E, conn_mat_path, TWODnet_path, NetBurst_path, batchrun_folder, batch_key):
    
        conn_summary_fig_path = f'{NetBurst_path}/{batch_key}_conn_summary_fig.svg' 
        height = raster_height + locations_height
        width = locations_width      
        try: 
            assert fresh_figs == False
            assert os.path.exists(conn_summary_fig_path)
            return conn_summary_fig_path
        except:            
            try: 
                # Load the SVG files
                conn_mat = sg.fromfile(conn_mat_path)
                TWODnet = sg.fromfile(TWODnet_path)
                activity_trace = sg.fromfile(sample_trace_path_E)

                # Get the root of each SVG file
                plot1 = TWODnet.getroot()
                plot2 = conn_mat.getroot()
                plot3 = activity_trace.getroot()

                # Measure the height and width of the SVGs
                svg1_height = float(TWODnet.height.rstrip('pt'))
                svg1_width = float(TWODnet.width.rstrip('pt'))
                svg2_height = float(conn_mat.height.rstrip('pt'))
                svg2_width = float(conn_mat.width.rstrip('pt'))
                svg3_height = float(activity_trace.height.rstrip('pt'))
                svg3_width = float(activity_trace.width.rstrip('pt'))

                # Calculate the scale factors
                scale_factor_height = max(svg2_height, svg3_height) / svg1_height
                scale_factor_width = (svg2_width + svg3_width + 5) / svg1_width  # 5 is the space between the plots

                # Scale the top plot
                plot1.scale(scale_factor_width, scale_factor_height)

                # Move the second plot down by the height of the first plot plus a small margin
                plot2.moveto(0, svg1_height * scale_factor_height + 10)  # Adjust the margin as needed

                # Move the third plot down by the height of the first plot plus a small margin, and to the right by the width of the second plot
                plot3.moveto(svg2_width + 5, svg1_height * scale_factor_height + 10)  # Adjust the margin as needed

                # Create a new SVG figure
                fig = sg.SVGFigure(svg1_width * scale_factor_width, svg1_height * scale_factor_height + max(svg2_height, svg3_height) + 10)

                # Set the width and height attributes of the SVGFigure
                fig.set_size((str(fig._width), str(fig._height)))

                # Add the plots to the figure
                fig.append([plot1, plot2, plot3])

                # Save the figure
                fig.save(conn_summary_fig_path)
                return conn_summary_fig_path

            except:
                print(f'Error generating conn_summary figure at: {NetBurst_path}')
            return None
            #pass

    def generate_param_summary_fig(conn_summary_path, net_activity_raster_path, NetBurst_path, batchrun_folder, batch_key):
        param_summary_fig_path = f'{NetBurst_path}/{batch_key}_param_summary_fig.svg'
        width  = raster_width + locations_width
        height = conn_mat_height + locations_height 
        try: 
            assert fresh_figs == False
            assert fresh_figs == True
            assert os.path.exists(param_summary_fig_path)
            return param_summary_fig_path
        except:            
            try: 
                # Load the SVG files
                conn_summary = sg.fromfile(conn_summary_path)
                net_activity_raster = sg.fromfile(net_activity_raster_path)

                # Get the root of each SVG file
                plot1 = conn_summary.getroot()
                plot2 = net_activity_raster.getroot()

                # Set the width and height of the first SVG
                # svg1_width = 1000.0  # Replace with your desired width
                # svg1_height = 500.0  # Replace with your desired height
                # conn_summary.set_size((str(svg1_width), str(svg1_height)))
                svg1_width = float(conn_summary.width.rstrip('pt')) 
                svg1_height = float(conn_summary.height.rstrip('pt'))

                # Set the width and height of the second SVG
                # svg2_width = svg1_width / 2  # Half the width of the first SVG
                # svg2_height = svg1_height  # Same height as the first SVG
                # net_activity_raster.set_size((str(svg2_width), str(svg2_height)))
                svg2_width = float(net_activity_raster.width.rstrip('pt'))
                svg2_height = float(net_activity_raster.height.rstrip('pt'))

                # Scale the second SVG to equal height                
                
                # Move the second plot to the right by the width of the first plot plus a small margin
                plot2.moveto(svg1_width+10, 0)  # Adjust the margin as needed

                # Create a new SVG figure
                #fig = sg.SVGFigure(width, height)
                fig = sg.SVGFigure(svg1_width + svg2_width + 5, svg1_height)

                # Set the width and height attributes of the SVGFigure
                fig.set_size((str(fig._width), str(fig._height)))

                # Add the plots to the figure
                fig.append([plot1, plot2])

                # Create a title
                #title = f'{batchrun_folder}{batch_key}'
                #title_text = sg.TextElement(5, 20, title, size=12, weight="bold")  # Adjust the position and style as needed

                # Add the title to the figure
                #fig.append(title_text)

                # Save the figure
                fig.save(param_summary_fig_path)
                return param_summary_fig_path

            except:
                print(f'Error generating param_summary figure at: {NetBurst_path}')
                return None

    #Simulation plots will be saved in the batch folder as png files
    #Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.dirname(script_dir)
    output_path = f'{output_path}/output' 
    batch_path = f'{output_path}/{batchLabel}'    
    
    #for each file named *_cfg.json in the batch folder, get raster data
    #for file in os.listdir(batch_path):
    #walk through the batch folder and subfolders
    for root, dirs, files in os.walk(batch_path):
        for file in files:
            if file.endswith("_data.json"):
                    
                #get file info
                file_path, batchrun_folder, batch_key = get_batchrun_info(root, file)
                raster_plot_path, sample_trace_path_I, sample_trace_path_E, network_activity_path, TWODnet_path, locs_path, conn_mat_path, NetBurst_path = generate_primary_figures(
                    root, file,  size = size, 
                    net_activity_params = net_activity_params)
                net_activity_raster_path = generate_net_activity_raster(
                    raster_plot_path, network_activity_path, NetBurst_path, batchrun_folder, batch_key) 
                # trace_fig_path = generate_trace_fig(
                #     sample_trace_path_E, sample_trace_path_I, NetBurst_path, batchrun_folder, batch_key)
                conn_summary_fig_path = generate_conn_summarry_fig(
                    #trace_fig_path, 
                    sample_trace_path_E,
                    conn_mat_path, TWODnet_path, NetBurst_path, batchrun_folder, batch_key)
                param_summary_fig_path = generate_param_summary_fig(
                    conn_summary_fig_path, net_activity_raster_path, NetBurst_path, batchrun_folder, batch_key)
               
                
                #combined_figure_path = f'{NetBurst_path}/{batch_key}_combined_figure.png'  # replace with your actual path