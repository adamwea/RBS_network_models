#from BurstingPlotDevelopment import plot_network_activity
import os
import json
from plot_network_activity import plot_network_activity
from netpyne import sim
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import scipy
from scipy.ndimage import zoom

def plot_combined_fig(binSize = 2, gaussianSigma = 13, batchLabel = 'batch', batch_path = None):
    #simulation plots will be saved in the batch folder as png files
    #location: nb2/batch_run_development/aw_grid_*batchLabel*/*_data.json
    #batchLabel = 'aw_grid'
    print(os.getcwd())
    batch_path  = os.getcwd() + '/' + batchLabel
    print(f'Saving to path: {batch_path}')
    

    #binSize=0.02*100
    #gaussianSigma=0.13*100
    
    #for each file named *_cfg.json in the batch folder, get raster data
    for file in os.listdir(batch_path):
        if file.endswith("_data.json"):
            #print(file) 
            sim.loadAll(batch_path+'/'+file)
            batch_key = file.split('grid')[1].split('data')[0]
            rasterData = sim.analysis.prepareRaster()

            #root = f'/home/adam/workspace/git_workspace/netpyne/hdmea_simulations/BurstingPlotDevelopment/nb2/batch_run_development/{batchLabel}'
            NetworkBurstingFolder = 'NetworkBurst_and_Raster_Figs'
            if not os.path.exists(f'{root}/{NetworkBurstingFolder}'):
                os.makedirs(f'{root}/{NetworkBurstingFolder}')
            combined_figure_path = f'{root}/NetworkBurst_and_Raster_Figs/{batch_key}combined_figure.png'  # replace with your actual path
                
            # Define the paths to the raster plot and network activity images
            raster_plot_path = batch_path +'/'+'aw_grid'+batch_key+'raster.png'
            network_activity_path = batch_path +'/'+batch_key+'_network_activity_def.png'
            TWODnet_path = batch_path +'/'+'aw_grid'+batch_key+'plot_2Dnet.png'     
            
        # Define the sizes for the plots
            raster_height = 6
            raster_width = 6
            network_activity_height = raster_height / 3
            network_activity_width = raster_width
            locations_height = raster_height + network_activity_height
            locations_width = locations_height*2

            # Generate the raster plot with a size of (10, 10)
            rasterPlot = sim.analysis.plotRaster(saveFig=raster_plot_path, showFig=False, figSize=(raster_width, raster_height))

            # Generate the 2Dnet plot with a size of (15, 20)
            locations_plot = sim.analysis.plot2Dnet(saveFig=TWODnet_path, showFig=False, figSize=(locations_width, locations_height))

            # Generate the network activity plot with a size of (10, 5)
            plot_network_activity(
                rasterData, 
                binSize=binSize, 
                gaussianSigma=gaussianSigma, 
                thresholdBurst=1.2, 
                figSize=(network_activity_width, network_activity_height), 
                saveFig=network_activity_path
            )        

            # Load the images
            raster_plot = mpimg.imread(raster_plot_path)
            network_activity = mpimg.imread(network_activity_path)
            TWODnet = mpimg.imread(TWODnet_path)
            
            # Create a new figure
            fig = plt.figure(figsize=(14, 6))

            # Add the 2Dnet plot to the left part of the figure
            ax1 = plt.subplot2grid((4, 3), (0, 0), rowspan=4, colspan=2)
            ax1.imshow(TWODnet)
            ax1.axis('off')  # Hide axes

            # Add the raster plot to the top right part of the figure
            ax2 = plt.subplot2grid((4, 3), (0, 2), rowspan=3)
            ax2.imshow(raster_plot)
            ax2.axis('off')  # Hide axes

            # Add the network activity plot to the bottom right part of the figure
            ax3 = plt.subplot2grid((4, 3), (3, 2))
            ax3.imshow(network_activity)
            ax3.axis('off')  # Hide axes

            # Adjust layout to fit subplots tightly
            plt.tight_layout()
            
            # Save the combined figure in the batch folder
            # batch key as fig title
            fig.suptitle(batch_key, fontsize=16)      
            plt.savefig(combined_figure_path)