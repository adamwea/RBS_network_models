import os
import netpyne
import subprocess
from fitness_functions import fitnessFunc
from fitness_config import *
kwargs = fitnessFuncArgs
#from USER_INPUTS import *
import pandas as pd
import json

# Initialize an empty DataFrame
df = pd.DataFrame()

import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image
from json2table import convert

# surpress all print statements
import sys, os
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# enable all print statements
def enablePrint():
    sys.stdout = sys.__stdout__

def generate_pdf_page(data_file_path, elite_paths_cull, gen_rank):
    import matplotlib.pyplot as plt

    def plot_params(cfg_data, params, cgf_file_path):
        
        def plot_each_cfg(cfg_data, color = 'k', markersize = 10, markerfacecolor = 'none'):
            # Find common keys between cfg_data and params, create a dictionary with cfg_data values, and corresponding ranges in params
            param_dict = {}
            for key in cfg_data.keys():
                if key in params.keys():
                    param_dict[key] = [cfg_data[key], params[key]]

            # Normalize values and ranges to 0-1
            for key, value in param_dict.items():
                #print(key)
                # if 'tau' in key: 
                #     pass
                try: assert len(value[1]) == 2, "Parameter without a range. Skipping."
                except: continue
                val = value.copy()
                # Normalize value
                val[0] = (val[0] - val[1][0])/(val[1][1] - val[1][0])
                # Normalize range
                val[1] = [0, 1]
                assert 0 <= val[0] <= 1, f"Error: {key} has an invalid value."
                param_dict[key] = val

            #any values where len(value[1]) != 2 is a constant value, put at top of list
            # Separate items with len(value[1]) != 2 and len(value[1]) == 2
            constant_items = {key: value for key, value in param_dict.items() if isinstance(value[1], int) or isinstance(value[1], float)}
            variable_items = {key: value for key, value in param_dict.items() if isinstance(value[1], list) and len(value[1]) == 2}
            #this should be none
            erroneous_items = {key: value for key, value in param_dict.items() if isinstance(value[1], list) and len(value[1]) != 2}
            if len(erroneous_items) > 0: raise Exception(f"Error: {erroneous_items} has an invalid range.")
            # Combine the dictionaries, with constant_items first
            param_dict = {**constant_items, **variable_items}            
            
            # Create a plot where on the y-axis you have the names of each key, and x-axis is the normalized value. 
            # Each line should show a dot at the normalized value

            # Plot each parameter
            for i, (key, value) in enumerate(param_dict.items()):
                try: assert len(value[1]) == 2, "Parameter without a range. Solid Red Line."
                except:
                    ax.hlines(i, 0, 1, colors='r')  # Plot a red line for the normalized value 
                    continue
                #ax.plot(value[0], i, f'{color}o')  # Plot a {color} dot at the normalized value
                ax.plot(value[0], i, marker='o', color=color, markersize=markersize, markerfacecolor=markerfacecolor)  # Plot an empty {color} circle at the normalized value                #dotted line for range, black
                ax.plot(value[1], [i, i], 'k--')  # Plot a red line for the normalized range
                #ax.hlines(i, *value[1], colors='r')  # Plot a red line for the normalized range

            # Set the y-axis labels to be the parameter names
            ax.set_yticks(range(len(param_dict)))
            ax.set_yticklabels(param_dict.keys())

        # Create figure and axis
        fig, ax = plt.subplots()
        #set size of plot
        fig.set_size_inches(10, 10)

        # plot the cfg_data of interest
        main_cfg_data = cfg_data.copy()
        main_cfg_path = cgf_file_path     

        #plot the rest for comparison
        #gen_dir = os.path.dirname(os.path.dirname(cgf_file_path))
        
        #os walk through gen_dir and find all .json files with containing cfg in the name
        #for each fitness_save_path, in elite_paths_cull, plot the cfg_data
        #for i in range(len(elite_paths_cull)):
            #for root, dirs, files in os.walk(elite_paths_cull[i][1]['fitness_save_path']):
        gen_path = os.path.dirname(cgf_file_path)
        for root, dirs, files in os.walk(gen_path):
            for file in files:
                if '.json' in file and 'cfg' in file:
                    #create corresponding data file path
                    cfg_file_path = os.path.join(root, file)
                    if cfg_file_path in elite_paths_cull: continue
                    #print(cfg_file_path)
                    if cfg_file_path == main_cfg_path: continue
                    #check if data file_path exists
                    if os.path.exists(cfg_file_path): pass
                    else: continue
                    #load the data file using netpyne loadall
                    cfg_data = json.load(open(cfg_file_path))
                    cfg_data = cfg_data['simConfig']
                    # plot the cfg_data of interest
                    plot_each_cfg(cfg_data, color = 'k', markersize = 8, markerfacecolor = 'none')

        #plot main cfg_data last, so it is on top
        plot_each_cfg(main_cfg_data, color = 'r', markersize = 10, markerfacecolor = 'red')        
        
        # Set the x-axis label
        ax.set_xlabel('Normalized Param Value')
        #tight layout
        plt.tight_layout()
        # Show the plot
        # plt.show()
        #plt.savefig(os.path.join(os.path.dirname(cgf_file_path), 'params_plot.png'))
        return plt
       
    if '.archive' in data_file_path:
        cand_plot_path = data_file_path.replace('/output/.archive/', '/plots/')
        cand_plot_path = os.path.dirname(cand_plot_path)
    else:
        cand_plot_path = data_file_path.replace('/output/', '/plots/')
        cand_plot_path = os.path.dirname(cand_plot_path)
    simLabel = os.path.basename(data_file_path).split('_data')[0]
    #dd underscore to end of simLabel for unique file matching
    simLabel_ = simLabel + '_'
    # Create a new PDF
    from reportlab.lib.pagesizes import landscape, letter

    # Create a new PDF with landscape orientation
    c = canvas.Canvas(os.path.join(cand_plot_path, f"{simLabel}.pdf"), pagesize=landscape(letter))

    # Get the page width and height
    page_width, page_height = landscape(letter)

    # Define the width and height for the images
    # Assuming the images have an aspect ratio of 1:1
    
    try: 
        files = [f for f in os.listdir(cand_plot_path) if simLabel_ in f and f.endswith('.png') and '_params_plot.png' not in f]
        img_width = page_width/len(files)
        #quick fix for less than 3 plots
        if len(files) < 3: img_width = page_width/3
        img_height = img_width
    except: 
        raise Exception('No .png files found in plots directory')
    

    # Walk through the plots directory
    j=0
    for root, dirs, files in os.walk(cand_plot_path):
        # Sort the files to ensure they're added in the correct order
        files.sort()
        for i, file in enumerate(files):
            if simLabel_ in file and file.endswith('.png') and '_params_plot.png' not in file:                
                # Open the image file
                img = Image.open(os.path.join(root, file))
                # Convert the image to RGB (required by reportlab)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Save the image to the PDF, adjusting the position for each image
                c.drawInlineImage(img, j * img_width, 0, width=img_width, height=img_height)
                #use for plotting params later
                final_x_pos = j * img_width
                j += 1
        # # Start a new page after adding all images on the current page
        # c.showPage()

    fitness_path = data_file_path.replace('_data.json', '_Fitness.json')
    fitness_data = json.load(open(fitness_path))
    # Convert the JSON data to an HTML table
    html_table = convert(fitness_data, build_direction="LEFT_TO_RIGHT")
    #add table to pdf. top left corner is 0,0
    # c.drawString(0, 0, 'Fitness Data')
    # c.drawString(0, 0, html_table)
    # Add each item in the JSON data as a new line    
    # Add each item in the JSON data as a new line
        # Get the page width and height
    #set font for all prints smaller than default
    c.setFont("Helvetica", 8)
    
    #page_width, page_height = letter
    y_position = page_height - 15  # Start near the top of the page
    #split data_file_path at 2DNetworkSimulations
    data_file_path_print = data_file_path.split('2DNetworkSimulations')[1]
    data_directory_str = f"Data Directory: {data_file_path_print}"
    c.drawString(10, y_position, data_directory_str)
    y_position -= 10  # Move down for the next line
    generation_str = f"Generation: {os.path.basename(os.path.dirname(data_file_path))}"
    c.drawString(10, y_position, generation_str)
    y_position -= 10  # Move down for the next line
    generation_rank_str = f"Generation Rank: {gen_rank}/{len(elite_paths_cull)}"
    c.drawString(10, y_position, generation_rank_str)
    y_position -= 10  # Move down for the next line
    #y_position -= 15  # Move down for the next line
    
    #import dictionary pops from fitness_config.py and covert print the same way as fitness_data
    from fitness_config import pops
    y_position = y_position - 15  # Start near the top of the page
    for key, value in pops.items():
        c.drawString(10, y_position, f"{key}: {value}")
        y_position -= 10  # Move down for the next line
    
    y_position = y_position - 15  # Start near the top of the page
    for key, value in fitness_data.items():
        c.drawString(10, y_position, f"{key}: {value}")
        y_position -= 10  # Move down for the next line

    #plot params
    from USER_evol_param_space import params
    cgf_file_path = data_file_path.replace('_data.json', '_cfg.json')
    cfg_data = json.load(open(cgf_file_path))
    cfg_data = cfg_data['simConfig']
    plt = plot_params(cfg_data, params, cgf_file_path)
    params_plot_path = os.path.join(cand_plot_path, f'{simLabel}_params_plot.png')
    plt.savefig(params_plot_path)
    img = Image.open(params_plot_path)
     # Convert the image to RGB (required by reportlab)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Save the image to the PDF, adjusting the position for each image
    c.drawInlineImage(img, final_x_pos, img_height, width=img_width, height=img_height)

    #print info in top right corner
    #print data_file_path
    #split data_file_path at 2DNetworkSimulations
    #from reportlab.lib.pagesizes import letter
    #from reportlab.pdfgen import canvas
    #from reportlab.lib.units import inch

    # Create a new canvas object with the page size of a letter
    #c = canvas.Canvas("example.pdf", pagesize=letter)


    # Save the PDF
    c.save()

#surpress all print statements
#blockPrint()

def plot_elites(job_dir):
    #job_dir = os.path.abspath(job_dir)
    gen_dirs = [f.path for f in os.scandir(job_dir) if f.is_dir() and 'gen' in f.name]

    for gen_dir in gen_dirs:
        elite_paths = {}
        for root, dirs, files in os.walk(gen_dir):
            for file in files:
                #if '.json' in file and 'Fitness' in file:
                if '.json' in file and '_data' in file:
                    #create corresponding data file path
                    fit_file_path = os.path.join(root, file.replace('_data', '_Fitness'))
                    data_file_path = os.path.join(root, file)

                    #temp
                    if '.archive' in data_file_path: continue
                    
                    #check if data file_path exists
                    if os.path.exists(data_file_path): pass
                    else: continue

                    try:
                        #load the data file using netpyne loadall
                        netpyne.sim.loadAll(data_file_path)
                        simData = netpyne.sim.allSimData
                        batch_saveFolder = netpyne.sim.cfg.saveFolder
                        #get cwd
                        # cwd = os.getcwd()
                        # cwd_basename = os.path.basename(cwd)
                        #get path to this script
                        NERSC_path = os.path.dirname(os.path.realpath(__file__))
                        repo_path = os.path.dirname(NERSC_path)
                        cwd = repo_path
                        cwd_basename = os.path.basename(cwd)
                        #make bach_saveFolder relative after basename in batch_saveFolder
                        batch_saveFolder = f'{cwd}{batch_saveFolder.split(cwd_basename)[1]}'
                        #batch_saveFolder = batch_saveFolder[1:] #remove leading '/'
                        simLabel = netpyne.sim.cfg.simLabel
                        fitness_save_path = os.path.dirname(data_file_path)
                        avgScaledFitness = fitnessFunc(
                            simData, sim_obj = None, plot = False, simLabel = simLabel, 
                            data_file_path = data_file_path, batch_saveFolder = batch_saveFolder, 
                            fitness_save_path = None, **kwargs)
                        elite_paths[simLabel] = {'avgScaledFitness': avgScaledFitness, 
                                                 'data_file_path': data_file_path,
                                                 'batch_saveFolder': batch_saveFolder,
                                                 'fitness_save_path': fitness_save_path,
                                                 'simData': simData,}
                    except: pass
        #elite_rate = 0.10
        #get path in job_dir ending in _batch.json and load
        batch_file_path = [f.path for f in os.scandir(job_dir) if f.is_file() and '_batch.json' in f.name][0]
        batch_data = json.load(open(batch_file_path))
        num_elites = batch_data['batch']['evolCfg']['num_elites']
        assert num_elites < len(elite_paths), f"Error: num_elites must be less than the number of elites in the generation."
        elite_paths = sorted(elite_paths.items(), key=lambda x: x[1]['avgScaledFitness'], reverse=False)
        elite_paths_cull = elite_paths[:num_elites]
        for simLabel, data in elite_paths_cull:
            #rank is the index of the elite in the elite_paths list
            gen_rank = elite_paths.index((simLabel, data))
            
            #initialize variables
            data_file_path = data['data_file_path']
            batch_saveFolder = data['batch_saveFolder']
            fitness_save_path = data['fitness_save_path']
            simData = data['simData']
        
            #plot
            avgScaledFitness = fitnessFunc(
            simData, sim_obj = netpyne.sim, plot = True, simLabel = simLabel, 
            data_file_path = data_file_path, batch_saveFolder = batch_saveFolder, 
            fitness_save_path = fitness_save_path, **kwargs)
            print('Plots saved to: ...')

            #Network plot, Raster plot, and Trace .pngs should have been generated to plots for each candidate in outputs.
            # Get each PNG, line them up in a row, and save them to a single PDF.
            # os walk through plots folder and generate PDF for each candidate
            try: generate_pdf_page(data_file_path, elite_paths_cull, gen_rank)
            except: pass
                    
if __name__ == '__main__':
    job_dir = '/home/adamm/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240426_Run12_26AprSAFE_1x100'
    plot_elites(job_dir)
            
            