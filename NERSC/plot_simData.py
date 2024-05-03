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
import copy

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# enable all print statements
def enablePrint():
    sys.stdout = sys.__stdout__

class SimCopy:
    def __init__(self, sim):
        for attr in dir(sim):
            try:
                if not attr.startswith('__'):  # Exclude built-in attributes
                    value = getattr(sim, attr)
                    #if isinstance(value, dict):
                    setattr(self, attr, copy.deepcopy(value))
            except Exception:
                pass

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
        elite_cfg_paths = [f[1]['data_file_path'].replace('_data', '_cfg') for f in elite_paths_cull]
        for root, dirs, files in os.walk(gen_path):
            for file in files:
                if '.json' in file and 'cfg' in file:
                    #create corresponding data file path
                    cfg_file_path = os.path.join(root, file)
                    if cfg_file_path not in elite_cfg_paths: continue
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
        files = [f for f in os.listdir(cand_plot_path) 
                 if simLabel_ in f and f.endswith('.png') 
                 and '_params_plot.png' not in f
                 and not 'E0' in f
                 and not 'I0' in f
                 and not '_sample_trace.png' in f
                ]
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
                if 'E0' in file or 'I0' in file: continue
                if '_sample_trace.png' in file: continue
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

    '''
    Prints the JSON data to PDF
    '''
    fitness_path = data_file_path.replace('_data.json', '_Fitness.json')
    fitness_data = json.load(open(fitness_path))
    c.setFont("Helvetica", 8)
    
    ## Generation-Simualtion info
    #page_width, page_height = letter
    y_position = page_height - 15  # Start near the top of the page
    #split data_file_path at 2DNetworkSimulations
    data_file_path_print = data_file_path.split('2DNetworkSimulations')[1]
    data_directory_str = f"Data Directory: {data_file_path_print}"
    c.drawString(10, y_position, data_directory_str)
    y_position -= 10  # Move down for the next line
    generation_str = f"SimLabel: {simLabel}"
    c.drawString(10, y_position, generation_str)
    y_position -= 10  # Move down for the next line
    generation_rank_str = f"Generation Rank: {gen_rank}/{len(elite_paths_cull)}"
    c.drawString(10, y_position, generation_rank_str)
    y_position -= 10  # Move down for the next line
    
    ## Fitness Data
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

    ## Network Convolution Params
    from USER_INPUTS import USER_raster_convolve_params
    y_position = y_position - 15  # Start near the top of the page
    for key, value in USER_raster_convolve_params.items():
        c.drawString(10, y_position, f"{key}: {value}")
        y_position -= 10  # Move down for the next line

    # Close the PDF
    c.save()

#surpress all print statements
blockPrint()

def plot_elites(job_dir):
    #job_dir = os.path.abspath(job_dir)
    gen_dirs = [f.path for f in os.scandir(job_dir) if f.is_dir() and 'gen' in f.name]

    #sort gendirs numerically such that, gen_9 comes before gen_10
    gen_dirs = sorted(gen_dirs, key=lambda x: int(x.split('_')[-1]))
    

    for gen_dir in gen_dirs:
        
        #skip if gen_dir is less than start_gen
        if start_gen is not None:
            if int(gen_dir.split('_')[-1]) < start_gen: 
                enablePrint()
                print(f"Skipping {os.path.basename(gen_dir)}")
                blockPrint()
                continue

        #print gen_dir bold and green
        enablePrint()
        print(f"\033[1;32m{gen_dir}\033[0m")
        blockPrint()
        if verbose == True: enablePrint()
            
        
        elite_paths = {}
        enablePrint()    
        print(f"Collecting Elites...")
        blockPrint()
        if verbose == True: enablePrint()
        for root, dirs, files in os.walk(gen_dir):
            for file in files:
                #if '.json' in file and 'Fitness' in file:
                if '.json' in file and '_data' in file:

                    #create corresponding data file path
                    fit_file_path = os.path.join(root, file.replace('_data', '_Fitness'))
                    data_file_path = os.path.join(root, file)
                    
                    #skip if gen_dir is less than start_gen
                    if cand is not None:
                        if int(data_file_path.split('_')[-2]) != cand: 
                            enablePrint()
                            print(f"Skipping {os.path.basename(data_file_path)}")
                            blockPrint()
                            continue

                    # enablePrint()    
                    # print(data_file_path)
                    # blockPrint()
                    # if verbose == True: enablePrint()

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
                        simLabel = netpyne.sim.cfg.simLabel

                        #pathing
                        NERSC_path = os.path.dirname(os.path.realpath(__file__))
                        repo_path = os.path.dirname(NERSC_path)
                        cwd = repo_path
                        cwd_basename = os.path.basename(cwd)
                        batch_saveFolder = f'{cwd}{batch_saveFolder.split(cwd_basename)[1]}'                        
                        fitness_save_path = os.path.dirname(data_file_path)

                        #measure fitness
                        avgScaledFitness = fitnessFunc(
                            simData, plot = False, simLabel = simLabel, 
                            data_file_path = data_file_path, batch_saveFolder = batch_saveFolder, 
                            fitness_save_path = fitness_save_path, **kwargs)
                        elite_paths[simLabel] = {'avgScaledFitness': avgScaledFitness, 
                                                 'data_file_path': data_file_path,
                                                 'batch_saveFolder': batch_saveFolder,
                                                 'fitness_save_path': fitness_save_path,
                                                 'simData': simData,
                                                 }
                    except: pass

        #elite_rate = 0.10
        #get path in job_dir ending in _batch.json and load
        batch_file_path = [f.path for f in os.scandir(job_dir) if f.is_file() and '_batch.json' in f.name][0]
        batch_data = json.load(open(batch_file_path))
        num_elites = batch_data['batch']['evolCfg']['num_elites']
        # try: assert num_elites < len(elite_paths), f"Error: num_elites must be less than the number of elites in the generation."
        # except: 
        #     if cand is not None: pass
        #     else: raise Exception(f"Error: num_elites must be less than the number of elites in the generation.")

        elite_paths = sorted(elite_paths.items(), key=lambda x: x[1]['avgScaledFitness'], reverse=False)
        elite_paths_cull = elite_paths[:num_elites]
        for simLabel, data in elite_paths_cull:
            
            #print simLabel bold and yellow
            enablePrint()
            print(f"\033[1;33m{simLabel}\033[0m, {data['avgScaledFitness']}")

            #skip if already plotted
            if new_plots == False:
                assert 'archive' not in data_file_path, 'Error: Cannot plot from archive.'
                expected_plot_path = os.path.join(data['fitness_save_path'], simLabel + '.pdf').replace('output', 'plots')              
                if os.path.exists(expected_plot_path): 
                    print('Already plotted. Skipping...')
                    continue                
            blockPrint()
            if verbose == True: enablePrint()
            
            #rank is the index of the elite in the elite_paths list
            gen_rank = elite_paths.index((simLabel, data))
            
            #initialize variables
            data_file_path = data['data_file_path']
            batch_saveFolder = data['batch_saveFolder']
            fitness_save_path = data['fitness_save_path']
            simData = data['simData']
        
            #plot
            avgScaledFitness = fitnessFunc(
                simData, plot = True, simLabel = simLabel, 
                data_file_path = data_file_path, batch_saveFolder = batch_saveFolder, 
                fitness_save_path = fitness_save_path, **kwargs)
            print('Plots saved to: ...')

            #Network plot, Raster plot, and Trace .pngs should have been generated to plots for each candidate in outputs.
            # Get each PNG, line them up in a row, and save them to a single PDF.
            # os walk through plots folder and generate PDF for each candidate
            try: generate_pdf_page(data_file_path, elite_paths_cull, gen_rank)
            except: pass
                   
import signal

# Function to handle the timeout
def handler(signum, frame):
    raise Exception("Program took too long to finish")

# Set the signal handler
signal.signal(signal.SIGALRM, handler)

if __name__ == '__main__':
    #set to some value to skip gens less than start_gen
    start_gen = 10
    #start_gen = None
    cand = 70
    cand = None

    #set to True to plot all candidates, regardless of whether they have already been plotted
    new_plots = False

    #set to True to print verbose output
    verbose = False

    #HOF Mode
    HOF_mode = True

    job_dirs = [
        #'/home/adamm/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240426_Run12_26AprSAFE_1x100',
        #'/home/adamm/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240429_Run2_debug_node_run',        
        #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240426_Run12_26AprSAFE_1x100',
        #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240429_Run1_debug_node_run',
        #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240429_Run2_debug_node_run',
        #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240430_Run1_interactive_node_run',
        '/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240430_Run2_debug_node_run',
    ]

    
    for job_dir in job_dirs:
        try: plot_elites(job_dir)
        except Exception as e: 
            enablePrint()
            print(f'An error occurred while plotting {os.path.basename(job_dir)}')
            print(f"Error: {e}")
            blockPrint()
            
            