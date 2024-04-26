import os
import netpyne
import subprocess
from fitness_functions import fitnessFunc
from fitness_config import *
kwargs = fitnessFuncArgs
#from USER_INPUTS import *

#get output directory
output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')

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

def generate_pdf_page(data_file_path, ):
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
    
    files = [f for f in os.listdir(cand_plot_path) if simLabel_ in f and f.endswith('.png')]
    img_width = page_width/len(files)
    img_height = img_width

    # Walk through the plots directory
    j=0
    for root, dirs, files in os.walk(cand_plot_path):
        # Sort the files to ensure they're added in the correct order
        files.sort()
        for i, file in enumerate(files):
            if simLabel_ in file and file.endswith('.png'):                
                # Open the image file
                img = Image.open(os.path.join(root, file))
                # Convert the image to RGB (required by reportlab)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Save the image to the PDF, adjusting the position for each image
                c.drawInlineImage(img, j * img_width, 0, width=img_width, height=img_height)
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

    #import dictionary pops from fitness_config.py and covert print the same way as fitness_data
    from fitness_config import pops
    y_position = page_height - 20  # Start near the top of the page
    for key, value in pops.items():
        c.drawString(10, y_position, f"{key}: {value}")
        y_position -= 15  # Move down for the next line
    
    y_position = y_position - 20  # Start near the top of the page
    for key, value in fitness_data.items():
        c.drawString(10, y_position, f"{key}: {value}")
        y_position -= 15  # Move down for the next line

    # Save the PDF
    c.save()

#surpress all print statements
print('Running fitness function on all data files in output directory...')
blockPrint()
#os walk through output path and find all .json files with containing fitness in the name
#output_path = '/home/adamm/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240425_Run3_overnightRun24Apr_1x128'
for root, dirs, files in os.walk(output_path):
    for file in files:
        #if '.json' in file and 'Fitness' in file:
        if '.json' in file and '_data' in file:
            #create corresponding data file path
            fit_file_path = os.path.join(root, file.replace('_data', '_Fitness'))
            data_file_path = os.path.join(root, file)
            
            #check if data file_path exists
            if os.path.exists(data_file_path): pass
            else: continue

            #load the data file using netpyne loadall
            netpyne.sim.loadAll(data_file_path)
            simData = netpyne.sim.allSimData
            batch_saveFolder = netpyne.sim.cfg.saveFolder
            #get cwd
            cwd = os.getcwd()
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
            if avgScaledFitness < 750:
                #enable all print statements
                enablePrint()

                avgScaledFitness = fitnessFunc(
                simData, sim_obj = netpyne.sim, plot = True, simLabel = simLabel, 
                data_file_path = data_file_path, batch_saveFolder = batch_saveFolder, 
                fitness_save_path = fitness_save_path, **kwargs)
                print('Plots saved to: ...')
                #Network plot, Raster plot, and Trace .pngs should have been generated to plots for each candidate in outputs.
                # Get each PNG, line them up in a row, and save them to a single PDF.
                # os walk through plots folder and generate PDF for each candidate
                generate_pdf_page(data_file_path)

                #create an array collecting info as follows:
                #avgScaledFitness, 

                #surpress all print statements
                blockPrint()

from PyPDF2 import PdfFileMerger
import os

# Create a PDF merger
merger = PdfFileMerger()

# Walk through the output directory
for root, dirs, files in os.walk(output_path):
    # Sort the files to ensure they're merged in the correct order
    files.sort()
    for file in files:
        if file.endswith('.pdf'):
            # Merge the PDFs
            merger.append(os.path.join(root, file))

# Save the merged PDF
merger.write(os.path.join(output_path, "data.pdf"))
merger.close()          
            

            
            #print('Average Scaled Fitness:', avgScaledFitness)

            # ## get info from .csv
            # indiv_csv_dir = os.path.dirname(os.path.dirname(data_file_path))
            # for root, dirs, files in os.walk(indiv_csv_dir):
            #     for file in files:
            #         if 'indiv.csv' in file:
            #             csv_file_path = os.path.join(root, file)
            #             # Read the CSV file line by line
            #             with open(csv_file_path, 'r') as f:
            #                 for line in f:
            #                     #get first line to define columns
            #                     if 'gen' in line:
            #                         columns = line.split(',')
            #                         df = pd.DataFrame(columns=columns)
            #                         print('Columns:', columns)
            #                     # Check if avgScaledFitness is in the line
            #                     if str(avgScaledFitness) in line:
            #                         print('Found line with average scaled fitness:', line)
            #                         # Convert the line to a DataFrame and append it to df
            #                         #line_df = pd.DataFrame([line.split(',')])
            #                         line_df = pd.DataFrame([line.split(',')], columns=columns)
            #                         #df = df.append(line_df, ignore_index=True)
            #                         # # Extend row with fitness values at data at fit_file_path, name new columns by .json key names
            #                         # with open(fit_file_path, 'r') as f:
            #                         #     fitness_data = json.load(f)
            #                         #     for key, value in fitness_data.items():
            #                         #         df[key] = value
            #                         # break

            # # Save the DataFrame to a CSV file
            # #plots_path = '/home/adamm/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/plots'
            # #df.to_csv(os.path.join(plots_path, 'fitness_data.csv'), index=False)
            # print('Saved fitness data to:', os.path.join(indiv_csv_dir, 'fitness_data.csv'))
            
            