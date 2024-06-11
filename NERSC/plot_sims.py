# Standard library imports
import os
import subprocess
import sys
import datetime
import json

# Third-party imports for data handling
import pandas as pd
import numpy as np

# Third-party imports for plotting and image handling
import matplotlib.pyplot as plt
from PIL import Image
from math import pi

# Third-party imports for PDF generation
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Local application/library specific imports
from fitness_functions import fitnessFunc
from fitness_functions import load_clean_sim_object
# from NERSC.fitness_functions import fitnessFunc
# from NERSC.fitness_functions import load_clean_sim_object
from fitness_config import *
from USER_INPUTS import *
from plotting_functions import plot_params

# Netpyne
import netpyne

'''functions'''
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
def exempted_print(*args, **kwargs):
    enablePrint()
    print(*args, **kwargs)
    if not verbose: blockPrint()
def generate_pdf_page(data_file_path, elite_paths_cull, gen_rank, HOF_path=None):

    '''supporting functions'''
    def preprocess_df(df):
        # Add a newline after every '{' in each cell
        def add_newline_after_brace(text):
            #text = text.replace(': {', ': {\n\t')
            #text = text.replace('},', '\n')
            #text = text.replace(", '", ", \n'")
            #for every colon, find the matching comma.
            #for every colon found before the comma, find it's matching comma first.
            #only after finding the comma, that belongs to the first colon, add a newline.
            #basically treat a comma like an open brace and colon like a close brace.
            #add line after close brace.
            open_braces = 0
            new_text = text
            #TODO fix shitty hard coded solution
            if 'cutoff' in text:
                new_text = ""
                status = None
                for char in text:
                    if char == ":":
                        status = 1
                        open_braces += 1
                    elif char == ",":
                        open_braces -= 1
                        if last_char == "}":
                            new_text += ",\n"
                        elif open_braces <= 0 and status == 1:
                            status = 0
                            new_text += ",\n"
                        else: new_text += char
                    else: 
                        new_text += char
                    last_char = char
            return new_text

            #return text
        
        for col in df.columns:
            df[col] = df[col].astype(str).apply(add_newline_after_brace)
        return df
    def add_table_to_canvas(canvas, df, x, y, width, fontsize=7):
        #fontsize = 7
        
        #if 'Targets' in df.columns.values.tolist(): df = preprocess_df(df)  # Preprocess the DataFrame to add newlines
        #import textwrap
        # Calculate the maximum width of a cell in characters
        data = [df.columns.values.tolist()] + df.values.tolist()
        # adj_page_width = page_width/3.05 #idk why I need to adjust it like this but whatever it works.
        # # Wrap the text in each cell to the calculated width
        # wrapped_data = [[textwrap.fill(str(cell), width=adj_page_width) for cell in row] for row in data]
        # ... rest of the code ...
        
        # # Create paragraph style for formatting
        # styles = getSampleStyleSheet()
        # style = styles["BodyText"]
        # style.fontSize = fontsize
        
        # # Apply formatting to data
        # formatted_data = []
        # for row in data:
        #     formatted_row = [str(cell) for cell in row]
        #     formatted_data.append(formatted_row)
        
        table_width = 100000000
        while table_width > page_width:
            canvas.setFont("Helvetica", fontsize)
            table = Table(data)
            #table = Table(wrapped_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), fontsize),  # Adjust the font size here\
                ('LEADING', (0, 0), (-1, -1), fontsize+1.5),  # Add this line
                ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),  # Add this line
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),  # Add this line
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black)  # Adjust the border thickness here
            ]))
            table.wrapOn(canvas, 0, 0)
            table_height = table._height
            table_width = table._width
            fontsize -= 0.25
        # if table_width > page_width:
        #     # font_size = font_size - 1
        #     # table.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), font_size)]))
        #     table.wrapOn(canvas, page_width, 0)
        table.drawOn(canvas, x, y - table_height)
        return table_height, table_width, fontsize
    def plot_json_info(canvas, data_file_path, simLabel, gen_rank, elite_paths_cull, page_width, page_height):
        # Load fitness data
        fitness_path = data_file_path.replace('_data.json', '_Fitness.json')
        fitness_data = json.load(open(fitness_path))

        # Convert JSON data to DataFrames
        fitness_df = pd.DataFrame(fitness_data.items(), columns=['Metric', 'Value'])
        from fitness_config import pops
        pops_df = pd.DataFrame(pops.items(), columns=['Criteria', 'Targets'])
        from USER_INPUTS import USER_raster_convolve_params
        convolve_params_df = pd.DataFrame(USER_raster_convolve_params.items(), columns=['Parameter', 'Value'])

        # Add generation and simulation info
        data_file_path_print = os.path.abspath(data_file_path).split('2DNetworkSimulations')[1]
        gen_info = {
            "Data Directory": data_file_path_print,
            "SimLabel": simLabel,
            "Generation Rank": f"{gen_rank}/{len(elite_paths_cull)}"
        }
        gen_info_df = pd.DataFrame(gen_info.items(), columns=['Description', 'Value'])

        # Define the region for the tables
        region_x = 10  # Left-aligned with some margin
        region_y = page_height - 10  # Start near the top of the page with some margin
        region_width = page_width / 3 - 20  # Width for each region with some margin

        '''# Print the tables in the PDF'''
        #canvas.setFont("Helvetica", 6)
        #print the generation info table and convolve params table to the right of it
        total_table_height = 10000000000000 #just needing a large number to start the loop
        fontsize = 7
        while total_table_height > page_height/2:
            total_table_height = 0
            canvas.setFont("Helvetica", fontsize)
            '''tables 1 and 2'''
            table_height, gen_info_table_width, fontsize = add_table_to_canvas(canvas, gen_info_df, region_x, region_y, region_width)
            convolve_table_x = region_x + gen_info_table_width + 5 # Adjust x position for convolve table
            table_height_conv, _, fontsize = add_table_to_canvas(canvas, convolve_params_df, convolve_table_x, region_y, region_width)
            table_height = max(table_height, table_height_conv)  # Adjust height for the tallest table
            total_table_height += table_height
            '''tables 3 '''
            #print the pops table below the gen info table
            region_y -= table_height + 5  # Adjust position for next table with margin
            table_height, _, fontsize = add_table_to_canvas(canvas, pops_df, region_x, region_y, region_width)
            total_table_height += table_height
            '''tables 4'''
            #print the fitness table below the pops table
            region_y -= table_height + 5  # Adjust position for next table with margin
            table_height, _, fontsize = add_table_to_canvas(canvas, fitness_df, region_x, region_y, region_width)
            total_table_height += table_height
            fontsize -= 0.5

    '''Main function to generate PDFs for each candidate'''
    # Determine the candidate plot path based on the data file path
    if '.archive' in data_file_path: cand_plot_path = data_file_path.replace('/output/.archive/', '/plots/')
    elif HOF_path is not None:
        HOF_dt_folder = HOF_path.split('/plots/')[1]
        cand_plot_path = data_file_path.replace('/output/', f'/plots/{HOF_dt_folder}/')
    else: cand_plot_path = data_file_path.replace('/output/', '/plots/')
    cand_plot_path = os.path.dirname(cand_plot_path)
    simLabel = os.path.basename(data_file_path).split('_data')[0]
    simLabel_ = simLabel + '_'

    # Create a new PDF with landscape orientation
    from reportlab.lib.pagesizes import landscape, letter
    if HOF_mode:
        if os.path.exists(cand_plot_path) == False: os.makedirs(cand_plot_path)
    c = canvas.Canvas(os.path.join(cand_plot_path, f"{simLabel}.pdf"), pagesize=landscape(letter))
    page_width, page_height = landscape(letter)

    '''# Define regions for the plots'''
    #region_width = page_width / 3
    region_width = page_width / 4 #moved plot params to bottom row for now.
    #region_height = page_height / 2

    '''# Sample call to the function'''
    exempted_print(f"Generating Info Table for {simLabel}...")
    plot_json_info(c, data_file_path, simLabel, gen_rank, elite_paths_cull, page_width, page_height)

    # '''# Plot parameters in region 3''' #moved to bottom row for now.
    # exempted_print(f"Plotting parameters for {simLabel}")
    # from USER_evol_param_space import params
    # cgf_file_path = data_file_path.replace('_data.json', '_cfg.json')
    # cfg_data = json.load(open(cgf_file_path))['simConfig']
    # param_plot = plot_params(cfg_data, params, cgf_file_path)
    # params_plot_path = os.path.join(cand_plot_path, f'{simLabel}_params_plot.png')
    # if USER_svg_mode: param_plot.savefig(params_plot_path.replace('.png', '.svg')); print("SVG mode enabled.") 
    # param_plot.savefig(params_plot_path)
    # # img = Image.open(params_plot_path).convert('RGB') 
    # # c.drawInlineImage(img, 2 * region_width, region_width, width=region_width, height=region_width)

    #'''Plot comparision in radial format'''
    # exempted_print(f"Plotting radar plot for {simLabel}")
    # from fitness_config import fitnessFuncArgs
    # #fit_config = fitnessFuncArgs['pops']
    # with open(USER_experimental_data, 'r') as f: fit_config = json.load(f)
    # fit_json_path = data_file_path.replace('_data.json', '_Fitness.json')
    # sim_data_fitness_list = [fit_json_path]
    # radar_plt = radar_compare_sims(fit_config, sim_data_fitness_list, saveFig=True, figSize=(10, 10), figName=f'radar_compare_sims_{simLabel}.png')
    # radar_plt_path = os.path.join(cand_plot_path, f'{simLabel}_radar_plot.png')
    # if USER_svg_mode: radar_plt.savefig(radar_plt_path.replace('.png', '.svg')); print("SVG mode enabled.")
    # radar_plt.savefig(radar_plt_path)

    '''Get Files for Region 4, 5, 6''' #added params plot to bottom row.
    def include_exclude_file(file_name, simLabel_):
        """Check if a file should be included based on its name."""
        if not file_name.endswith('.png'):
            return False
        # if '_params_plot.png' in file_name:
        #     return False
        if 'E0' in file_name:
            return False
        if 'I0' in file_name:
            return False
        if 'cell' in file_name:
            return False
        if '_sample_trace.png' in file_name:
            return False
        if 'connections' in file_name:
            return False
        if 'radar_plot' in file_name:
            return False
        if simLabel_ not in file_name:
            return False
        return True
    try: files = [f for f in os.listdir(cand_plot_path) if include_exclude_file(f, simLabel_)]
    except: raise Exception('No .png files found in plots directory')
    raster_plot = None
    network_activity_plot = None
    other_plots = []

    for j, file in enumerate(files):
        if simLabel_ in file and file.endswith('.png'):
            img = Image.open(os.path.join(cand_plot_path, file)).convert('RGB')
            if 'raster' in file:
                raster_plot = img
            elif 'NetworkActivity' in file:
                network_activity_plot = img
            else:
                #other_plots.append((file, img))
                other_plots.append(file)

    # Draw the raster plot on top of the network activity plot
    sim_plot_height = page_height / 4
    sim_plot_width = sim_plot_height * 2
    if raster_plot and network_activity_plot:
        c.drawInlineImage(raster_plot, 0, sim_plot_height, width=sim_plot_width, height=sim_plot_height)
        c.drawInlineImage(network_activity_plot, 0, 0, width=sim_plot_width, height=sim_plot_height)

    # Draw the other plots
    other_plot_height = sim_plot_height
    other_plot_width = sim_plot_width/2
    if '_params_plot' in other_plots[-1]: pass
    else: other_plots.append(other_plots.pop(other_plots.index([file for file in other_plots if '_params_plot' in file][0])))
    for j, file in enumerate(other_plots):
        img = Image.open(os.path.join(cand_plot_path, file)).convert('RGB')
        if 'params_plot' in file: c.drawInlineImage(img, sim_plot_width + other_plot_width*j, 0, width=other_plot_width, height=other_plot_height*2)
        else: c.drawInlineImage(img, sim_plot_width + other_plot_width*j, 0, width=other_plot_width, height=other_plot_height)
        
    '''Place radar plot directly to the right of the raster plot'''
    radar_plot_path = os.path.join(cand_plot_path, f'{simLabel}_radar_plot.png')
    img = Image.open(radar_plot_path).convert('RGB')
    c.drawInlineImage(img, sim_plot_width, sim_plot_height, width=sim_plot_width/2, height=sim_plot_height)
    
    '''# Plot connections'''
    x_coord_for_cons = sim_plot_width + other_plot_width*j + other_plot_width
    conns_plot_width = page_width - x_coord_for_cons
    conns_plot_height = conns_plot_width/2
    conns_plot_path = os.path.join(cand_plot_path, f'{simLabel}_connections.png')
    try: 
        img = Image.open(conns_plot_path).convert('RGB')
        c.drawInlineImage(img, x_coord_for_cons, 0, width=conns_plot_width, height=conns_plot_height)
    except Exception as e: 
        exempted_print(f"Connections plot not found: {e}")
        exempted_print(f"Error: {e}")
    
    '''save the PDF'''
    c.save()
    exempted_print(f"PDF saved to {os.path.join(cand_plot_path, f'{simLabel}.pdf')}")
def recalc_fitness(data_file_path):
    #assert that data_file_path exists, if it doesnt, probably wokring with locally instead of on NERSC
    try: assert os.path.exists(data_file_path), f"Error: {data_file_path} does not exist."
    except: return None, None, None, None, None
    
    print(f"Recalculating Fitness for {os.path.basename(data_file_path)}")
    #load the data file using netpyne loadall
    load_clean_sim_object(data_file_path)
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
    from fitness_config import fitnessFuncArgs
    kwargs = fitnessFuncArgs
    average_fitness = fitnessFunc(
        simData, plot = False, simLabel = simLabel, 
        data_file_path = data_file_path, batch_saveFolder = batch_saveFolder, 
        fitness_save_path = fitness_save_path, **kwargs)
    
    return average_fitness, simData, batch_saveFolder, fitness_save_path, simLabel
def plot_elite_paths(elite_paths, HOF_path = None):

    '''Get pathing information from elite_paths'''
    exempted_print(f"Plotting {len(elite_paths)} elites...")
    assert isinstance(elite_paths, dict), 'Error: elite_paths must be a dict.'
    #elite_paths = dict(elite_paths)
    job_dir = os.path.dirname(os.path.dirname(list(elite_paths.values())[0]['data_file_path']))
    #job_dir = os.path.dirname(os.path.dirname(elite_paths[0]['data_file_path']))
    exempted_print(f"Job Directory: {job_dir}")#; sys.exit() #debugging in terminal
    batch_file_path = [f.path for f in os.scandir(job_dir) if f.is_file() and '_batch.json' in f.name][0]
    exempted_print(f"Batch File Path: {batch_file_path}")
    batch_data = json.load(open(batch_file_path))

    #exempted_print(f"Batch Data: {elite_paths}")
    for simLabel, data in elite_paths.items():

        '''Print simLabel and rank'''
        exempted_print(f"\033[1;33m{simLabel}\033[0m, {data['average_fitness']}")         #print simLabel bold and yellow               
        # Get the average_fitness values for each elite
        average_fitness_values = [data_dict['average_fitness'] for simLabel, data_dict in elite_paths.items()]
        # Rank is the index of the elite in the sorted average_fitness_values list
        gen_rank = average_fitness_values.index(data['average_fitness'])

        ''' initialize variables'''
        data_file_path = data['data_file_path']
        exempted_print(f"Data File Path: {data_file_path}")
        batch_saveFolder = data['batch_saveFolder']
        fitness_save_path = data['fitness_save_path']
        simData = data['simData']
        if simData is None: 
            try:
                exempted_print(f"Error: {simLabel} simData is None. Attempting to load...")
                #exempted_print(data_file_path)
                load_clean_sim_object(data_file_path)
                simData = netpyne.sim.allSimData
                #exempted_print(simData)
                #simData = simData.allSimData
            except Exception as e: 
                exempted_print(f"Error: {e}")
                exempted_print(f"Error: {simLabel} simData failed to load. Skipping...")
        
            #sys.exit() 
            #continue       
    
        '''Prep HOF mode as needed'''
        HOF_path = None
        saveFig = USER_plotting_params['saveFig']
        if HOF_mode:
            date_str = datetime.datetime.now().strftime("%y%m%d")             #get date string YYMMDD
            HOF_plot_dir = f'{date_str}_HOF'                       #prep HOF_plot_dir 
            HOF_path = os.path.join(saveFig, HOF_plot_dir)
            saveFig = HOF_path
            exempted_print(HOF_path)
        
        '''Plotting'''
        #exempted_print(simData)
        #sys.exit()
        data_file_path = data['data_file_path']
        expected_plot_path = os.path.join(data['fitness_save_path'], simLabel + '.pdf').replace('output', 'plots')              
        plots_exist = os.path.exists(expected_plot_path)
        if plots_exist and new_plots == False: exempted_print('Already plotted. Skipping...') #print('Already plotted. Skipping...')
        elif new_plots == True or plots_exist == False or HOF_mode == True: 
            '''fitness plots'''
            from fitness_config import fitnessFuncArgs
            kwargs = fitnessFuncArgs
            assert 'archive' not in data_file_path, 'Error: Cannot plot from archive.'
            exempted_print(f"Plotting {simLabel}...")
            _ = fitnessFunc(
                simData, plot = True, simLabel = simLabel, 
                data_file_path = data_file_path, batch_saveFolder = batch_saveFolder, 
                fitness_save_path = fitness_save_path, plot_save_path = saveFig, **kwargs)
            '''comparison plots'''

            '''Adjust cand_plot_path as needed'''
            if HOF_mode:
                assert HOF_path is not None, 'Error: HOF_path is None.'
                HOF_dt_folder = HOF_path.split('/plots/')[1]
                cand_plot_path = data_file_path.replace('/output/', f'/plots/{HOF_dt_folder}/')
                if os.path.exists(cand_plot_path) == False: os.makedirs(cand_plot_path)
            else: cand_plot_path = data_file_path.replace('/output/', '/plots/')
            cand_plot_path = os.path.dirname(cand_plot_path)
            simLabel = os.path.basename(data_file_path).split('_data')[0] #get simLabel

            '''Plot parameter range fig'''
            exempted_print(f"Plotting parameter ranges for {simLabel}...")
            from evol_param_space import params
            cgf_file_path = data_file_path.replace('_data.json', '_cfg.json')
            cfg_data = json.load(open(cgf_file_path))['simConfig']
            param_plot = plot_params(elite_paths, cfg_data, params, cgf_file_path)
            params_plot_path = os.path.join(cand_plot_path, f'{simLabel}_params_plot.png')
            if USER_svg_mode: param_plot.savefig(params_plot_path.replace('.png', '.svg')); print("SVG mode enabled.") 
            param_plot.savefig(params_plot_path)
            # img = Image.open(params_plot_path).convert('RGB') 
            # c.drawInlineImage(img, 2 * region_width, region_width, width=region_width, height=region_width)

            '''Plot comparision in radial format'''
            exempted_print(f"Plotting radar plot comparison for {simLabel}...")
            from fitness_config import fitnessFuncArgs
            #fit_config = fitnessFuncArgs['pops']
            with open(USER_experimental_data, 'r') as f: fit_config = json.load(f)
            fit_json_path = data_file_path.replace('_data.json', '_Fitness.json')
            sim_data_fitness_list = [fit_json_path]
            radar_plt = radar_compare_sims(fit_config, sim_data_fitness_list, saveFig=True, figSize=(10, 10), figName=f'radar_compare_sims_{simLabel}.png')
            radar_plt_path = os.path.join(cand_plot_path, f'{simLabel}_radar_plot.png')
            if USER_svg_mode: radar_plt.savefig(radar_plt_path.replace('.png', '.svg')); print("SVG mode enabled.")
            radar_plt.savefig(radar_plt_path)

        '''PDF Generation'''
        if USER_svg_mode: continue #skip pdf generation if svg mode is enabled
        if plots_exist and new_pdfs == False:  exempted_print('Skipping pdf generation. They may or may not already exist.')
        elif new_pdfs == True or plots_exist == False:
            exempted_print(f"Generating PDF for {simLabel}...")
            #Network plot, Raster plot, and Trace .pngs should have been generated to plots for each candidate in outputs.
            # Get each PNG, line them up in a row, and save them to a single PDF.
            # os walk through plots folder and generate PDF for each candidate
            try: generate_pdf_page(data_file_path, elite_paths, gen_rank, HOF_path = HOF_path)
            except Exception as e: 
                exempted_print(f'Error: {e}')
                sys.exit() 
def get_elite_paths(gen_dir):       
    
    '''Get elite_paths for each gen_dir'''
    elite_paths = {}        
    if verbose == True: enablePrint()
    # exempted_print(f"Processing {os.path.basename(gen_dir)}")
    # exempted_print(int(gen_dir.split('_')[-1]))
    for root, dirs, files in os.walk(gen_dir):
        for file in files:
            '''Iterate through files in gen_dir and get elite_paths'''
            if '.json' in file and '_data' in file:
                fit_file_path = os.path.join(root, file.replace('_data', '_Fitness'))  #create corresponding data file paths
                data_file_path = os.path.join(root, file)                
                
                '''Skip Cases'''
                if only_gen is not None:
                    if int(gen_dir.split('_')[-1]) != only_gen: 
                        exempted_print(f"Skipping generation: {os.path.basename(gen_dir)}")
                        continue
                # exempted_print(only_gen)
                # exempted_print(f"Processing {os.path.basename(gen_dir)}")
                # exempted_print(int(gen_dir.split('_')[-1]))
                if cand is not None:
                    if int(data_file_path.split('_')[-2]) != cand: 
                        exempted_print(f"Skipping candidate: {os.path.basename(data_file_path)}")
                        continue
                if '.archive' in data_file_path: 
                    exempted_print(f"Skipping archived path: {os.path.basename(data_file_path)}")
                    continue
                if os.path.exists(data_file_path): pass #check if data_file_path exists, if it doesnt, probably wokring with locally instead of on NERSC
                else: continue
                
                '''Recalculate fitness for each data_file_path and add to potential elite_paths'''
                
                if recalc_fitness_flag == True:
                    try: 
                        #exempted_print(f"Recalculating Fitness for {os.path.basename(data_file_path)}")
                        avgFitness, simData, batch_saveFolder, fitness_save_path, simLabel = recalc_fitness(data_file_path)
                        elite_paths[simLabel] = {
                            'average_fitness': avgFitness, 
                            'data_file_path': data_file_path,
                            'batch_saveFolder': batch_saveFolder,
                            'fitness_save_path': fitness_save_path,
                            'simData': simData,
                            }
                    except: pass
                else:
                    try:
                        #exempted_print(f"Loading Fitness for {os.path.basename(data_file_path)}")
                        fit_data = json.load(open(fit_file_path))
                        avgFitness = fit_data['average_fitness']
                        #batch_saveFolder = os.path.dirname(os.path.dirname(data_file_path))
                        batch_saveFolder =os.path.dirname(data_file_path)
                        print(f"Batch Save Folder: {batch_saveFolder}")
                        simLabel = os.path.basename(data_file_path).split('_data')[0]
                        #exempted_print(f"SimLabel: {simLabel}")
                        elite_paths[simLabel] = {
                            'average_fitness': avgFitness, 
                            'data_file_path': data_file_path,
                            'batch_saveFolder': batch_saveFolder,
                            'fitness_save_path': fit_file_path,
                            'simData': None,
                            }
                        if cand and elite_paths[simLabel]['average_fitness'] >= cand_fit_threshold:
                            exempted_print(f"Elite: {simLabel}, {elite_paths[simLabel]['average_fitness']}")
                            exempted_print(f"Skipping candidate, threshold not met.")
                            elite_paths.pop(simLabel) #remove candidate from elite_paths
                            continue
                    except Exception as e:
                        exempted_print(f"Error: {e}")
                        pass
                    #exempted_print(type(elite_paths))
                    #sys.exit()

    '''Cull elite_paths as needed'''
    elite_paths = sorted(elite_paths.items(), key=lambda x: x[1]['average_fitness'], reverse=False)
    if USER_num_elites is not None: num_elites = USER_num_elites
    else: num_elites = 10
    if not HOF_mode: elite_paths_cull = elite_paths[:num_elites]
    else: elite_paths_cull = elite_paths
    #convert elite_paths_cull to dict of dicts
    elite_paths_cull = dict(elite_paths_cull)


    #exempted_print(f"Elite Paths: {elite_paths_cull}")
    #sys.exit()
    
    return elite_paths_cull
def plot_elites(job_dir):

    '''Get gen_dirs and sort them numerically'''
    #sort gendirs numerically such that, gen_9 comes before gen_10
    gen_dirs = [f.path for f in os.scandir(job_dir) if f.is_dir() and 'gen' in f.name]   
    gen_dirs = sorted(gen_dirs, key=lambda x: int(x.split('_')[-1])) 
    if reverse_sort: gen_dirs = gen_dirs[::-1]
    
    '''Iterate through gen_dirs and plot elites for each gen_dir'''
    for gen_dir in gen_dirs:
        #skip if gen_dir is less than start_gen
        if start_gen is not None:
            if int(gen_dir.split('_')[-1]) < start_gen: 
                exempted_print(f"Skipping {os.path.basename(gen_dir)}")
                continue
        print(f"Processing {os.path.basename(gen_dir)}")
        print(int(gen_dir.split('_')[-1]))
        if only_gen is not None:
            if int(gen_dir.split('_')[-1]) != only_gen: 
                exempted_print(f"Skipping {os.path.basename(gen_dir)}")
                continue
        # if cand is not None:
        #     if int(data_file_path.split('_')[-2]) != cand: 
        #         exempted_print(f"Skipping candidate: {os.path.basename(data_file_path)}")
        #         continue

        '''get elite_paths for each gen_dir'''
        #print gen_dir bold and green
        exempted_print(f"\033[1;32m{gen_dir}\033[0m")
        exempted_print(f"Collecting Elites...")
        elite_paths = get_elite_paths(gen_dir)
        if len(elite_paths) == 0: 
            exempted_print(f"No elites found in {os.path.basename(gen_dir)}, continuing...")
            continue
        else: exempted_print(f"Found {len(elite_paths)} elites in {os.path.basename(gen_dir)}")
        #exempted_print(f"Elite Paths: {elite_paths}")
        #sys.exit()
        
        '''plot elite_paths for each gen_dir'''
        exempted_print(f"Plotting Elites...")
        try: plot_elite_paths(elite_paths)
        except Exception as e: 
            exempted_print(f"Error: {e}")
            pass                                        
def plot_HOFs(HOF_dirs):
    elite_paths = {}
    for HOF_dir in HOF_dirs:
        exempted_print(f"\033[1;32m{HOF_dir}\033[0m")
        data_file_path = HOF_dir
        #make sure HOF_dir is functional relative path
        if not os.path.exists(data_file_path): data_file_path = f".{HOF_dir}"
        #recalc fitness
        average_fitness, simData, batch_saveFolder, fitness_save_path, simLabel = recalc_fitness(data_file_path)
        if average_fitness is None: continue
        #add to elite_paths
        elite_paths[simLabel] = {
            'average_fitness': average_fitness,
            'data_file_path': data_file_path,
            'batch_saveFolder': batch_saveFolder,
            'fitness_save_path': fitness_save_path,
            'simData': simData,
            }    
    plot_elite_paths(elite_paths)
def HOF_get_dirs():
    #get HOF dirs
    HOF_dir = 'NERSC/HOF/hof.csv'
    #load HOF csv
    HOF = pd.read_csv(HOF_dir)
    HOF_dirs = HOF.values.tolist()
    HOF_dirs = [f[0] for f in HOF_dirs]
    return HOF_dirs
def radar_compare_sims(fit_config, sim_data_fitness_list, saveFig=False, figSize=(6, 6), figName='radar_compare_sims.png'):
    
    def parse_targets2(d, dict_rad = {}):
        for k, v in d.items():
            #exempted_print(f'{k}, {v}')
            if isinstance(v, dict):
                if 'features' in k.lower(): continue #exclude features subdict
                parse_targets2(v, dict_rad=dict_rad)            
            try: dict_rad[k] = v['Value']
            except: pass
        #remove any None values, replace with 0
        for key in dict_rad.keys():
            if dict_rad[key] == None:
                dict_rad[key] = 0
        return dict_rad
    #get targets
    dict_temp = parse_targets2(fit_config, dict_rad = {})
    targets = dict_temp.copy()

    #load .json files in sim_data-fit_list and parse the fitness values in the same way
    fit_data_dict = {}
    for fit_data in sim_data_fitness_list:
        batchRun = os.path.basename(os.path.dirname(os.path.dirname(fit_data)))
        gen_cand = os.path.basename(fit_data).split('_Fitness')[0]
        with open(fit_data) as f:
            data = json.load(f)
        dict_temp=parse_targets2(data, dict_rad = {})
        fit_data_dict[f'{batchRun}_{gen_cand}']=dict_temp.copy()

    #check which key is missing from fit_data_dict
    for key in targets.keys():
        for key2 in fit_data_dict.keys():
            if key not in fit_data_dict[key2].keys():
                print(f'{key} missing from {key2}')
    assert len(targets) == len(fit_data_dict[list(fit_data_dict.keys())[0]]), 'Number of targets and number of data points do not match'
    
    '''group similar targets together'''
    #move thresh_fit next to baseline_fit
    targets['thresh_fit'] = targets.pop('thresh_fit')
    #move baseline_fit to the end
    targets['baseline_fit'] = targets.pop('baseline_fit')
    # Rearrange fit_data_dict to match order of targets
    for key in fit_data_dict.keys():
        fit_data_dict[key] = {k: fit_data_dict[key][k] for k in targets.keys()}

    # Sample data
    categories = [key for key in targets.keys()]
    target_values = [values for values in targets.values()]
    for key in fit_data_dict.keys():
        values = [values for values in fit_data_dict[key].values()]
        fit_data_dict[key] = values

    # Number of variables
    num_vars = len(categories)

    ##do not edit above this point

    # Compute angle of each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop

    # Create the figure
    fig, ax = plt.subplots(figsize=figSize, subplot_kw=dict(polar=True))

    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], categories, color='grey', size=15)

    # Normalize target values to 100 and convert to log scale
    #target_values = [np.log10(10) for _ in target_values]

    # Plot target data normalize and plot target values
    target_values_plot = [(v / t * 100) for v, t in zip(target_values, target_values)]
    ax.plot(angles, target_values_plot + target_values_plot[:1], linewidth=1, linestyle='solid', label='Target')
    ax.fill(angles, target_values_plot + target_values_plot[:1], 'b', alpha=0.1)

    # Plot comparison data
    colors = ['orange', 'green', 'red', 'purple', 'brown', 'pink']  # Add more colors if needed
    for i, (key, values) in enumerate(fit_data_dict.items()):
        # Normalize comparison values to target values and convert to log scale
        #values = [np.log10(v / t * 100) for v, t in zip(values, target_values)]
        values = [(v / t * 100) for v, t in zip(values, target_values)]
        #convert any -inf values to 0
        for j in range(len(values)):
            if values[j] == float('-inf'):
                values[j] = 1 #set to 1
        #if any values are negative or nan, set to 1
        for j in range(len(values)):
            if values[j] < 0 or np.isnan(values[j]):
                values[j] = 1 #set to 1 for visualization purposes
        #if any values are zero, set to 1
        for j in range(len(values)):
            if values[j] == 0:
                values[j] = 1
        if 'gen' in key: label = f"gen_{key.split('_gen_')[1]}" #should be gen_a_cand_b
        else: label = key
        ax.plot(angles, values + values[:1], linewidth=1, linestyle='solid', label=label, color=colors[i % len(colors)])
        ax.fill(angles, values + values[:1], colors[i % len(colors)], alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1.15), prop={'size': 20})
    
    # Set y-axis to log scale
    ax.set_yscale('log')

    # # Print corresponding values for debug
    # print('Categories:', categories)
    # print('Target values:', [10**v for v in target_values])  # Convert back to linear scale for printing
    # for key, values in fit_data_dict.items():
    #     print(key, [10**v for v in values])  # Convert back to linear scale for printing

    #plt.show()

    #if saveFig:
    #fig.savefig(figName, dpi=600)
    return plt
if __name__ == '__main__':

    ''' Prep flags for argument parsing'''
    import argparse
    parser = argparse.ArgumentParser(description='Plot simulations')
    parser.add_argument('--HOF', action='store_true', help='Plot Hall of Fame candidates')
    parser.add_argument('--job_dirs', nargs='+', help='Plot simulations in job_dirs')
    parser.add_argument('--HOF_dirs', nargs='+', help='Plot Hall of Fame candidates in HOF_dirs')
    parser.add_argument('--cand', type=int, help='Plot candidate cand')
    parser.add_argument('--reverse_sort', action='store_true', help='Sort generations in reverse order')
    parser.add_argument('--new_plots', action='store_true', help='Plot all candidates, regardless of whether they have already been plotted')
    parser.add_argument('--new_pdfs', action='store_true', help='Generate PDFs for all candidates, regardless of whether they have already been generated')
    parser.add_argument('--start_gen', type=int, help='Skip gens less than start_gen')
    parser.add_argument('--elites', type=int, help='Number of elites to plot')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--experimental_data', type=str, help='Path to experimental data for radar_compare_sims comparison')
    parser.add_argument('--recalc', action='store_true', help='Recalculate fitness for all candidates')
    parser.add_argument('--gen', type=int, help='Plot generation gen')
    args = parser.parse_args()
    #args = parser.parse_known_args()
    if args.verbose: verbose = True
    else: verbose = False #set to True to print verbose output
    #exempted_print(args)
    #exempted_print(args.gen)
    #sys.exit()
    
    '''Options for plotting'''

    '''Parse Args'''
    verbose = args.verbose
    recalc_fitness_flag = args.recalc
    USER_num_elites = 10
    new_plots = True
    new_pdfs = True
    start_gen = args.start_gen if args.start_gen is not None else None
    only_gen = args.gen if args.gen is not None else None
    cand = args.cand if args.cand is not None else None
    reverse_sort = args.reverse_sort
    cand_fit_threshold = 700
    # if args.recalc: recalc_fitness_flag = True
    # else: recalc_fitness_flag = False
    # USER_num_elites = 10
    # new_plots = True     #set to True to plot all candidates, regardless of whether they have already been plotted
    # new_pdfs = True
    # #start_gen = 19     #set to some value to skip gens less than start_gen
    # if args.start_gen: start_gen = args.start_gen
    # else: start_gen = None
    # if args.gen is not None: only_gen = args.gen; exempted_print(f"Only plotting generation {only_gen}"); sys.exit()
    # else: only_gen = None
    # if args.cand: cand = args.cand
    # else: cand = None
    # #cand = 70
    # cand = None
    # reverse_sort = False #set to True to sort generations in reverse order
    USER_experimental_data = './experimental_data/KCNT1/wt/experimental_data_Fitness.json'    #for radar_compare_sims comparison
    USER_experimental_data = os.path.abspath(USER_experimental_data)

    '''Modes'''
    if args.HOF: HOF_mode = True #Hall of Fame Mode
    else: HOF_mode = False #HOF Mode

    # ##debugging
    # HOF_mode = True
    # verbose = True
    
    '''Main Function Calls, Different plotting modes for different use cases.'''
    blockPrint()  # suppress all print statements

    if HOF_mode:
        '''Re-plot and re-report Hall of Fame candidates'''
        HOF_dirs = [os.path.abspath(f'.{f}') for f in HOF_get_dirs()]
        print(HOF_dirs)
        try: 
            plot_HOFs(HOF_dirs)
        except Exception as e: 
            exempted_print(f'An error occurred while plotting HOFs')
            exempted_print(f"{e}")
    else:
        '''Typical Case, just plotting simulations to review better solutions. Choosing HOF candidates.'''
        if args.job_dirs is not None: 
            # Split the paths by spaces to create a list
            job_dirs = [os.path.abspath(f'./{job_dir}') for job_dir in args.job_dirs]
        else:
            #check if path is relative or absolute
            job_dirs = [
                #'./NERSC/output/240528_Run1_this_does_work',
            ]
            assert len(job_dirs) > 0, 'Error: job_dirs must be specified in script or as an argument.'        
            job_dirs = [os.path.abspath(job_dir) for job_dir in job_dirs]  # get full paths
        
        '''plot simulations'''
        #run plot_elites    
        for job_dir in job_dirs:
            try: 
                plot_elites(job_dir)
            except Exception as e: 
                exempted_print(f'An error occurred while plotting {os.path.basename(job_dir)}')
                exempted_print(f"Error: {e}")
            
            