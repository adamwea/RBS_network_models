'''imports'''
import os
import netpyne
import subprocess
from fitness_functions import fitnessFunc
from fitness_config import *
import pandas as pd
import json
import os
from reportlab.pdfgen import canvas
from PIL import Image
import sys, os
import datetime
from USER_INPUTS import *
from reportlab.lib.pagesizes import letter

'''functions'''
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
def generate_pdf_page(data_file_path, elite_paths_cull, gen_rank, HOF_path=None):
    import matplotlib.pyplot as plt
    import os
    import json
    from reportlab.pdfgen import canvas
    from PIL import Image

    # Function to plot parameters from configuration data
    def plot_params(cfg_data, params, cgf_file_path):
        
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
                assert 0 <= val[0] <= 1, f"Error: {key} has an invalid value."
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
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)

        # Plot the main configuration data
        main_cfg_data = cfg_data.copy()
        main_cfg_path = cgf_file_path

        # Plot other configuration data for comparison
        elite_cfg_paths = [f[1]['data_file_path'].replace('_data', '_cfg') for f in elite_paths_cull]
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
    import pandas as pd
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
    from reportlab.lib.pagesizes import landscape, letter

    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    import pandas as pd
    import json
    import os

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

    def add_table_to_canvas(canvas, df, x, y, width):
        fontsize = 6
        canvas.setFont("Helvetica", fontsize)
        if 'Targets' in df.columns.values.tolist(): df = preprocess_df(df)  # Preprocess the DataFrame to add newlines
        data = [df.columns.values.tolist()] + df.values.tolist()
        
        # # Create paragraph style for formatting
        # styles = getSampleStyleSheet()
        # style = styles["BodyText"]
        # style.fontSize = fontsize
        
        # # Apply formatting to data
        # formatted_data = []
        # for row in data:
        #     formatted_row = [str(cell) for cell in row]
        #     formatted_data.append(formatted_row)
        
        table = Table(data)
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
        table.wrapOn(canvas, width, 0)
        table_height = table._height
        table.drawOn(canvas, x, y - table_height)
        return table_height

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

        # Print the tables in the PDF
        #canvas.setFont("Helvetica", 6)
        table_height = add_table_to_canvas(canvas, gen_info_df, region_x, region_y, region_width)
        region_y -= table_height + 5  # Adjust position for next table with margin
        table_height = add_table_to_canvas(canvas, pops_df, region_x, region_y, region_width)
        region_y -= table_height + 5  # Adjust position for next table with margin
        table_height = add_table_to_canvas(canvas, fitness_df, region_x, region_y, region_width)
        region_y -= table_height + 5  # Adjust position for next table with margin
        add_table_to_canvas(canvas, convolve_params_df, region_x, region_y, region_width)

    # Determine the candidate plot path based on the data file path
    if '.archive' in data_file_path:
        cand_plot_path = data_file_path.replace('/output/.archive/', '/plots/')
    elif HOF_path is not None:
        HOF_dt_folder = HOF_path.split('/plots/')[1]
        cand_plot_path = data_file_path.replace('/output/', f'/plots/{HOF_dt_folder}/')
    else:
        cand_plot_path = data_file_path.replace('/output/', '/plots/')
    cand_plot_path = os.path.dirname(cand_plot_path)
    simLabel = os.path.basename(data_file_path).split('_data')[0]
    simLabel_ = simLabel + '_'

    # Create a new PDF with landscape orientation
    from reportlab.lib.pagesizes import landscape, letter
    c = canvas.Canvas(os.path.join(cand_plot_path, f"{simLabel}.pdf"), pagesize=landscape(letter))
    page_width, page_height = landscape(letter)

    # Define regions for the plots
    region_width = page_width / 3
    region_height = page_height / 2

    # Sample call to the function
    plot_json_info(c, data_file_path, simLabel, gen_rank, elite_paths_cull, page_width, page_height)
    #c.save()

    # Plot connections in region 2
    conns_plot_path = os.path.join(cand_plot_path, f'{simLabel}_connections.png')
    img = Image.open(conns_plot_path).convert('RGB')
    c.drawInlineImage(img, region_width, region_width, width=region_width, height=region_width / 2)

    # Plot parameters in region 3
    from USER_evol_param_space import params
    cgf_file_path = data_file_path.replace('_data.json', '_cfg.json')
    cfg_data = json.load(open(cgf_file_path))['simConfig']
    plt = plot_params(cfg_data, params, cgf_file_path)
    params_plot_path = os.path.join(cand_plot_path, f'{simLabel}_params_plot.png')
    plt.savefig(params_plot_path)
    img = Image.open(params_plot_path).convert('RGB')
    c.drawInlineImage(img, 2 * region_width, region_width, width=region_width, height=region_width)

    # Define the width and height for the bottom row images
    try:
        files = [f for f in os.listdir(cand_plot_path)
                 if simLabel_ in f and f.endswith('.png') and '_params_plot.png' not in f
                 and 'E0' not in f and 'I0' not in f and 'cell' not in f
                 and '_sample_trace.png' not in f and 'connections' not in f]
        img_width = region_width
        img_height = region_width
    except:
        raise Exception('No .png files found in plots directory')

    # Add images to the bottom row (regions 4, 5, 6)
    for j, file in enumerate(sorted(files)):
        if simLabel_ in file and file.endswith('.png') and '_params_plot.png' not in file:
            img = Image.open(os.path.join(cand_plot_path, file)).convert('RGB')
            c.drawInlineImage(img, j * region_width, 0, width=region_width, height=region_width)

    # Save and close the PDF
    c.save()
    #open the pdf
    #evince not found
    os.system(f'evince {os.path.join(cand_plot_path, f"{simLabel}.pdf")}')
    print(f"PDF saved to {os.path.join(cand_plot_path, f'{simLabel}.pdf')}")

def recalc_fitness(data_file_path):
    #assert that data_file_path exists, if it doesnt, probably wokring with locally instead of on NERSC
    try: assert os.path.exists(data_file_path), f"Error: {data_file_path} does not exist."
    except: return None, None, None, None, None
    
    print(f"Recalculating Fitness for {os.path.basename(data_file_path)}")
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
    
    return avgScaledFitness, simData, batch_saveFolder, fitness_save_path, simLabel
def plot_elite_paths(elite_paths, HOF_path = None):
    #elite_rate = 0.10
    #get path in job_dir ending in _batch.json and load
    job_dir = os.path.dirname(os.path.dirname(list(elite_paths.values())[0]['data_file_path']))
    print(f"Job Directory: {job_dir}")
    batch_file_path = [f.path for f in os.scandir(job_dir) if f.is_file() and '_batch.json' in f.name][0]
    batch_data = json.load(open(batch_file_path))
    
    # try: assert num_elites < len(elite_paths), f"Error: num_elites must be less than the number of elites in the generation."
    # except: 
    #     if cand is not None: pass
    #     else: raise Exception(f"Error: num_elites must be less than the number of elites in the generation.")

    elite_paths = sorted(elite_paths.items(), key=lambda x: x[1]['avgScaledFitness'], reverse=False)
    num_elites = batch_data['batch']['evolCfg']['num_elites']
    num_elites = len(elite_paths)
    if not HOF_mode: elite_paths_cull = elite_paths[:num_elites]
    else: elite_paths_cull = elite_paths
    for simLabel, data in elite_paths_cull:
        
        #print simLabel bold and yellow
        enablePrint()
        print(f"\033[1;33m{simLabel}\033[0m, {data['avgScaledFitness']}")               
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
        HOF_path = None
        saveFig = USER_plotting_params['saveFig']
        if HOF_mode:
            #get date string YYMMDD            
            date_str = datetime.datetime.now().strftime("%y%m%d")
            #prep HOF_plot_dir
            HOF_plot_dir = f'{date_str}_HOF'            
            HOF_path = os.path.join(saveFig, HOF_plot_dir)
            enablePrint()
            saveFig = HOF_path
            print(HOF_path)
            blockPrint()
            #print(HOF_path)
        #sys.exit()
        data_file_path = data['data_file_path']
        expected_plot_path = os.path.join(data['fitness_save_path'], simLabel + '.pdf').replace('output', 'plots')              
        plots_exist = os.path.exists(expected_plot_path)
        if plots_exist and new_plots == False: 
            print('Already plotted. Skipping...')
            #continue
            pass
        elif new_plots == True or plots_exist == False: 
            assert 'archive' not in data_file_path, 'Error: Cannot plot from archive.'
            avgScaledFitness = fitnessFunc(
                simData, plot = True, simLabel = simLabel, 
                data_file_path = data_file_path, batch_saveFolder = batch_saveFolder, 
                fitness_save_path = fitness_save_path, plot_save_path = saveFig, **kwargs)
            print('Plots saved to: ...')
            plots_exist = os.path.exists(expected_plot_path)

        if plots_exist and new_pdfs == False:
            #print('PDF already generate. Skipping...')
            #print('Plots not generated for this PDF. Skipping...')
            #continue
            print('Skipping pdf generation. They may or may not already exist.')
            pass
        elif new_pdfs == True or plots_exist == False:
            #Network plot, Raster plot, and Trace .pngs should have been generated to plots for each candidate in outputs.
            # Get each PNG, line them up in a row, and save them to a single PDF.
            # os walk through plots folder and generate PDF for each candidate
            try: generate_pdf_page(data_file_path, elite_paths_cull, gen_rank, HOF_path = HOF_path)
            except Exception as e:
                enablePrint()
                print(f'Error: {e}')
                blockPrint()           
                pass
        #sys.exit()   
def get_elite_paths(gen_dir):       
    elite_paths = {}        
    if verbose == True: enablePrint()
    for root, dirs, files in os.walk(gen_dir):
        for file in files:
            #if '.json' in file and 'Fitness' in file:
            if '.json' in file and '_data' in file:

                #create corresponding data file paths
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
                    avgScaledFitness, simData, batch_saveFolder, fitness_save_path, simLabel = recalc_fitness(data_file_path)
                    elite_paths[simLabel] = {'avgScaledFitness': avgScaledFitness, 
                                                'data_file_path': data_file_path,
                                                'batch_saveFolder': batch_saveFolder,
                                                'fitness_save_path': fitness_save_path,
                                                'simData': simData,
                                                }
                except: pass
    return elite_paths
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

        enablePrint()    
        print(f"Collecting Elites...")
        blockPrint()    

        elite_paths = get_elite_paths(gen_dir)
        
        try: plot_elite_paths(elite_paths)
        except: pass                                
def plot_HOFs(HOF_dirs):
    elite_paths = {}
    for HOF_dir in HOF_dirs:
        enablePrint()
        print(f"\033[1;32m{HOF_dir}\033[0m")
        blockPrint()
        if verbose == True: enablePrint()
        data_file_path = HOF_dir
        #make sure HOF_dir is functional relative path
        if not os.path.exists(data_file_path): data_file_path = f".{HOF_dir}"
        #recalc fitness
        avgScaledFitness, simData, batch_saveFolder, fitness_save_path, simLabel = recalc_fitness(data_file_path)
        if avgScaledFitness is None: continue
        #add to elite_paths
        elite_paths[simLabel] = {
            'avgScaledFitness': avgScaledFitness,
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

if __name__ == '__main__':
    kwargs = fitnessFuncArgs
    
    #surpress all print statements
    blockPrint()
    
    #set to some value to skip gens less than start_gen
    start_gen = 10
    start_gen = None
    cand = 70
    cand = None

    #set to True to plot all candidates, regardless of whether they have already been plotted
    new_plots = False
    new_pdfs = True

    #set to True to print verbose output
    verbose = True

    #HOF Mode
    HOF_mode = False
    
    if HOF_mode:
        #new_plots = True
        HOF_dirs = HOF_get_dirs()
        #these should be relative paths, get full paths
        HOF_dirs = [f'.{f}' for f in HOF_dirs]
        HOF_dirs = [os.path.abspath(f) for f in HOF_dirs]
        #enablePrint()
        print(HOF_dirs)
        #blockPrint()
        try: plot_HOFs(HOF_dirs)
        except Exception as e: 
            enablePrint()
            print(f'An error occurred while plotting HOFs')
            print(f"{e}")
            blockPrint()

    else:
        job_dirs = [
            #'/home/adamm/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240426_Run12_26AprSAFE_1x100',
            #'/home/adamm/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240429_Run2_debug_node_run',        
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240426_Run12_26AprSAFE_1x100',
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240429_Run1_debug_node_run',
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240429_Run2_debug_node_run',
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240430_Run1_interactive_node_run',
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240430_Run2_debug_node_run',
            
            # '/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240505_Run16_debug_node_run',
            # '/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240505_Run17_debug_node_run',
            # '/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240505_Run18_debug_node_run',
            # '/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240506_Run1_overnightRun',

            # '/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run13_interactive',
            # '/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run14_OMPTest',
            # '/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run15_OMPTest2'

            #'./NERSC/output/240517_Run1_best_case',

            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run22_2proc',
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run23_4proc',
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run24_8proc',
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run25_16proc',
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run26_8proc',
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run27_16proc',
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run28_32proc',
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run29_64proc',
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run30_128proc',
            
            #'/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240521_Run1_overnight_debug',
            
            #'./NERSC/output/240521_Run2_overnight_reg',

            './NERSC/output/240522_Run3_it_srun_sims',

            ]
        
        job_dirs = [os.path.abspath(job_dir) for job_dir in job_dirs] #: job_dir = os.path.abspath(job_dir)
            
            #
        
        #run plot_elites    
        for job_dir in job_dirs:
            try: plot_elites(job_dir)
            except Exception as e: 
                enablePrint()
                print(f'An error occurred while plotting {os.path.basename(job_dir)}')
                print(f"Error: {e}")
                blockPrint()
            
            