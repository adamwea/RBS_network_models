import json
import shutil
import os
try: from batch_run_files.aw_batch_tools import generate_all_figures
except: from aw_batch_tools import generate_all_figures
import signal

# Timeout handler function
def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

def plot_sim_figs(path, fitness_threshold = None, simLabel = None, net_activity_params = {'binSize': .03*1000, 'gaussianSigma': .12*1000, 'thresholdBurst': 1.0}, timeout = None):
    plot_report = {}
    exclude_running = False
    output_only = False
    fresh_figs = False
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('_Fitness.json'):
                if '.archive' in root:
                    continue
                if exclude_running and 'output/' in root: continue
                if output_only and 'output/' not in root: continue
                #specific_output = root.split('optimizing/')[1].split('/gen_')[0]
                try:
                    fitness = json.load(open(os.path.join(root, file)))
                    average_fitness = fitness['fitness']
                    scaled_average_fitness = fitness['average_scaled_fitness']
                except:
                    print(f'Error reading {file}')
                
                #if overall_fitness < fitness_thresh:
                # print(f'Overall fitness for {file} is {overall_fitness}')
                # print(f'Generating plots for {file}')
                #move_plots_to_plots_folder(root)
                #generate_pdf_reports(root, params)
                #print('Plots and pdf reports generated
                #print(fitness)
                batchdata_path = os.path.join(root, file.replace('_Fitness.json', '_data.json'))
                assert os.path.exists(batchdata_path), f'{batchdata_path} does not exist'
                try:
                    #Target sim as needed
                    simLabel_temp = batchdata_path.split('/')[-1].replace('_data.json', '')
                    if simLabel is not None:
                        if simLabel_temp != simLabel:
                            continue
                    
                    #Apply fitness_thresh as needed
                    if fitness_threshold is not None:
                        max_fitness_metric = max(average_fitness, scaled_average_fitness)
                        #lower is better, if above threshold, skip
                        if max_fitness_metric > fitness_threshold: continue                    

                    #plot
                    gen_path = os.path.dirname(os.path.dirname(batchdata_path))
                    run_basename = os.path.basename(gen_path)
                    print(f'Generating plots for Candidate: {simLabel_temp}, Run: {run_basename}')

                    # fresh_figs = False
                    assert os.path.exists(batchdata_path), f'{batchdata_path} does not exist'
                    cfg_path = batchdata_path.replace('_data.json', '_cfg.json')
                    assert os.path.exists(cfg_path), f'{cfg_path} does not exist'
                    fitness_path = batchdata_path.replace('_data.json', '_Fitness.json')
                    assert os.path.exists(fitness_path), f'{fitness_path} does not exist'

                    def generate_figs_wrapper():
                        #plot_report[simLabel_temp] = 
                        print(f'Generating...')
                        generate_all_figures(
                            fresh_figs = fresh_figs,
                            # net_activity_params = {'binSize': .03*1000, 
                            #                     'gaussianSigma': .12*1000, 
                            #                     'thresholdBurst': 1.0},
                            net_activity_params = net_activity_params,
                            batchLabel = 'batchRun_evol',
                            #batchLabel = params['filename'][0],
                            #minimum peak distance = 0.5 seconds
                            batch_path = batchdata_path,
                            simLabel = simLabel_temp
                        )
                        print(f'Plots successfully generated')#for Candidate: {simLabel_temp}, Run: {run_basename}')
                        
                    
                    if timeout is None:
                        plot_report[simLabel_temp] = 'attempted'
                        generate_figs_wrapper()
                        plot_report[simLabel_temp] = 'successfully plotted'
                    else:
                        # Set the signal handler and a timeout alarm
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(timeout)
                        try:
                            plot_report[simLabel_temp] = 'attempted'
                            generate_figs_wrapper()
                            #reset the alarm
                            signal.alarm(0)
                            plot_report[simLabel_temp] = 'successfully plotted'
                        except TimeoutError as e:
                            #print(f"Timeout error: {e}")
                            # except TimeoutError as e:
                            print(f"Error: {e}")
                            print(f"Plotting timed out after {timeout} seconds.")
                            #reset the alarm
                            signal.alarm(0)
                            plot_report[simLabel_temp] = 'timed out'
                            continue
                except:
                    print(f'Error generating plots for {file}')
                    continue
    
    return plot_report