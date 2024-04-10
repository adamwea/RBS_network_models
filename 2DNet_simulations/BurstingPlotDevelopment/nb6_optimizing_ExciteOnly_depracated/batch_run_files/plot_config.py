import json
import shutil
import os
try: from batch_run_files.aw_batch_tools import generate_all_figures
except: from aw_batch_tools import generate_all_figures

def plot_sim_figs(run_path, simLabel = None):
    exclude_running = False
    output_only = False
    fresh_figs = False
    for root, dirs, files in os.walk(run_path):
        for file in files:
            if file.endswith('_Fitness.json'):
                if '.archive' in root:
                    continue
                if exclude_running and 'output/' in root: continue
                if output_only and 'output/' not in root: continue
                #specific_output = root.split('optimizing/')[1].split('/gen_')[0]
                try:
                    fitness = json.load(open(os.path.join(root, file)))
                    overall_fitness = fitness['fitness']
                except:
                    print(f'Error reading {file}')
                
                #if overall_fitness < fitness_thresh:
                print(f'Overall fitness for {file} is {overall_fitness}')
                print(f'Generating plots for {file}')
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

                    gen_path = os.path.dirname(os.path.dirname(batchdata_path))
                    run_basename = os.path.basename(gen_path)
                    print(f'Generating plots for Candidate: {simLabel_temp}, Run: {run_basename}')

                    # fresh_figs = False
                    assert os.path.exists(batchdata_path), f'{batchdata_path} does not exist'
                    cfg_path = batchdata_path.replace('_data.json', '_cfg.json')
                    assert os.path.exists(cfg_path), f'{cfg_path} does not exist'
                    fitness_path = batchdata_path.replace('_data.json', '_Fitness.json')
                    assert os.path.exists(fitness_path), f'{fitness_path} does not exist'

                    generate_all_figures(
                        fresh_figs = fresh_figs,
                        net_activity_params = {'binSize': .03*1000, 
                                            'gaussianSigma': .12*1000, 
                                            'thresholdBurst': 1.0},
                        batchLabel = 'batchRun_evol',
                        #batchLabel = params['filename'][0],
                        #minimum peak distance = 0.5 seconds
                        batch_path = batchdata_path,
                        simLabel = simLabel_temp
                    )

                    # output_path = batchdata_path
                    # output_path = os.path.dirname(output_path)
                    # output_path = f'{output_path}/NetworkBurst_and_Raster_Figs/'
                    # run_grand_path = os.path.dirname(gen_path)
                    # plot_row_path = f'{run_grand_path}/goodFit_plots_rows/'
                    # if not os.path.exists(plot_row_path+specific_output):
                    #     os.makedirs(plot_row_path+specific_output)
                    # ## shutil copy any files in output_path with the string 'row' in the name to plot path
                    # for rooti, dirsi, filesi in os.walk(output_path):
                    #     if '.archive' in rooti: continue
                        
                    #     for filei in filesi:
                    #         if 'row' in filei:
                    #             if os.path.exists(plot_row_path+filei):
                    #                 if fresh_figs: os.remove(plot_row_path+specific_output+'/'+filei)
                    #                 else: continue
                    #             shutil.copy(rooti+'/'+filei, plot_row_path+specific_output+'/'+filei)
                except:
                    print(f'Error generating plots for {file}')
                    continue