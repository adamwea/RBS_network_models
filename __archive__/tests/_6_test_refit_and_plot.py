global PROGRESS_SLIDES_PATH, SIMULATION_RUN_PATH, REFERENCE_DATA_NPY, CONVOLUTION_PARAMS, DEBUG_MODE
#from DIV21.utils.sim_helper import *
from RBS_network_models.developing.utils.sim_helper import reprocess_simulation
from netpyne import sim
# ===================================================================================================
# sim_data_path = (
#     '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/'
#     'CDKL5_DIV21/tests/outputs/test_run_a_simulation/test_run_5_data.pkl')
output_dir =(
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/'
    'CDKL5_DIV21/tests/outputs/test_sensitivity_analysis')
reference_data_path = (
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/'
    'CDKL5_DIV21/src/experimental_reference_data/CDKL5-E6D_T2_C1_05212024_240611_M06844_Network_000076_network_metrics_well000.npy')
# ===================================================================================================
# old code.
    # # run sensitivity analysis
    # _run_sensitivity_analysis = True
    # if _run_sensitivity_analysis:
    #     run_sensitivity_analysis(
    #         sim_data_path, 
    #         output_dir,
    #         plot = True,
    #         reference_data_path = reference_data_path,  # NOTE: this is only needed if plotting is requested, 
    #                                                     #       plots are generated with comparison to reference data
    #         option='parallel',                          # NOTE: this if commented out, the default is 'sequential' - which is better for debugging obvs
    #         num_workers=2,                             # NOTE: this is only needed if option='parallel'. I'm choosing 2 workers for 2 simulations per node.
    #         duration_seconds=30,
    #         #debug=True,                                # NOTE: if true (if uncommented), will load old data and not run new simulations
    #         )
    #     print('Sensitivity analysis successfully ran!')

    # plot_sensitivity_analysis = True
    # if plot_sensitivity_analysis:
    #     #collect all pdfs with "comparison_summary_plot" in the name and make one pdf with all of them
    #     #look in output_dir for all pdfs with "comparison_summary_plot" in the name
    #     import glob
    #     pdfs = glob.glob(output_dir + '/*comparison_summary_slide*.pdf')
    #     #print(pdfs)

    #     # combine all pdfs into one
    #     from PyPDF2 import PdfMerger
    #     merger = PdfMerger()
    #     for pdf in pdfs:
    #         merger.append(pdf)
    #     merger.write(output_dir + '/comparison_summary_plots.pdf')
                
    #     # collect all network_metrics.npy files in output_dir, collect associated summary_plot.png files
    #     network_metrics_files = glob.glob(output_dir + '/*network_metrics.npy')
    #     summary_plots = []
    #     for network_metrics_file in network_metrics_files:
    #         summary_plot = network_metrics_file.replace('network_metrics.npy', 'summary_plot.png')
    #         if os.path.exists(summary_plot):
    #             summary_plots.append(summary_plot)
    #         else:
    #             print('No summary plot found for', network_metrics_file)
                
        
    #     #import numpy as np
        
    #     # burst_rates = {}
    #     # for i, network_metrics_file in enumerate(network_metrics_files):
    #     #     data = np.load(network_metrics_file, allow_pickle=True).item()
    #     #     mean_burst_rate = data['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Rate']
    #     #     burst_rates[i] = {
    #     #         'summary_plot': network_metrics_file.replace('network_metrics.npy', 'summary_plot.png'),
    #     #         'br': mean_burst_rate
    #     #     }
    #     #     print('network_metrics loaded. mean_burst_rate:', mean_burst_rate)
        
    #     # Use parallel processing to load and process files
    #     import concurrent.futures
    #     def process_file(network_metrics_file):
    #         try:
    #             basename = os.path.basename(network_metrics_file)
    #             print('Loading', basename, '...')
    #             data = np.load(network_metrics_file, allow_pickle=True).item()
    #             mean_burst_rate = data['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Rate']
    #             summary_plot = network_metrics_file.replace('network_metrics.npy', 'summary_plot.png')
    #             print('Done!')
    #             return mean_burst_rate, summary_plot
    #         except Exception as e:
    #             print('Error loading', network_metrics_file, ':', e)
    #             return None, None
    #     burst_rates = {}
    #     def cpu_count():
    #         try:
    #             return len(os.sched_getaffinity(0))
    #         except AttributeError:
    #             return os.cpu_count()
    #     workers = np.min([len(network_metrics_files), cpu_count()])
    #     workers = int(workers)
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
    #         results = list(executor.map(process_file, network_metrics_files))
    #     for i, (mean_burst_rate, summary_plot) in enumerate(results):
    #         burst_rates[i] = {
    #             'summary_plot': summary_plot,
    #             'br': mean_burst_rate
    #         }
    #         #print('network_metrics loaded. mean_burst_rate:', mean_burst_rate)
            
    #     # Plot n_params x 2 grid of summary plots
    #     from CDKL5_DIV21.src.evol_params import params
    #     import matplotlib.pyplot as plt
    #     import matplotlib.image as mpimg
    #     print('Plotting summary grid')
    #     n_params = len(burst_rates)
    #     n_rows = (n_params - 1) // 2 # subtract original sim, divide by 2 1 row per two plots
    #     fig, axs = plt.subplots(n_rows, 2, figsize=(15, 5*n_rows))
    #     real_param_idx = 0
    #     real_param_idx_dict = {}

    #     for param_idx, param in enumerate(params):
    #         reduced_plot = None
    #         increased_plot = None
    #         for summary in burst_rates.values():
    #             summary_plot_path = summary['summary_plot']
    #             if param in summary_plot_path:
    #                 if 'increased' in summary_plot_path:
    #                     increased_plot = summary_plot_path
    #                 elif 'reduced' in summary_plot_path:
    #                     reduced_plot = summary_plot_path

    #         if reduced_plot and increased_plot:
    #             real_param_idx_dict[param] = real_param_idx
    #             idx_param = real_param_idx
    #             real_param_idx += 1

    #             img_reduced = mpimg.imread(reduced_plot)
    #             axs[idx_param, 0].imshow(img_reduced)
    #             axs[idx_param, 0].axis('off')
    #             axs[idx_param, 0].set_title(f'param: {param} (reduced)')

    #             img_increased = mpimg.imread(increased_plot)
    #             axs[idx_param, 1].imshow(img_increased)
    #             axs[idx_param, 1].axis('off')
    #             axs[idx_param, 1].set_title(f'param: {param} (increased)')

    #     plt.tight_layout()
    #     print('Saving summary grid to', output_dir + '/_summary_grid.png')
    #     plt.savefig(output_dir + '/_summary_grid.png', dpi=300)
        
    #     # createa a parallel plot of burst rates. 1 row per parameter. 2 columns, one for reduced, one for increased
    #     # plot burst rates in each square. 
    #     # establish a color gradient. More red = higher burst rate than original.
    #     # more blue = lower burst rate than original
    #     # white = same as original
    #     import matplotlib.colors as mcolors
    #     import matplotlib.cm as cm
    #     import numpy as np
    #     from matplotlib.patches import Rectangle
    #     import matplotlib.pyplot as plt
        
    #     # Create a color gradient
    #     colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # B -> W -> R
    #     n_bins = 100  # Discretizes the interpolation into bins
    #     cmap_name = 'my_list'
    #     cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    #     # Plot n_params x 2 grid of burst rate values with color modulation
    #     print('Plotting summary grid with color gradient')
    #     n_params = len(burst_rates)
    #     n_rows = (n_params - 1) // 2 # subtract original sim, divide by 2 1 row per two plots
    #     fig, axs = plt.subplots(n_rows, 2, figsize=(15, 5*n_rows))
    #     real_param_idx = 0
    #     real_param_idx_dict = {}

    #     # Placeholder for original burst rate value
    #     #original_burst_rate = 1.0  # Replace this with the actual original burst rate value
    #     basename = os.path.basename(sim_data_path)
    #     #remove file extension
    #     basename = os.path.splitext(basename)[0]
    #     basename = basename.replace('_data', '')
    #     rerun_original = [
    #         #x for x in burst_rates.values()
    #         x for x in burst_rates.values() 
    #         if basename in x['summary_plot']
    #         ]
    #     assert len(rerun_original) == 1, f'Expected 1 rerun original, found {len(rerun_original)}'
    #     original_path = rerun_original[0]['summary_plot']
    #     # original_data = [
    #     #     results for results in burst_rates.values() 
    #     #     if results['summary_plot'] == original_path
    #     #                  ]
    #     # assert len(original_data) == 1, f'Expected 1 original data, found {len(original_data)}'
    #     original_burst_rate = rerun_original[0]['br']

    #     for param_idx, param in enumerate(params):
    #         reduced_br = None
    #         increased_br = None
    #         for summary in burst_rates.values():
    #             summary_plot_path = summary['summary_plot']
    #             if param in summary_plot_path:
    #                 if 'increased' in summary_plot_path:
    #                     increased_br = summary['br']
    #                 elif 'reduced' in summary_plot_path:
    #                     reduced_br = summary['br']

    #         if reduced_br is not None and increased_br is not None:
    #             real_param_idx_dict[param] = real_param_idx
    #             idx_param = real_param_idx
    #             real_param_idx += 1

    #             # Calculate color based on burst rate
    #             reduced_color = cmap((reduced_br - original_burst_rate) / 2 + 0.5)
    #             increased_color = cmap((increased_br - original_burst_rate) / 2 + 0.5)

    #             # Plot reduced burst rate
    #             axs[idx_param, 0].add_patch(Rectangle((0, 0), 1, 1, transform=axs[idx_param, 0].transAxes,
    #                                                 color=reduced_color, alpha=0.5))
    #             axs[idx_param, 0].text(0.5, 0.5, f'{reduced_br:.2f}', ha='center', va='center', fontsize=12)
    #             axs[idx_param, 0].axis('off')
    #             axs[idx_param, 0].set_title(f'param: {param} (reduced)')

    #             # Plot increased burst rate
    #             axs[idx_param, 1].add_patch(Rectangle((0, 0), 1, 1, transform=axs[idx_param, 1].transAxes,
    #                                                 color=increased_color, alpha=0.5))
    #             axs[idx_param, 1].text(0.5, 0.5, f'{increased_br:.2f}', ha='center', va='center', fontsize=12)
    #             axs[idx_param, 1].axis('off')
    #             axs[idx_param, 1].set_title(f'param: {param} (increased)')

    #     # Add color bar legend
    #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    #     sm.set_array([])
    #     cbar = fig.colorbar(sm, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)
    #     cbar.set_label('Burst Rate Change (relative to original)')

    #     plt.tight_layout()
    #     print('Saving summary grid to', output_dir + '/_color_summary_grid.png')
    #     plt.savefig(output_dir + '/_color_summary_grid.png', dpi=300)
            
                
    #     # # extract burst rate information from each fitness.pkl file, create a dict of burst rates and assiciated summary_plot.png files
    #     # burst_rates = {}
    #     # for fitness_file in fitness_files:
    #     #     with open(fitness_file, 'rb') as f:
    #     #         data = pickle.load(f)
    #     #     burst_rates[fitness_file] = data['burst_rate']

#walk through output_dir and get all .pkl file paths
sim_data_paths = []
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith('_data.pkl'):
            sim_data_paths.append(os.path.join(root, file))

# refit and plot simulations against reference data            
for sim_data_path in sim_data_paths:
    try:
        sim.clearAll()
    except:
        pass
    
    print('Processing', sim_data_path)
    reprocess_simulation(
        sim_data_path, 
        reference_data_path, 
        #conv_params_path,
        #target_script_path=target_script_path, 
        #duration_seconds=duration_seconds,
        #save_data=save_data, 
        #overwrite_cfgs=overwrite_cfgs
        DEBUG_MODE=False,
        #DEBUG_MODE=False
    )