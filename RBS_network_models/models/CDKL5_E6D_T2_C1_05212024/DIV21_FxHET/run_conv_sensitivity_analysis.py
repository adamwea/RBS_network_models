'''
This script is based on analyze_sorted.py. Tweaks downstream logic to tune conv_params for the CDKL5_E6D_T2_C1_05212024 model reference data.
'''
# Imports =====================================================================
import os
from RBS_network_models import conv_sensitivity_analysis as csa

# Paths =============================================================================
sorted_data_dirs = [
    '/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/sorted/well005',
    #'/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/sorted/well001',
    ] # NOTE: this is a list of paths to sorted data files that you want to extract features from.

output_dirs = [
    sorted_dir.replace('sorted', 'conv_analysis') for sorted_dir in sorted_data_dirs
    ] # NOTE: this is a list of output directories for each network analysis of each sorted data file.

# Parallelism =============================================================================
'''check available cores'''
print("Number of cores available: ", os.cpu_count())
#max_workers = 128 # aw 2025-02-24 04:07:33 - I got an odd error trying to use 256 cores... just going to use 128 for now.

# Main =============================================================================
'''main'''
kwargs = {
    'sorted_data_dirs': sorted_data_dirs,
    'output_dirs': output_dirs,
    
    # typical configuration worker spec
    #'max_workers': 32, # 1/4 node - use in login node
    #'max_workers': os.cpu_count(), # use all available cores
    
    # specify bin_size, gaussian_sigma like so [min, max, step]
    'bin_size_range': [0.01, 1, 0.01], # aw 2025-01-25 12:16:01 - updated to be more sensitive
    'gaussian_sigma_range': [0.01, 1, 0.01], # aw 2025-01-25 12:16:01 - updated to be more sensitive
    
    # batch optimized implementation
    'max_workers': 8, # aw 2025-01-25 12:16:01 - updated to be more sensitive
    'child_max_workers': 64, # aw 2025-01-25 12:16:01 - updated to be more sensitive
    #'child_max_workers': 1, # aw 2025-01-25 12:16:01 - updated to be more sensitive
}

# typical way to run the analysis, slow but works
#feature_data = csa.run_analysis(**kwargs)

# batch optimized implementation of the analysis
feature_data = csa.batch_optimized_run_analysis(**kwargs)
print("Network Analysis Complete.")

# Perlmutter =============================================================================
'''
#run in interactive node
salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=1 --image=adammwea/axonkilo_docker:v7
shifter --image=adammwea/axonkilo_docker:v7 /bin/bash
python /global/homes/a/adammwea/workspace/repos/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_WT/analyze_sorted.py
'''
# saving plot logic for later.
    #plot network activity
    # if plot:
        
    #     #plot neuron locations and class. Inhibitory neurons are red, excitatory neurons are blue.
    #     #save_path = save_path.replace('.npy', '_neuron_locations.pdf')
    #     #save_path = save_path.replace('.npy', '_neuron_locations.png')
    #     #neuron_loc_plot_path = save_path.replace('network_metrics', 'neuron_loc_plots')
    #     neuron_loc_plot_path = save_path.replace('network_analysis', 'location_plots')
    #     neuron_loc_parent_dir = os.path.dirname(neuron_loc_plot_path)
    #     if not os.path.exists(neuron_loc_parent_dir):
    #         os.makedirs(neuron_loc_parent_dir)
    #     neuron_loc_plot_path = os.path.join(neuron_loc_parent_dir, f"neuron_locations.png")
    #     #plot_neuron_locations_and_class(sorting_object, wf_extractor, network_metrics, save_path=neuron_loc_plot_path)
    #     plot_neuron_locations_and_class(sorting_object, sorting_analyzer, network_metrics, save_path=neuron_loc_plot_path)
        
        
    #     try: 
    #         print("Generating network summary plot...")
    #         # aw 2025-01-20 17:25:21 - I guess I'll just plot both for now. I like how mine looks, but additional context is nice for Roy.
    #         #network_plot_path = save_path.replace('network_metrics', 'network_plots')
    #         network_plot_path = save_path.replace('network_analysis', 'network_plots')
    #         network_plot_parent_dir = os.path.dirname(network_plot_path)
    #         if not os.path.exists(network_plot_parent_dir):
    #             os.makedirs(network_plot_parent_dir)
    #         #network_plot_path = os.path.join(network_plot_parent_dir, f"network_summary_plot.pdf")
    #         network_plot_path_3p = os.path.join(network_plot_parent_dir, f"network_summary_3pannels.pdf")
    #         network_plot_path_2p = os.path.join(network_plot_parent_dir, f"network_summary_2pannels.pdf")
    #         network_plot_path_3p_classed = os.path.join(network_plot_parent_dir, f"network_summary_3pannels_classed.pdf")
    #         network_plot_path_2p_classed = os.path.join(network_plot_parent_dir, f"network_summary_2pannels_classed.pdf")
            
    #         #
    #         # unit_types = network_metrics['unit_types']
    #         # print(f"unit_types: {unit_types}")
    #         # import sys
    #         # sys.exit()
    #         #
            
    #         plot_network_metrics(
    #             network_metrics, 
    #             bursting_plot_path, 
    #             bursting_fig_path,
    #             save_path=network_plot_path_3p,
    #             #mode = '2p',
    #             mode = '3p',
    #             limit_seconds = limit_seconds,
    #             plot_class = False,
    #             )
            
    #         plot_network_metrics(
    #             network_metrics, 
    #             bursting_plot_path, 
    #             bursting_fig_path,
    #             save_path=network_plot_path_2p,
    #             mode = '2p',
    #             limit_seconds = limit_seconds,
    #             plot_class = False,
    #             )  
            
    #         plot_network_metrics(
    #             network_metrics, 
    #             bursting_plot_path, 
    #             bursting_fig_path,
    #             save_path=network_plot_path_3p_classed,
    #             #mode = '2p',
    #             mode = '3p',
    #             limit_seconds = limit_seconds,
    #             plot_class = True,
    #             )
            
    #         # #debug
    #         # # print keys in network_metrics
    #         # print(f"Keys in network_metrics:")
    #         # for key in network_metrics.keys():
    #         #     print(f"{key}")
            
    #         # import sys
    #         # sys.exit() 
            
    #         plot_network_metrics(
    #             network_metrics, 
    #             bursting_plot_path, 
    #             bursting_fig_path,
    #             save_path=network_plot_path_2p_classed,
    #             mode = '2p',
    #             limit_seconds = limit_seconds,
    #             plot_class = True,
    #             )  
        # except Exception as e:
        #     print(e)
        #     #print(f"Error: Could not plot network activity for {well_id}") 
        #     traceback.print_exc()
        #     print(f"Error: Could not plot network activity")