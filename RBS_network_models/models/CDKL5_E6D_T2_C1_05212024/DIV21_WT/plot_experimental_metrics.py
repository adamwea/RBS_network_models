'''
plot network metrics without having to re-run the analysis
'''

from RBS_network_models.extract_features import plot_network_metrics
import numpy as np
import os

#metrics_path = '/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well005/metrics.npy'
#metrics_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-23/gen_7/gen_7_cand_50_metrics.npy'
#metrics_path = '/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well001/metrics.npy'   
# /global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26/gen_4/gen_4_cand_56
metrics_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26/gen_4/gen_4_cand_56_metrics.npy'
# /global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26/gen_0/gen_0_cand_30
metrics_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26/gen_0/gen_0_cand_30_metrics.npy'
#global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26/gen_1/gen_1_cand_111 
metrics_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26/gen_1/gen_1_cand_111_metrics.npy'

#generate paths based on metrics_path
network_plot_path = metrics_path.replace('network_analysis', 'network_plots')
network_plot_parent_dir = os.path.dirname(network_plot_path)
os.makedirs(network_plot_parent_dir, exist_ok=True)
network_plot_path_3p = os.path.join(network_plot_parent_dir, f"network_summary_3pannels.pdf")
network_plot_path_2p = os.path.join(network_plot_parent_dir, f"network_summary_2pannels.pdf")

# Load network metrics
network_metrics = np.load(metrics_path, allow_pickle=True).item()
bursting_plot_path = None
bursting_fig_path = None

try: 
    print("Generating network summary plot...")    
    plot_network_metrics(
        network_metrics, 
        bursting_plot_path, 
        bursting_fig_path,
        save_path=network_plot_path_3p,
        #mode = '2p',
        mode = '3p',
        #limit_seconds = limit_seconds,
        )
    
    plot_network_metrics(
        network_metrics, 
        bursting_plot_path, 
        bursting_fig_path,
        save_path=network_plot_path_2p,
        mode = '2p',
        #limit_seconds = limit_seconds,
        )  
except Exception as e:
    print(e)
    #print(f"Error: Could not plot network activity for {well_id}") 
    
print("Network summary plots generated.")