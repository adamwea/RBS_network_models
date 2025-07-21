import argparse
import glob
import numpy as np
import os
import json
from multiprocessing import Pool
#from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_network_metrics_v2
from MEA_Analysis.NetworkAnalysis_aw.plot_network_activity import batch_plot

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze and plot network metrics.")
    parser.add_argument('--batch-paths', nargs='+', required=True,
                        help='List of batch directories to search for _metrics.npy files.')
    parser.add_argument('--top-n', type=int, default=16,
                        help='Number of top fitness files to include.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes to use. Set to 1 to disable parallelism.')
    parser.add_argument('--y-lim', type=float, nargs=2, default=[0, 16],
                        help='Y-axis limits for plotting.')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel plotting. Ignored if num-workers == 1.')
    return parser.parse_args()


#def main():
    #debug = False  # 🔁 Flip to True when debugging

    # uncomment this to run in debug mode
    # debug = True  # 🔁 Flip to True when debugging
    # debug_args = [
    #     '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-25_normBRandFRR_optBLandAmps_db'
    # #    '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-20_BRandFRratios',
    # #    '--output_dir',
    # #    '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-20_BRandFRratios/drift',
    #     '--workers', '12',
    # #    '--no_jitter',
    # ]

    
    # args = parse_args()

    # if args.num_workers == 1:
    #     args.parallel = False

    # metrics_files = get_metrics_files(args.batch_paths)
    # selected_files, fitness_vals = sort_by_fitness(metrics_files, args.top_n)

    # print(f"Selected top {len(selected_files)} metrics files based on fitness.")

    # if args.num_workers > 1:
    #     with Pool(processes=args.num_workers) as pool:
    #         npy_list = pool.map(load_metrics_file, selected_files)
    # else:
    #     npy_list = list(map(load_metrics_file, selected_files))

TARGET_DIR = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_FxHET/250715/sensitivity_analysis/run0000'

pkwargs = {
    # paths
    'target_dirs': [TARGET_DIR], # NOTE: can be passed as a list of directories

    #runtime
    'num_workers': 16, 
    'parallel': True,  # enable parallel plotting
    #'parallel': False,  # disable parallel plotting for now
    
    # debug
    # 'limit_load': 5, # if set, limit to load this many .npy files

    # force bounds on axes
    #'y_lim': tuple(args.y_lim)
    'y_lim': (0, 16),
    'x_lim': (10, None), # if set, limit x-axis to this many seconds e.g. (0, 100) or (30, 140) etc.

    # output file types
    'output_types': [
        #'pdf', 
        'png', 
        #'svg'
        ],  # options: 'pdf', 'png', 'svg'

    }


# This block enables the script to be run interactively or as a module
if __name__ == "__main__":
    batch_plot(**pkwargs)
    print("Done plotting network metrics")
