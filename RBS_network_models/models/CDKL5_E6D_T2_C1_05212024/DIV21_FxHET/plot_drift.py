import argparse
import glob
import json
import os
import re
from multiprocessing import Pool
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Import your evol parameter bounds
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.evol_params import params as evol_params

#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.linear_model import GLSAR

def find_cfg_jsons(batch_paths):
    cfg_jsons = []
    for batch_path in batch_paths:
        cfg_jsons += glob.glob(f'{batch_path}/**/*_cfg.json', recursive=True)

    entries = []
    for fpath in cfg_jsons:
        m = re.search(r'gen_(\d+)_cand_(\d+)', fpath)
        if m:
            gen = int(m.group(1))
            cand = int(m.group(2))
            entries.append((gen, cand, fpath))
    return sorted(entries, key=lambda x: (x[0], x[1]))

def load_cfg_file(entry):
    gen, cand, fpath = entry
    fit = None
    cfg = {}
    try:
        fit_path = fpath.replace('_cfg.json', '_fitness.json')
        with open(fpath, 'r') as f:
            cfg = json.load(f).get('simConfig', {})
        with open(fit_path, 'r') as f:
            fit = json.load(f)['fit']
    except Exception:
        cfg = {}
    filtered = {k: cfg.get(k, np.nan) for k in evol_params.keys() if k in cfg}
    return (gen, cand, fit, filtered)

def group_by_param_and_gen(cfg_entries):
    grouped = {p: {} for p in evol_params.keys()}
    for gen, cand, fit, cfg in cfg_entries:
        for p, val in cfg.items():
            if not np.isnan(val):
                grouped[p].setdefault(gen, []).append(val)
    return grouped

def plot_param_boxplots(grouped, output_dir, jitter=True):
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []

    for param, gen_dict in grouped.items():
        gens = sorted(gen_dict.keys())
        data = [gen_dict[g] for g in gens]
        num_boxes = len(gens)

        #fig, ax = plt.subplots(figsize=(max(6, 0.8 * num_boxes), 4))  # auto-scale width to count
        fig, ax = plt.subplots(figsize=(max(3, 0.5 * num_boxes), 4))  # auto-scale width to count

        # Wider box width (up to 1.0 for full slot), no gaps
        bplot = ax.boxplot(
            data,
            labels=gens,
            patch_artist=True,
            showfliers=False,
            widths=0.90  # max width to fill slot
        )

        # Uniform styling
        for box in bplot['boxes']:
            box.set_facecolor('lightblue')
            box.set_edgecolor('black')
            box.set_linewidth(1.5)

        for whisker in bplot['whiskers']:
            whisker.set_color('black')
        for cap in bplot['caps']:
            cap.set_color('black')

        # Scatter points
        for i, gen in enumerate(gens):
            y = gen_dict[gen]
            if jitter:
                x = np.random.normal(loc=i + 1, scale=0.05, size=len(y))
            else:
                x = np.full_like(y, fill_value=i + 1)
            #ax.scatter(x, y, color='black', s=15, alpha=0.6, edgecolor='white', linewidth=0.5, zorder=3)
            ax.scatter(x, y, color='red', s=15, alpha=0.6, edgecolor='white', linewidth=0.5, zorder=3)
        # Boundaries
        low, high = evol_params[param]
        ax.axhline(low, color='gray', linestyle='--', linewidth=1)
        ax.axhline(high, color='gray', linestyle='--', linewidth=1)

        # Tight layout without x padding
        ax.set_xlim(0.5, num_boxes + 0.5)
        ax.margins(x=0)
        #ax.set_title(param, fontsize=9)
        #ax.set_xlabel('Generation')
        ax.set_ylabel(param)
        ax.grid(True, axis='y')
        fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.15)
        plt.tight_layout()

        # Save
        pdf = os.path.join(output_dir, f"{param}_evolution.pdf")
        png = pdf.replace('.pdf', '.png')
        svg = os.path.join(output_dir, f"{param}_evolution.svg")
        #fig.savefig(pdf)
        fig.savefig(png, dpi=300)
        #fig.savefig(svg)
        plt.close(fig)

        print(f"✅ Saved {param} to:\n  {pdf}\n  {png}")
        plot_paths.append((param, png))

    return plot_paths


def create_summary_page(plot_paths, output_dir):
    summary_path = os.path.join(output_dir, "summary.pdf")
    with PdfPages(summary_path) as pdf:
        plots_per_row = 5
        total = len(plot_paths)
        rows = int(np.ceil(total / plots_per_row))

        fig_width, fig_height = 11.69, 8.27  # A4 landscape in inches
        fig = plt.figure(figsize=(fig_width, fig_height))
        for idx, (param, path) in enumerate(plot_paths):
            img = plt.imread(path)
            ax = fig.add_subplot(rows, plots_per_row, idx + 1)
            ax.imshow(img)
            #ax.set_title(param, fontsize=8)
            ax.axis('off')

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return summary_path

def plot_single_param_inline(param, gen_dict, ax, jitter=True):
    gens = sorted(gen_dict.keys())
    data = [gen_dict[g] for g in gens]

    #bplot = ax.boxplot(data, labels=gens, patch_artist=True, showfliers=False, widths=0.85)
    # white box, black edge, black median
    bplot = ax.boxplot(
        data,
        labels=gens,
        patch_artist=True,
        showfliers=False,
        widths=0.85
    )

    for box in bplot['boxes']:
        box.set_facecolor('#cccccc')  # medium-light grey
        box.set_edgecolor('black')
        box.set_linewidth(1.5)

    for median in bplot['medians']:
        median.set_color('black')
        median.set_linewidth(1)
        median.set_linestyle('--')

    for whisker in bplot['whiskers']:
        whisker.set_color('black')

    for cap in bplot['caps']:
        cap.set_color('black')

    # for i, gen in enumerate(gens):
    #     y = gen_dict[gen]
    #     x = np.random.normal(loc=i + 1, scale=0.05, size=len(y)) if jitter else np.full_like(y, i + 1)
    #     #ax.scatter(x, y, color='red', s=8, alpha=0.6, edgecolor='white', linewidth=0.5, zorder=3)
    #     ax.scatter(x, y, color='red', s=8, alpha=0.6, edgecolor='white', linewidth=0.5)


    # plot trendline and test if trend is significant 
    
    # METHOD 1 - there are some poor assuptions here
    # this assumes that the data is normally distributed and that the errors are homoscedastic
    # and that each data point is independent of the others - at least in this case, that's definitely not true
    # and 1st order trend line
    #all_data = np.concatenate(data)
    # all_gens = np.concatenate([[g] * len(gen_dict[g]) for g in gens])
    # all_data = np.concatenate(data)
    # coeffs = np.polyfit(all_gens, all_data, 1)
    # trendline = np.polyval(coeffs, all_gens)
    # adjusted_gens = all_gens+1
    # ax.plot(adjusted_gens, trendline, color='red', 
    # #linestyle='--', 
    # linewidth=1)

    # # test if trend is significant, add asterisk above plot if so.
    # from scipy.stats import linregress
    # slope, intercept, r_value, p_value, std_err = linregress(all_gens, all_data)
    
    # if p_value < 0.001:
    #     # add asterisk above the plot
    #     ax.text(0.5, 1.15, f'***', transform=ax.transAxes, fontsize=14, ha='center', va='top', color='red')
    # elif p_value < 0.01:
    #     # add asterisk above the plot
    #     ax.text(0.5, 1.15, f'**', transform=ax.transAxes, fontsize=14, ha='center', va='top', color='red')
    # elif p_value < 0.05:
    #     # add asterisk above the plot
    #     ax.text(0.5, 1.15, f'*', transform=ax.transAxes, fontsize=14, ha='center', va='top', color='red')

    # METHOD 2 - this method might be better. Assumes possible autocorrelation between generations
    # still assumes that the data is normally distributed and that the errors are homoscedastic
    # Prepare data
    all_gens = np.concatenate([[g] * len(gen_dict[g]) for g in gens])
    all_data = np.concatenate(data)
    adjusted_gens = all_gens + 1  # for visualization purposes

    # Convert to DataFrame
    df = pd.DataFrame({
        'gen': all_gens,
        'value': all_data
    })

    # Fit GLSAR model (1st order autoregression)
    model = GLSAR(df['value'], sm.add_constant(df['gen']), rho=1)
    res = model.iterative_fit(maxiter=10)

    # Predict trendline
    df['trendline'] = res.predict(sm.add_constant(df['gen']))

    # Plotting
    #fig, ax = plt.subplots()
    #ax.plot(adjusted_gens, df['value'], 'o', alpha=0.5)
    ax.plot(adjusted_gens, df['trendline'], color='red', linewidth=1)

    # Test significance and add annotation
    p_value = res.pvalues[1]
    if p_value < 0.001:
        ax.text(0.5, 1.15, '***', transform=ax.transAxes, fontsize=14, ha='center', va='top', color='red')
    elif p_value < 0.01:
        ax.text(0.5, 1.15, '**', transform=ax.transAxes, fontsize=14, ha='center', va='top', color='red')
    elif p_value < 0.05:
        ax.text(0.5, 1.15, '*', transform=ax.transAxes, fontsize=14, ha='center', va='top', color='red')

    
    # Boundaries    
    low, high = evol_params[param]
    ax.axhline(low, color='gray', linestyle='--', linewidth=1)
    ax.axhline(high, color='gray', linestyle='--', linewidth=1)

    #ax.set_title(param, fontsize=8)
    #ax.set_xlabel('Gen')
    # update y-tick fontsize
    ax.tick_params(axis='y', labelsize=6)

    ax.set_ylabel(param, fontsize=7)
    ax.set_xlim(0.5, len(gens) + 0.5)
    ax.margins(x=0)
    ax.grid(True, axis='y')

def create_summary_page_inline(grouped, output_dir, jitter=True):
    summary_path = os.path.join(output_dir, "summary.pdf")
    os.makedirs(output_dir, exist_ok=True)

    params = list(grouped.keys())
    plots_per_row = 6
    rows = int(np.ceil(len(params) / plots_per_row))

    fig_width, fig_height = 11.69, 8.27  # A4 landscape
    fig = plt.figure(figsize=(fig_width, fig_height))

    for idx, param in enumerate(params):
        ax = fig.add_subplot(rows, plots_per_row, idx + 1)
        plot_single_param_inline(param, grouped[param], ax, jitter=jitter)
    
    # for all subplots except last row, remove x ticks
    for i in range(0, len(params) - plots_per_row):
        ax = fig.axes[i]
        ax.set_xticks([])
        ax.set_xticklabels([])
        
    fig.tight_layout()
    png_path = os.path.join(output_dir, "summary.png")
    fig.savefig(png_path, dpi=300)
    print(f"✅ Saved summary image to: {png_path}")

    with PdfPages(summary_path) as pdf:
        pdf.savefig(fig)
        plt.close(fig)

    return summary_path

#TODO: get n_elites from batchConfig or something...
def distribute_elites(cfg_entries, population_size=None, n_elites=32):
    from collections import defaultdict

    # Map gen → list of (cand, full_fit_list, fit_score)
    gens = defaultdict(list)

    #
    #isinstance(cfg_entries, list)

    for gen, cand, fit, cfg in cfg_entries:
        # Find the score where path == 'fit'
        #fit_score = next((fit for path, fit in fit_list if path == 'fit'), None)
        if fit is None:
            continue  # skip if no fitness score
        gens[gen].append((cand, cfg, fit))

    # population size should = len(gens[0])
    population_size = len(gens[0])
    
    sorted_gens = sorted(gens.keys())
    new_entries = []
    elites = []

    for i, gen in enumerate(sorted_gens):
        current_gen = gens[gen]

        # print len of current gen for debugging
        print(f"Gen {gen}: {len(current_gen)} candidates")
        #gen_pop = len(current_gen)

        # Inject elites into this generation (except for gen 0)
        if i > 0 and elites:
            # Remove worst entries to make space for elites
            current_gen = sorted(current_gen, key=lambda x: x[2])  # sort by fit_score
            #current_gen = current_gen[:len(current_gen) - len(elites)] + elites
            current_gen = current_gen[:population_size - len(elites)] + elites

        # Recompute elites from the combined generation
        current_gen_sorted = sorted(current_gen, key=lambda x: x[2])

        # assert adjusted gen has pop size of population_size
        if len(current_gen_sorted) != population_size:
            print(f"❌ Adjusted gen {gen} does not have population size of {population_size}.")
            print(f"Current gen size: {len(current_gen_sorted)}")
            print(f"NOTE: ignore this if this is the last gen.")
            #continue

        elites = current_gen_sorted[:n_elites]

        # Add full fitness data to the new list
        for cand, cfg, fit in current_gen:
            new_entries.append((gen, cand, fit, cfg))

    return new_entries

def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Plot parameter evolution from batch cfg files")
    parser.add_argument('batch_paths', nargs='+', help='One or more paths to batch directories')
    parser.add_argument('--output_dir', default=None, help='Path to output plots (defaults to first batch path)')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes (default: 8)')
    parser.add_argument('--no_jitter', action='store_true', help='Disable scatter jitter')

    # arg_list=None means: use sys.argv
    return parser.parse_args(arg_list)

def main():
    debug = False  # 🔁 Flip to True when debugging

    # uncomment this to run in debug mode
    # debug = True  # 🔁 Flip to True when debugging
    # debug_args = [
    #     '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-20_BRandFRratios',
    #     '--output_dir',
    #     '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-20_BRandFRratios/drift',
    #     '--workers', '12',
    #     '--no_jitter',
    # ]

    args = parse_args(debug_args if debug else None)

    output_dir = args.output_dir or args.batch_paths[0]
    cfg_entries = find_cfg_jsons(args.batch_paths)

    print(f"📂 Found {len(cfg_entries)} config files.")
    with Pool(processes=args.workers) as pool:
        cfg_entries = pool.map(load_cfg_file, cfg_entries)

    cfg_entries = distribute_elites(cfg_entries)
    grouped = group_by_param_and_gen(cfg_entries)
    #plot_paths = plot_param_boxplots(grouped, output_dir, jitter=not args.no_jitter)
    #summary = create_summary_page(plot_paths, output_dir)
    summary = create_summary_page_inline(grouped, output_dir, jitter=not args.no_jitter)

    print(f"✅ All individual and summary plots saved to: {output_dir}")
    print(f"📄 Summary PDF: {summary}")
    print(f"📂 Output directory: {output_dir}")

if __name__ == "__main__":
    main()

