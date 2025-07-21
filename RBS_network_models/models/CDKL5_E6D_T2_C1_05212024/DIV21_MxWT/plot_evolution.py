import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from PIL import Image
from scipy.stats import norm
import traceback
import argparse
import os
import traceback
from multiprocessing import Pool
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import GLSAR
import os


# batch_paths = [
#     #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-15/'

#     # aw 2025-05-16 11:59:16
#     #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-16/'

#     # aw 2025-05-19 10:24:20 
#     #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-18/'

#     # aw2025-05-19 19:58:06
#     #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-19_BRandFRs/'

#     # aw 2025-05-19 22:52:52
#     #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-19_BRandFRratios/'

#     # aw 2025-05-20 21:31:11 
#     #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-20_BRandFRratios/',

#     #
#     #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-23_normBRandFRR_optBLandAmps/',
#     '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-23_normBRandFRR_optBLandAmps_2/',
# ]

def find_fitness_jsons(batch_paths):
    fitness_jsons = []
    for batch_path in batch_paths:
        fitness_jsons += glob.glob(f'{batch_path}/**/*_fitness.json', recursive=True)

    json_info = []
    for fpath in fitness_jsons:
        match = re.search(r'gen_(\d+)_cand_(\d+)', fpath)
        if match:
            gen = int(match.group(1))
            cand = int(match.group(2))
            json_info.append((gen, cand, fpath))
        else:
            print(f'❌ No match for: {fpath}')
    return sorted(json_info, key=lambda x: (x[0], x[1]))

def load_fit_file(entry):
    gen, cand, fpath = entry
    fit_values = []
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
            fit_values = extract_fitness_paths(data)
    except Exception as e:
        print(f'⚠️ Failed to load {fpath}: {e}')
    return (gen, cand, fit_values)

def extract_fitness_paths(data, prefix=""):
    results = []
    if isinstance(data, dict):
        for key, val in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(val, (int, float)):
                results.append((new_prefix, val))
            else:
                results.extend(extract_fitness_paths(val, new_prefix))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            list_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            results.extend(extract_fitness_paths(item, list_prefix))
    return results

def distribute_elites(fitness_entries, n_elites=32, population_size=64):
    from collections import defaultdict

    # Map gen → list of (cand, full_fit_list, fit_score)
    gens = defaultdict(list)

    for gen, cand, fit_list in fitness_entries:
        # Find the score where path == 'fit'
        fit_score = next((fit for path, fit in fit_list if path == 'fit'), None)
        if fit_score is None:
            continue  # skip if no fitness score
        gens[gen].append((cand, fit_list, fit_score))

    sorted_gens = sorted(gens.keys())
    new_fitness_entries = []
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
        for cand, fit_list, _ in current_gen:
            new_fitness_entries.append((gen, cand, fit_list))

    return new_fitness_entries

def filter_fitness_entries(fitness_entries, threshold=1000):
    from collections import defaultdict

    filtered = []
    high_by_path = defaultdict(lambda: defaultdict(list))
    all_gens = set()

    for gen, cand, fit_list in fitness_entries:
        all_gens.add(gen)
        filtered_fit_list = []
        for path, fit in fit_list:
            # get list of keys and assert that 'fit' is the last key, else skip
            keys = path.split('.')
            if keys[-1] != 'fit':
                #print(f"❌ Skipping {path} because it does not end with 'fit'")
                continue
            
            if fit >= threshold:
                high_by_path[path][gen].append(fit)
            else:
                filtered_fit_list.append((path, fit))
        if filtered_fit_list:
            filtered.append((gen, cand, filtered_fit_list))

    # Fill in empty lists for generations that didn't appear
    for path in high_by_path:
        for gen in all_gens:
            high_by_path[path].setdefault(gen, [])

    return filtered, dict(high_by_path)

def group_by_generation(fitness_entries):
    grouped = {}
    for gen, cand, fit_list in fitness_entries:
        for path, fit in fit_list:
            grouped.setdefault(path, {}).setdefault(gen, []).append(fit)
    return grouped

def plot_histogram_with_gaussian(grouped_fits, output_dir, tag="default"):
    all_fitness = [fit for gen_fits in grouped_fits.values() for fit in gen_fits]
    filename = f"fitness_histogram_{tag.replace('.', '_')}.png"
    hist_path = os.path.join(output_dir, filename)
    plt.figure(figsize=(12, 4))
    n, bins, patches = plt.hist(all_fitness, bins=50, color='gray', alpha=0.7, edgecolor='black')
    mu, std = norm.fit(all_fitness)
    x = np.linspace(min(bins), max(bins), 1000)
    p = norm.pdf(x, mu, std) * (len(all_fitness) * (bins[1] - bins[0]))
    plt.plot(x, p, 'r--', linewidth=2, label=f'Gaussian Fit (μ={mu:.2f}, σ={std:.2f})')
    plt.title('Distribution of Candidate Fitness')
    plt.xlabel('Fitness')
    plt.ylabel('Count')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()
    return hist_path

def plot_high_fit_histogram(high_entries, path_key, output_dir):
    high_by_gen = high_entries.get(path_key, {})
    gens = sorted(high_by_gen.keys())
    counts = [len(high_by_gen[gen]) for gen in gens]

    # remove last gen - it's probably incomplete
    counts = counts[:-1]
    gens = gens[:-1]

    plt.figure(figsize=(12, 3))
    plt.bar(gens, counts, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel("Generation")
    plt.ylabel("Count")
    plt.title("Fitness ≥ 1000 Count by Generation")
    plt.tight_layout()
    filename = f"high_fitness_count_{path_key.replace('.', '_')}.png"
    hist_path = os.path.join(output_dir, filename)
    plt.savefig(hist_path, dpi=300)
    plt.close()
    return hist_path

def plot_fitness_boxplots(grouped_fits, output_dir, tag="default"):
    gens = sorted(grouped_fits.keys())
    data = [grouped_fits[gen] for gen in gens]

    plt.figure(figsize=(12, 3))
    bplot = plt.boxplot(data, labels=gens, patch_artist=True, showfliers=False)

    for i, box in enumerate(bplot['boxes']):
        box.set_facecolor((0.2, 0.4, 0.6, 0.3))
        box.set_edgecolor('black')
        box.set_linewidth(1.5)

    for whisker in bplot['whiskers']:
        whisker.set_color('black')
    for cap in bplot['caps']:
        cap.set_color('black')

    all_gen, all_fit = [], []
    for gen in gens:
        # # if last gen - just continue, probably incomplete
        # if gen == max(gens):
        #     continue

        all_gen.extend([gen] * len(grouped_fits[gen]))
        all_fit.extend(grouped_fits[gen])

    # remove last gen - it's probably incomplete
    #all_gen = all_gen[:-1]

    # Fit a 3rd order polynomial to the data
    # coeffs = np.polyfit(all_gen, all_fit, 3)
    # poly = np.poly1d(coeffs)
    # x_fit = np.linspace(min(gens), max(gens), 300)
    # y_fit = poly(x_fit)
    # plt.plot(x_fit + 1, y_fit, 'b--', linewidth=2, label='3rd Order Trend')

    #fit a 2nd order polynomial to the data
    coeffs = np.polyfit(all_gen, all_fit, 2)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(gens), max(gens), 300)
    y_fit = poly(x_fit)
    plt.plot(x_fit + 1, y_fit, 'b--', linewidth=2, label='2nd Order Trend')


    min_fit_by_gen = {gen: min(grouped_fits[gen]) for gen in gens if grouped_fits[gen]}
    sorted_gens = sorted(min_fit_by_gen)
    values = [min_fit_by_gen[gen] for gen in sorted_gens]
    # local_minima = [(sorted_gens[i], values[i]) for i in range(1, len(values) - 1)
    #                 if values[i] < values[i - 1] and values[i] < values[i + 1]]

    # loop through the values and find local minima in chronological order
    local_minima = []
    minima = None
    for i in range(0, len(values) - 1):
        if minima is None or values[i] < minima:
            minima = values[i]
            local_minima.append((sorted_gens[i], values[i]))

    top_n = 32
    local_elites = []
    new_elites = []

    for gen in sorted(grouped_fits.keys()):
        fits = grouped_fits[gen]
        sorted_fits = sorted(fits)

        for val in sorted_fits:
            # Fill the elite pool first
            if len(local_elites) < top_n:
                local_elites.append((gen, val))
                new_elites.append((gen, val))
            else:
                current_max = max(local_elites, key=lambda x: x[1])
                if val < current_max[1] and (gen, val) not in local_elites:
                    local_elites.append((gen, val))
                    local_elites = sorted(local_elites, key=lambda x: x[1])[:top_n]
                    new_elites.append((gen, val))




    # local_minima.insert(0, (sorted_gens[0], values[0]))
    # local_minima.append((sorted_gens[-1], values[-1]))

    for gen, val in new_elites:
        # dont plot gen 0
        if gen == 0:
            continue
        #plt.plot(gen + 1, val, marker='*', color='gold', markersize=10, label='New Elite')
        # do orange instead
        plt.plot(gen + 1, val, marker='*', color='orange', markersize=10, label='New Elite')

    first = True
    for gen, val in local_minima:
        label = 'Local Minima' if first else None
        plt.plot(gen + 1, val, 'r*', markersize=12, label=label)
        first = False

    plt.title('Candidate Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True, axis='y')
    plt.legend()

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())

    plt.tight_layout()
    filename = f"fitness_evolution_plot_{tag.replace('.', '_')}.png"
    png_path = os.path.join(output_dir, filename)
    plt.savefig(png_path, dpi=300)
    print(f"✅ Fitness plot saved to: {png_path}")
    plt.close()
    return png_path, coeffs

def plot_fitness_violinplots(grouped_fits, output_dir, tag="default"):
    gens = sorted(grouped_fits.keys())
    data = [grouped_fits[gen] for gen in gens]

    plt.figure(figsize=(12, 6))
    parts = plt.violinplot(data, positions=range(1, len(gens)+1), showmeans=True, showmedians=False)

    for pc in parts['bodies']:
        pc.set_facecolor((0.2, 0.4, 0.6, 0.3))
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
        vp = parts.get(partname)
        if vp:
            vp.set_color('black')
            vp.set_linewidth(1.5)

    all_gen, all_fit = [], []
    for gen in gens:
        all_gen.extend([gen] * len(grouped_fits[gen]))
        all_fit.extend(grouped_fits[gen])

    # Fit a 3rd order polynomial to the data
    # coeffs = np.polyfit(all_gen, all_fit, 3)
    # poly = np.poly1d(coeffs)
    # x_fit = np.linspace(min(gens), max(gens), 300)
    # y_fit = poly(x_fit)
    # plt.plot(x_fit + 1, y_fit, 'b--', linewidth=2, label='3rd Order Trend')


    # Fit a 2nd order polynomial to the data
    coeffs = np.polyfit(all_gen, all_fit, 2)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(gens), max(gens), 300)
    y_fit = poly(x_fit)
    plt.plot(x_fit + 1, y_fit, 'b--', linewidth=2, label='2nd Order Trend')

    min_fit_by_gen = {gen: min(grouped_fits[gen]) for gen in gens if grouped_fits[gen]}
    sorted_gens = sorted(min_fit_by_gen)
    values = [min_fit_by_gen[gen] for gen in sorted_gens]
    local_minima = []
    minima = None
    for i in range(len(values)):
        if minima is None or values[i] < minima:
            minima = values[i]
            local_minima.append((sorted_gens[i], values[i]))

    first = True
    for gen, val in local_minima:
        label = 'Local Minima' if first else None
        plt.plot(gen + 1, val, 'r*', markersize=10, label=label)
        first = False

    plt.xticks(range(1, len(gens)+1), gens)
    plt.title('Candidate Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    filename = f"fitness_evolution_plot_{tag.replace('.', '_')}.png"
    png_path = os.path.join(output_dir, filename)
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✅ Fitness plot saved to: {png_path}")
    return png_path, coeffs

def sort_metric_keys(keys):
    def key_func(k):
        parts = k.split('.')
        # Group by immediate parent (everything except the last part)
        parent_path = '.'.join(parts[:-1])
        return (parent_path, parts)
    
    return sorted(keys, key=key_func)

def add_cover_page_with_links(c, page_titles):
    """
    Adds a cover page with clickable links to each report page in two columns.
    Also returns a function to add a 'Back to Index' link on other pages.

    Parameters:
        c (Canvas): ReportLab canvas object.
        page_titles (list of tuples): List of (bookmark_name, display_title).

    Returns:
        add_back_link: function to add "Back to Index" footer on each page.
    """
    width, height = c._pagesize

    # Index title
    c.bookmarkPage("index")
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 1.0 * inch, "Fitness Evolution Report Index")

    # Prepare layout parameters
    num_links = len(page_titles)
    num_rows = (num_links + 1) // 2
    available_height = height - 1.5 * inch - 0.5 * inch
    row_height = available_height / max(1, num_rows)

    col_x_positions = [1.0 * inch, width / 2 + 0.2 * inch]
    column_width = (width / 2) - 1.2 * inch
    y_start = height - 1.5 * inch

    # Determine max font size that fits all titles horizontally
    font_size = min(12, int(row_height * 0.8))
    while font_size > 5:
        too_wide = any(
            stringWidth(title, "Helvetica", font_size) > column_width
            for _, title in page_titles
        )
        if too_wide:
            font_size -= 1
        else:
            break

    # Draw each title
    c.setFont("Helvetica", font_size)
    for i, (bookmark, title) in enumerate(page_titles):
        col = i // num_rows
        row = i % num_rows
        x = col_x_positions[col]
        y = y_start - row * row_height
        c.drawString(x, y, title)
        c.linkRect('', bookmark,
                   Rect=(x, y - 2, x + column_width, y + font_size + 2),
                   relative=0, thickness=0)

    c.showPage()

    # Return back link adder
    def add_back_link():
        c.setFont("Helvetica-Oblique", 8)
        back_text = "Back to Index"
        link_y = 0.4 * inch
        c.drawString(1 * inch, link_y, back_text)
        c.linkRect('', 'index', Rect=(1 * inch, link_y - 2, 2.5 * inch, link_y + 10), relative=0, thickness=0)

    return add_back_link

def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Generate fitness evolution PDF reports.")
    parser.add_argument('batch_paths', nargs='+', help='One or more paths to batch directories')
    parser.add_argument('--output_dir', default=None, help='Optional path to save output files')
    parser.add_argument('--workers', type=int, default=16, help='Number of worker processes (default: 16)')
    #return parser.parse_args()

    return parser.parse_args(arg_list)

def plot_all_fitness_boxplots_inline_dep(grouped_paths, output_dir, sorted_paths):
    """
    Generate a single A4-sized summary page with all fitness metric boxplots, 
    using consistent styles and statistical trendlines with significance annotations.
    """
    #sorted_paths = sort_metric_keys(grouped_paths.keys())
    num_plots = len(sorted_paths)
    plots_per_row = 6
    rows = int(np.ceil(num_plots / plots_per_row))

    fig_width, fig_height = 11.69, 8.27  # A4 landscape in inches
    fig = plt.figure(figsize=(fig_width, fig_height))

    for i, path in enumerate(sorted_paths):
        ax = fig.add_subplot(rows, plots_per_row, i + 1)
        gens = sorted(grouped_paths[path].keys())
        data = [grouped_paths[path][g] for g in gens]

        # Boxplot
        bplot = ax.boxplot(data, labels=gens, patch_artist=True, showfliers=False, widths=0.85)
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

        # Flattened data for GLSAR trendline
        all_gen = np.concatenate([[g] * len(grouped_paths[path][g]) for g in gens])
        all_fit = np.concatenate(data)
        df = pd.DataFrame({'gen': all_gen, 'value': all_fit})
        df['gen_plus1'] = df['gen'] + 1  # for visualization

        # GLSAR model
        model = GLSAR(df['value'], sm.add_constant(df['gen']), rho=1)
        res = model.iterative_fit(maxiter=10)
        df['trend'] = res.predict(sm.add_constant(df['gen']))

        ax.plot(df['gen_plus1'], df['trend'], color='red', linewidth=1)

        # Significance stars
        p_val = res.pvalues[1]
        if p_val < 0.001:
            stars = '***'
        elif p_val < 0.01:
            stars = '**'
        elif p_val < 0.05:
            stars = '*'
        else:
            stars = ''
        if stars:
            ax.text(0.5, 1.15, stars, transform=ax.transAxes,
                    ha='center', va='top', fontsize=14, color='red')

        #ax.set_title(path.split('.')[-1], fontsize=8)
        #ax.set_title(path, fontsize=5)

        # set title as y-axis label
        ax.set_ylabel(path, fontsize=6)
        ax.tick_params(axis='x', labelrotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(True, axis='y')

    fig.tight_layout()
    output_path = os.path.join(output_dir, "summary_all_fitness_boxplots.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✅ Summary boxplots saved to: {output_path}")
    return output_path

def plot_all_fitness_boxplots_inline_dep2(grouped_paths, output_dir, sorted_paths):
    """
    Generate a single A4-sized summary page with all fitness metric boxplots,
    using consistent styles and statistical trendlines with significance annotations.
    'fit' spans 2x2; new rows start with new group prefix.
    Dynamically allocates GridSpec size based on number of plots.
    """
    import matplotlib.gridspec as gridspec

    max_cols = 6
    group_keys = [p.split('.')[0] for p in sorted_paths if p != 'fit']
    group_changes = 1 + sum(g1 != g2 for g1, g2 in zip(group_keys, group_keys[1:]))
    total_cells = len(sorted_paths) - 1 + 4  # minus 'fit', plus 4 for its 2x2
    est_rows = int(np.ceil(total_cells / max_cols)) + group_changes

    fig_width, fig_height = 11.69, 2 + 1.5 * est_rows  # Auto-scale height
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(est_rows, max_cols, figure=fig, wspace=0.6, hspace=0.6)

    def run_glsar_and_plot(ax, gens, data):
        all_gen = np.concatenate([[g] * len(data[i]) for i, g in enumerate(gens)])
        all_fit = np.concatenate(data)
        df = pd.DataFrame({'gen': all_gen, 'value': all_fit})
        df['gen_plus1'] = df['gen'] + 1
        model = GLSAR(df['value'], sm.add_constant(df['gen']), rho=1)
        res = model.iterative_fit(maxiter=10)
        df['trend'] = res.predict(sm.add_constant(df['gen']))
        ax.plot(df['gen_plus1'], df['trend'], color='red', linewidth=1)
        p_val = res.pvalues[1]
        stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        if stars:
            ax.text(0.5, 1.15, stars, transform=ax.transAxes, ha='center', va='top', fontsize=14, color='red')

    row, col = 0, 0
    prev_group = None
    used_slots = np.zeros((est_rows, max_cols), dtype=bool)

    def find_next_available_cell(start_row=0):
        for r in range(start_row, est_rows):
            for c in range(max_cols):
                if not used_slots[r, c]:
                    return r, c
        raise RuntimeError("GridSpec ran out of space")

    for path in sorted_paths:

        group_key = path.split('.')[0]
        is_fit = (path == 'fit')

        # replace all underscores with . 
        #path = path.replace('_', '.')

        gens = sorted(grouped_paths[path].keys())
        data = [grouped_paths[path][g] for g in gens]

        # remove .fit from path for, it's implied
        path = path.replace('.fit', '')

        if is_fit:
            #ax = fig.add_subplot(gs[0:2, 0:2])
            #used_slots[0:2, 0:2] = True
            ax = fig.add_subplot(gs[0:2, 0:max_cols])
            used_slots[0:2, 0:max_cols] = True
            # start a new row after 'fit'
        else:
            if prev_group and group_key != prev_group:
                row += 1
                col = 0
            row, col = find_next_available_cell(row)
            ax = fig.add_subplot(gs[row, col])
            used_slots[row, col] = True

        bplot = ax.boxplot(data, labels=gens, patch_artist=True, showfliers=False, widths=0.85)
        for box in bplot['boxes']:
            box.set_facecolor('#cccccc')
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

        run_glsar_and_plot(ax, gens, data)

        #ax.set_ylabel(path, fontsize=6)
        # remove group prefix from label
        shortpath = path.replace(f"{group_key}.", "")
        # set title as y-axis label
        ax.set_ylabel(shortpath, fontsize=6)
        
        # if prev_group != group_key: bold the first label of each group
        if prev_group != group_key and not is_fit:
            ax.set_ylabel(path, fontsize=6, fontweight='bold')
        elif prev_group != group_key and is_fit:
            ax.set_ylabel('Overall Fitness', fontsize=6, fontweight='bold')
        ax.tick_params(axis='x', labelrotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(True, axis='y')

        prev_group = group_key

    fig.tight_layout()
    output_path = os.path.join(output_dir, "summary_all_fitness_boxplots.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✅ Summary boxplots saved to: {output_path}")
    return output_path

# Re-import required libraries after kernel reset
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.linear_model import GLSAR
from matplotlib.colors import to_rgba
from matplotlib import gridspec

# Re-define the updated plotting function
def plot_all_fitness_boxplots_inline(grouped_paths, output_dir, sorted_paths):
    """
    Generate a single A4-sized summary page with all fitness metric boxplots,
    using consistent styles and statistical trendlines with significance annotations.
    'fit' spans 2x2; new rows start with new group prefix.
    Labels are indented and shaded by nesting depth.
    """
    def get_depth(path):
        return len(path.split('.'))

    depths = {p: get_depth(p) for p in sorted_paths}
    max_depth = max(depths.values())

    max_cols = 6
    group_keys = [p.split('.')[0] for p in sorted_paths if p != 'fit']
    group_changes = 1 + sum(g1 != g2 for g1, g2 in zip(group_keys, group_keys[1:]))
    total_cells = len(sorted_paths) - 1 + 4
    est_rows = int(np.ceil(total_cells / max_cols)) + group_changes

    fig_width, fig_height = 11.69, 2 + 1.5 * est_rows
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(est_rows, max_cols, figure=fig, wspace=0.6, hspace=0.6)

    def run_glsar_and_plot(ax, gens, data):
        all_gen = np.concatenate([[g] * len(data[i]) for i, g in enumerate(gens)])
        all_fit = np.concatenate(data)
        df = pd.DataFrame({'gen': all_gen, 'value': all_fit})
        df['gen_plus1'] = df['gen'] + 1
        model = GLSAR(df['value'], sm.add_constant(df['gen']), rho=1)
        res = model.iterative_fit(maxiter=10)
        df['trend'] = res.predict(sm.add_constant(df['gen']))
        ax.plot(df['gen_plus1'], df['trend'], color='red', linewidth=1)
        p_val = res.pvalues[1]
        stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        if stars:
            ax.text(0.5, 1.15, stars, transform=ax.transAxes, ha='center', va='top', fontsize=14, color='red')

    row, col = 0, 0
    prev_group = None
    used_slots = np.zeros((est_rows, max_cols), dtype=bool)

    def find_next_available_cell(start_row=0):
        for r in range(start_row, est_rows):
            for c in range(max_cols):
                if not used_slots[r, c]:
                    return r, c
        raise RuntimeError("GridSpec ran out of space")

    for path in sorted_paths:
        group_key = path.split('.')[0]
        is_fit = (path == 'fit')
        gens = sorted(grouped_paths[path].keys())
        data = [grouped_paths[path][g] for g in gens]

        if is_fit:
            ax = fig.add_subplot(gs[0:2, 0:2])
            used_slots[0:2, 0:2] = True
        else:
            if prev_group and group_key != prev_group:
                row += 1
                col = 0
            row, col = find_next_available_cell(row)
            ax = fig.add_subplot(gs[row, col])
            used_slots[row, col] = True

        bplot = ax.boxplot(data, labels=gens, patch_artist=True, showfliers=False, widths=0.85)

        for box in bplot['boxes']:
            depth = depths[path]
            alpha = 0.2 + 0.8 * (depth / max_depth)
            box.set_facecolor(to_rgba('#cccccc', alpha))
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

        run_glsar_and_plot(ax, gens, data)

        indent = '  ' * (depths[path] - 1)
        ax.set_ylabel(f"{indent}{path}", fontsize=6)
        ax.tick_params(axis='x', labelrotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(True, axis='y')

        prev_group = group_key

    fig.tight_layout()
    output_path = os.path.join(output_dir, "summary_all_fitness_boxplots.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✅ Summary boxplots saved to: {output_path}")
    return output_pat

def plot_all_fitness_boxplots_columnwise(grouped_paths, output_dir, sorted_paths):
    """
    Generate a summary page of fitness boxplots arranged in vertical columns per top-level group.
    - Overall 'fit' spans full top row.
    - Each top-level metric starts a new column.
    - Plots are indented and shaded by path depth.
    """
    import matplotlib.gridspec as gridspec

    def get_depth(path): return len(path.split('.'))
    def get_group(path): return path.split('.')[0]

    # Extract plotting metadata
    paths = [p for p in sorted_paths if p != 'fit']
    groups = {}
    for p in paths:
        grp = get_group(p)
        groups.setdefault(grp, []).append(p)

    max_depth = max(get_depth(p) for p in paths)
    col_count = len(groups)
    max_rows = max(len(v) for v in groups.values()) + 2  # +2 for fit and spacing
    row_count = max(len(v) for v in groups.values()) + 1  # +1 for padding under fit

    fig_width = 2 + 2.2 * col_count
    fig_height = 2 + 1.4 * max_rows
    fig = plt.figure(figsize=(fig_width, fig_height))
    #gs = gridspec.GridSpec(max_rows, col_count, figure=fig, hspace=0.6, wspace=0.8)
    gs = gridspec.GridSpec(row_count + 1, col_count, height_ratios=[0.7] + [1]*row_count, figure=fig, hspace=0.6, wspace=0.4)

    def run_glsar_and_plot(ax, gens, data):
        all_gen = np.concatenate([[g] * len(data[i]) for i, g in enumerate(gens)])
        all_fit = np.concatenate(data)
        df = pd.DataFrame({'gen': all_gen, 'value': all_fit})
        df['gen_plus1'] = df['gen'] + 1
        model = GLSAR(df['value'], sm.add_constant(df['gen']), rho=1)
        res = model.iterative_fit(maxiter=10)
        df['trend'] = res.predict(sm.add_constant(df['gen']))
        ax.plot(df['gen_plus1'], df['trend'], color='red', linewidth=1)
        p_val = res.pvalues[1]
        stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        if stars:
            #ax.text(0.5, 1.35, stars, transform=ax.transAxes, ha='center', va='top', fontsize=14, color='red')
            # place stars to the right of the plot instead of above
            ax.text(1.05, 0.5, stars, transform=ax.transAxes,
                    ha='left', va='center', fontsize=14, color='red')

    # Plot the 'fit' metric across the top row
    if 'fit' in sorted_paths:
        gens = sorted(grouped_paths['fit'].keys())
        data = [grouped_paths['fit'][g] for g in gens]
        ax = fig.add_subplot(gs[0:2, :])
        bplot = ax.boxplot(data, labels=gens, patch_artist=True, showfliers=False, widths=0.85)
        for box in bplot['boxes']:
            #box.set_facecolor(to_rgba('#cccccc', 0.3))
            box.set_facecolor('#cccccc')  # consistent light grey
            box.set_edgecolor('black')
            box.set_linewidth(1.5)
        for median in bplot['medians']:
            median.set_color('black')
            median.set_linewidth(1)
            median.set_linestyle('--')
        for whisker in bplot['whiskers']: whisker.set_color('black')
        for cap in bplot['caps']: cap.set_color('black')

        run_glsar_and_plot(ax, gens, data)
        ax.set_title('Overall Fitness', fontsize=7, fontweight='bold')
        #ax.set_ylabel("Overall Fitness", fontsize=7, fontweight='bold')
        ax.tick_params(axis='x', labelrotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(True, axis='y')

    # Now plot all grouped metrics in columns
    for col_idx, (grp, paths_in_group) in enumerate(groups.items()):
       # for row_offset, path in enumerate(sorted(paths_in_group, key=get_depth)):
        for row_offset, path in enumerate(paths_in_group):
            gens = sorted(grouped_paths[path].keys())
            data = [grouped_paths[path][g] for g in gens]
            row_idx = row_offset + 2  # start 2 rows down

            ax = fig.add_subplot(gs[row_idx, col_idx])
            bplot = ax.boxplot(data, labels=gens, patch_artist=True, showfliers=False, widths=0.85)

            depth = get_depth(path)
            if depth > 2:
                path = path.replace(f"{grp}.", "")

            # based on depth, shift plot to the right within the column
            pos = ax.get_position()
            shift = 0.05 * (depth - 2)  # adjust multiplier to control shift
            ax.set_position([pos.x0 + shift, pos.y0, pos.width - shift, pos.height])

            alpha = 0.2 + 0.8 * (depth / max_depth)
            for box in bplot['boxes']:
                #box.set_facecolor(to_rgba('#cccccc', alpha))
                
                # consistent light grey for all boxes
                box.set_facecolor('#cccccc')
                box.set_edgecolor('black')
                box.set_linewidth(1.5)
            for median in bplot['medians']:
                median.set_color('black')
                median.set_linewidth(1)
                median.set_linestyle('--')
            for whisker in bplot['whiskers']: whisker.set_color('black')
            for cap in bplot['caps']: cap.set_color('black')

            run_glsar_and_plot(ax, gens, data)

            #label = path.replace(f"{grp}.", "")
            #indent = '   ' * (depth - 1)
            #label = indent + label
            label = path.replace(f".fit", "")
            
            ax.set_title(label, fontsize=6, fontweight='bold' if depth == 2 else 'normal')
            #ax.set_ylabel(label, fontsize=6, fontweight='bold' if depth == 2 else 'normal')
            ax.tick_params(axis='x', labelrotation=45, labelsize=7)
            ax.tick_params(axis='y', labelsize=6)
            ax.grid(True, axis='y')

    fig.tight_layout()
    output_path = os.path.join(output_dir, "summary_all_fitness_boxplots_columnwise.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✅ Columnwise summary boxplots saved to: {output_path}")
    return output_path

# Modified version of the user's function to include vertical tree lines connecting parent-child plots by depth.
def plot_all_fitness_boxplots_columnwise_with_tree_lines(grouped_paths, output_dir, sorted_paths):
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import ConnectionPatch

    def get_depth(path): return len(path.split('.'))
    def get_group(path): return path.split('.')[0]

    paths = [p for p in sorted_paths if p != 'fit']
    groups = {}
    for p in paths:
        grp = get_group(p)
        groups.setdefault(grp, []).append(p)

    max_depth = max(get_depth(p) for p in paths)
    col_count = len(groups)
    max_rows = max(len(v) for v in groups.values()) + 2
    row_count = max(len(v) for v in groups.values()) + 1

    fig_width = 2 + 2.2 * col_count
    fig_height = 2 + 1.4 * max_rows
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(row_count + 1, col_count, height_ratios=[0.7] + [1]*row_count, figure=fig, hspace=0.6, wspace=0.4)

    axes_positions = {}  # Store axis positions for connecting lines
    axes_parents = {}  # Store parent axes for each plot
    def run_glsar_and_plot(ax, gens, data):
        all_gen = np.concatenate([[g] * len(data[i]) for i, g in enumerate(gens)])
        all_fit = np.concatenate(data)
        df = pd.DataFrame({'gen': all_gen, 'value': all_fit})
        df['gen_plus1'] = df['gen'] + 1
        model = GLSAR(df['value'], sm.add_constant(df['gen']), rho=1)
        try:
            res = model.iterative_fit(maxiter=10)
            df['trend'] = res.predict(sm.add_constant(df['gen']))
            ax.plot(df['gen_plus1'], df['trend'], color='red', linewidth=1)
            p_val = res.pvalues[1]
            stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            if stars:
                ax.text(1.05, 0.5, stars, transform=ax.transAxes, ha='left', va='center', fontsize=14, color='red')
        except Exception as e:
            print(f"Error fitting GLSAR for {path}: {e}")
            df['trend'] = np.nan
            ax.text(1.05, 0.5, 'Error', transform=ax.transAxes, ha='left', va='center', fontsize=14, color='red')
                
    if 'fit' in sorted_paths:
        gens = sorted(grouped_paths['fit'].keys())
        data = [grouped_paths['fit'][g] for g in gens]
        ax = fig.add_subplot(gs[0:2, :])
        bplot = ax.boxplot(data, labels=gens, patch_artist=True, showfliers=False, widths=0.85)
        for box in bplot['boxes']:
            box.set_facecolor('#cccccc')
            box.set_edgecolor('black')
            box.set_linewidth(1.5)
        for median in bplot['medians']:
            median.set_color('black')
            median.set_linewidth(1)
            median.set_linestyle('--')
        for whisker in bplot['whiskers']: whisker.set_color('black')
        for cap in bplot['caps']: cap.set_color('black')
        run_glsar_and_plot(ax, gens, data)
        
        # Add elites and local minima for the 'fit' metric only
        #if 'fit' in grouped_paths:
        #grouped_fits = grouped_paths['fit']
        grouped_fits = grouped_paths['fit']
        gens = sorted(grouped_fits.keys())
        min_fit_by_gen = {gen: min(grouped_fits[gen]) for gen in gens if grouped_fits[gen]}
        sorted_gens = sorted(min_fit_by_gen)
        values = [min_fit_by_gen[gen] for gen in sorted_gens]

        # Find local minima in chronological order
        local_minima = []
        minima = None
        for i in range(0, len(values)):
            if minima is None or values[i] < minima:
                minima = values[i]
                local_minima.append((sorted_gens[i], values[i]))

        # top_n = 32
        # local_elites = []
        # new_elites = []

        # for gen in sorted(grouped_fits.keys()):
        #     fits = grouped_fits[gen]
        #     sorted_fits = sorted(fits)
        #     for val in sorted_fits:
        #         if len(local_elites) < top_n:
        #             local_elites.append((gen, val))
        #             new_elites.append((gen, val))
        #         else:
        #             current_max = max(local_elites, key=lambda x: x[1])
        #             if val < current_max[1] and (gen, val) not in local_elites:
        #                 local_elites.append((gen, val))
        #                 local_elites = sorted(local_elites, key=lambda x: x[1])[:top_n]
        #                 new_elites.append((gen, val))

        # # Plot new elites
        # for gen, val in new_elites:
        #     if gen == 0:
        #         continue
        #     #ax.plot(gen + 1, val, marker='*', color='orange', markersize=10, label='New Elite')
        #     # orange dot
        #     #ax.plot(gen + 1, val, marker='o', color='orange', markersize=3, label='New Elites')
        #     #black dots instead
        #     ax.plot(gen + 1, val, marker='o', color='black', markersize=3, label='New Elites')

        # aw 2025-05-27 21:47:29 - fixed new elite logic to only plot new elites
        top_n = 32
        prev_elite_values = set()
        new_elites = []

        for gen in sorted(grouped_fits.keys()):
            fits = grouped_fits[gen]
            top_fits = sorted(fits)[:top_n]  # smaller is better

            # Get new entries not in previous generation's elites
            current_elite_set = set(top_fits)
            new_vals = current_elite_set - prev_elite_values

            # Track new elite values along with generation
            for val in new_vals:
                new_elites.append((gen, val))

            prev_elite_values = current_elite_set

        # Plot only the new elite values
        for gen, val in new_elites:
            if gen == 0:
                continue
            ax.plot(gen + 1, val, marker='o', color='black', markersize=3, label='New Elites')


        # Plot local minima
        first = True
        for gen, val in local_minima:
            label = 'Local Minima' if first else None
            #ax.plot(gen + 1, val, 'r*', markersize=12, label=label)
            # red dot
            ax.plot(gen + 1, val, marker='o', color='red', markersize=3, label=label)
            first = False

        
        # add legend, avoid repeating labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=6, frameon=False)
        #ax.set_ylabel("Overall Fitness", fontsize=7, fontweight='bold')


        ax.set_title('Overall Fitness', fontsize=7, fontweight='bold')
        ax.tick_params(axis='x', labelrotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(True, axis='y')

    for col_idx, (grp, paths_in_group) in enumerate(groups.items()):
        parent_pos = None
        parent_depth = None

        for row_offset, path in enumerate(paths_in_group):
            gens = sorted(grouped_paths[path].keys())
            data = [grouped_paths[path][g] for g in gens]
            row_idx = row_offset + 2
            ax = fig.add_subplot(gs[row_idx, col_idx])

            bplot = ax.boxplot(data, labels=gens, patch_artist=True, showfliers=False, widths=0.85)
            for box in bplot['boxes']:
                box.set_facecolor('#cccccc')
                box.set_edgecolor('black')
                box.set_linewidth(1.5)
            for median in bplot['medians']:
                median.set_color('black')
                median.set_linewidth(1)
                median.set_linestyle('--')
            for whisker in bplot['whiskers']: whisker.set_color('black')
            for cap in bplot['caps']: cap.set_color('black')

            run_glsar_and_plot(ax, gens, data)

            depth = get_depth(path)
            label = path.replace(f".fit", "")
            ax.set_title(label, fontsize=6, fontweight='bold' if depth == 2 else 'normal')
            ax.tick_params(axis='x', labelrotation=45, labelsize=7)
            ax.tick_params(axis='y', labelsize=6)
            ax.grid(True, axis='y')

            this_pos = ax.get_position()
            axes_positions[path] = (this_pos.x0 + this_pos.width / 2, this_pos.y0 + this_pos.height / 2)
            axes_parents[path] = ax # Store the axis for this path

            # Build full ancestral tree, keeping `.fit` suffix
            family_tree = {}
            path_parts = path.split('.')[:-1]  # remove final 'fit'
            while len(path_parts) >= 1:
                ancestor = '.'.join(path_parts) + '.fit'
                if ancestor in axes_positions:
                    #family_tree.append(axes_positions[ancestor])
                    family_tree[ancestor] = axes_positions[ancestor]
                path_parts.pop()  # move up one level

            # Draw connection line to parent depth (shallower)
            base_indent = -0.25
            indent_step = 0.025

            for ancestor, pos in family_tree.items():
                if ancestor == path:
                    continue
                ancestor_depth = get_depth(ancestor)
                if ancestor_depth < depth:
                    #ancestor_pos = family_tree[ancestor]
                    ancestor_ax = axes_parents[ancestor]

                    #x_offset = base_indent + indent_step * (depth - 3)
                    x_offset = base_indent + indent_step * (ancestor_depth - 2)  # -3 so highest depth is at -0.115
                    line = ConnectionPatch(
                        xyA=(x_offset, 0), coordsA="axes fraction",
                        xyB=(x_offset, 0.5), coordsB="axes fraction",
                        axesA=ax, axesB=ancestor_ax,
                        color='gray', linewidth=0.5
                    )
                    fig.add_artist(line)
            # if parent_pos is not None:

            # if parent_depth and depth > parent_depth:
            #     x_offset = base_indent + indent_step * (depth - 3) #-3 so highest depth is at -0.115
            #     line = ConnectionPatch(
            #         xyA=(x_offset, 0), coordsA="axes fraction",
            #         xyB=(x_offset, 1), coordsB="axes fraction",
            #         axesA=ax, axesB=parent_pos,
            #         color='gray', linewidth=0.5
            #     )
            #     fig.add_artist(line)

            # parent_pos = ax
            # parent_depth = depth

            # parent_pos = ax
            # #parent_depth = depth
            # #only update parent_depth if this is a new group or the first path
            # if parent_depth is None or get_group(path) != get_group(paths_in_group[0]):
            #     parent_depth = depth

    fig.tight_layout()
    fig.subplots_adjust(top=0.99, 
        bottom=0.025, 
        #left=0.05, 
        #right=0.98, 
        hspace=0.6, wspace=0.4)
    output_path = os.path.join(output_dir, "summary_all_fitness_boxplots_columnwise_treelines.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✅ Tree-line version of summary saved to: {output_path}")
    return output_path

def main():
    
    debug = False  # 🔁 Flip to True when debugging
    
    #uncomment this to run in debug mode
    # debug = True  # 🔁 Flip to True when debugging
    # debug_args = [
    #     '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-25_normBRandFRR_optBLandAmps_db',
    #     #    '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-20_BRandFRratios',
    #     #    '--output_dir',
    #     #    '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-20_BRandFRratios/drift',
    #         '--workers', '12',
    #     #    '--no_jitter',
    # ]
    
    args = parse_args(debug_args if debug else None)

    batch_paths = args.batch_paths
    output_dir = args.output_dir or os.path.join(batch_paths[0], 'evolution')
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Batch Paths: {batch_paths}")
    print(f"📤 Output Dir: {output_dir}")
    print(f"👷 Workers: {args.workers}")

    fitness_jsons = find_fitness_jsons(batch_paths)
    with Pool(args.workers) as pool:
        fitness_entries = pool.map(load_fit_file, fitness_jsons)

    fitness_entries = distribute_elites(fitness_entries)
    fitness_entries, high_fit_entries = filter_fitness_entries(fitness_entries)
    grouped_paths = group_by_generation(fitness_entries)

    pdf_path = os.path.join(output_dir, "fitness_evolution_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    sorted_paths = sort_metric_keys(grouped_paths.keys())
    
    #plot_all_fitness_boxplots_inline(grouped_paths, output_dir, sorted_paths)
    #plot_all_fitness_boxplots_columnwise(grouped_paths, output_dir, sorted_paths)
    plot_all_fitness_boxplots_columnwise_with_tree_lines(grouped_paths, output_dir, sorted_paths)
    
    # page_titles = [(path.replace('.', '_'), path) for path in sorted_paths]

    # add_back_link = add_cover_page_with_links(c, page_titles)

    # for path in sorted_paths:
    #     grouped_fits = grouped_paths[path]
    #     grouped_fits = {k: v for k, v in grouped_fits.items() if k != max(grouped_fits.keys())}

    #     bookmark_name = path.replace('.', '_')
    #     c.bookmarkPage(bookmark_name)
    #     c.setFont("Helvetica-Bold", 14)
    #     c.drawCentredString(width / 2, height - 1 * inch, f"Fitness Report for: {path}")

    #     try:
    #         plot_path, coeffs = plot_fitness_boxplots(grouped_fits, output_dir, tag=path)
    #         hist_path = plot_histogram_with_gaussian(grouped_fits, output_dir, tag=path)
    #         high_path = plot_high_fit_histogram(high_fit_entries, path_key=path, output_dir=output_dir)

    #         c.setFont("Helvetica", 10)
    #         poly_str = ' + '.join([f"{coeff:.3g}x^{i}" for i, coeff in enumerate(reversed(coeffs))])
    #         c.drawString(1 * inch, height - 1.3 * inch, f"2nd Order Polynomial Fit: y = {poly_str}")

    #         images = [plot_path, high_path, hist_path]
    #         y_pos = height - 2 * inch
    #         max_width = 6.5 * inch

    #         for img_path in images:
    #             img = Image.open(img_path)
    #             img_width, img_height = img.size
    #             aspect = img_height / img_width
    #             scaled_height = max_width * aspect
    #             c.drawImage(img_path, 1 * inch, y_pos - scaled_height, width=max_width, height=scaled_height)
    #             y_pos -= scaled_height + 0.3 * inch

    #     except Exception as e:
    #         traceback.print_exc()
    #         c.drawString(1 * inch, height - 5 * inch, f"⚠️ Error for {path}: {e}")

    #     add_back_link()
    #     c.showPage()

    # c.save()
    # print(f"✅ Report saved to: {pdf_path}")
    # print("✅ Done!")

if __name__ == "__main__":
    main()