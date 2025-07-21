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

batch_paths = [
    #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-15/'

    # aw 2025-05-16 11:59:16
    #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-16/'

    # aw 2025-05-19 10:24:20 
    #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-18/'

    # aw2025-05-19 19:58:06
    #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-19_BRandFRs/'

    # aw 2025-05-19 22:52:52
    #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-19_BRandFRratios/'

    # aw 2025-05-20 21:31:11 
    #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-20_BRandFRratios/',

    #
    #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-23_normBRandFRR_optBLandAmps/',
    '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-23_normBRandFRR_optBLandAmps_2/',
]

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

from reportlab.lib.units import inch

from reportlab.pdfbase.pdfmetrics import stringWidth

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

if __name__ == "__main__":
    num_workers = 16
    fitness_jsons = find_fitness_jsons(batch_paths)

    with Pool(num_workers) as pool:
        fitness_entries = pool.map(load_fit_file, fitness_jsons)

    fitness_entries = distribute_elites(fitness_entries)
    fitness_entries, high_fit_entries = filter_fitness_entries(fitness_entries)
    grouped_paths = group_by_generation(fitness_entries)
    output_dir = os.path.join(batch_paths[0], 'evolution_reports')
    os.makedirs(output_dir, exist_ok=True)

    from PIL import Image
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.pagesizes import letter

    pdf_path = os.path.join(output_dir, "fitness_evolution_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    sorted_paths = sort_metric_keys(grouped_paths.keys())
    page_titles = [(path.replace('.', '_'), path) for path in sorted_paths]

    # Add cover/index page first
    add_back_link = add_cover_page_with_links(c, page_titles)

    for path in sorted_paths:
        grouped_fits = grouped_paths[path]
        bookmark_name = path.replace('.', '_')
        print(f"Processing {path}...")

        # Bookmark this page
        c.bookmarkPage(bookmark_name)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width / 2, height - 1 * inch, f"Fitness Report for: {path}")

        try:
            #remove last gen - it's probably incomplete
            grouped_fits = {k: v for k, v in grouped_fits.items() if k != max(grouped_fits.keys())}

            plot_path, coeffs = plot_fitness_boxplots(grouped_fits, output_dir, tag=path)
            #plot_path, coeffs = plot_fitness_violinplots(grouped_fits, output_dir, tag=path)
            hist_path = plot_histogram_with_gaussian(grouped_fits, output_dir, tag=path)
            high_path = plot_high_fit_histogram(high_fit_entries, path_key=path, output_dir=output_dir)

            c.setFont("Helvetica", 10)
            poly_str = ' + '.join([f"{coeff:.3g}x^{i}" for i, coeff in enumerate(reversed(coeffs))])
            c.drawString(1 * inch, height - 1.3 * inch, f"3rd Order Polynomial Fit: y = {poly_str}")

            images = [
                plot_path, 
                high_path,
                hist_path, 
                #high_path
                ]
            y_pos = height - 2 * inch
            max_width = 6.5 * inch

            for img_path in images:
                img = Image.open(img_path)
                img_width, img_height = img.size
                aspect = img_height / img_width
                scaled_height = max_width * aspect
                c.drawImage(img_path, 1 * inch, y_pos - scaled_height, width=max_width, height=scaled_height)
                y_pos -= scaled_height + 0.3 * inch

        except Exception as e:
            traceback.print_exc()
            c.drawString(1 * inch, height - 5 * inch, f"⚠️ Error for {path}: {e}")

        # Add navigation link and finish page
        add_back_link()
        c.showPage()

    c.save()
    print(f"✅ Multipage PDF report generated at: {pdf_path}")
    print("✅ Done!")
