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

batch_paths = [
    '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-09/'
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
    try:
        with open(fpath, 'r') as f:
            fitness = json.load(f).get('fit', np.nan)
    except Exception as e:
        print(f'⚠️ Failed to load {fpath}: {e}')
        fitness = np.nan
    return (gen, cand, fitness)

def group_by_generation(fitness_entries):
    grouped = {}
    for gen, cand, fit in fitness_entries:
        grouped.setdefault(gen, []).append(fit)
    return grouped

from scipy.stats import norm

def plot_histogram_with_gaussian(grouped_fits, output_dir):
    from scipy.stats import norm

    all_fitness = [fit for gen_fits in grouped_fits.values() for fit in gen_fits]
    hist_path = os.path.join(output_dir, "fitness_histogram.png")
    plt.figure(figsize=(12, 4))
    n, bins, patches = plt.hist(all_fitness, bins=50, color='gray', alpha=0.7, edgecolor='black')

    mu, std = norm.fit(all_fitness)
    x = np.linspace(min(bins), max(bins), 1000)
    p = norm.pdf(x, mu, std) * (len(all_fitness) * (bins[1] - bins[0]))
    plt.plot(x, p, 'r--', linewidth=2, label=f'Gaussian Fit (μ={mu:.2f}, σ={std:.2f})')
    plt.title('Distribution of Candidate Fitness')
    plt.xlabel('Fitness')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()
    
    return hist_path

def plot_fitness_boxplots(grouped_fits, output_dir):
    gens = sorted(grouped_fits.keys())
    data = [grouped_fits[gen] for gen in gens]

    plt.figure(figsize=(12, 6))
    bplot = plt.boxplot(
        data, labels=gens, patch_artist=True, showfliers=False
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(gens)))
    colors = [colors[0]] * len(gens)

    for i, (box, color) in enumerate(zip(bplot['boxes'], colors)):
        box.set_facecolor((*color[:3], 0.3))
        box.set_edgecolor('red' if i == 0 else 'black')
        box.set_linewidth(2.5 if i == 0 else 1.0)

    for whisker in bplot['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1)
    for cap in bplot['caps']:
        cap.set_color('black')
        cap.set_linewidth(1)

    for i, gen in enumerate(gens):
        q1, q3 = np.percentile(data[i], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [y for y in data[i] if y < lower_bound or y > upper_bound]
        x = np.random.normal(loc=i + 1, scale=0.05, size=len(outliers))
        plt.scatter(x, outliers, s=5, color='red', alpha=0.6, label='Outliers' if i == 0 else None)

    # Fit and plot a polynomial curve (3rd degree)
    all_gen = []
    all_fit = []
    for gen in gens:
        all_gen.extend([gen] * len(grouped_fits[gen]))
        all_fit.extend(grouped_fits[gen])

    coeffs = np.polyfit(all_gen, all_fit, deg=3)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(gens), max(gens), 300)
    y_fit = poly(x_fit)
    plt.plot(x_fit + 1, y_fit, color='blue', linestyle='--', linewidth=2, label='3rd Order Trend')

    plt.title('Candidate Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(output_dir, "fitness_evolution_plot.png")
    plt.savefig(png_path, dpi=300)
    # Save boxplot
    plt.close()

    
    

    print(f"✅ Saved plot to: {png_path}")
    return png_path, coeffs

def generate_pdf_report(output_dir, plot_image_path, hist_image_path, grouped_fits, poly_coeffs=None):
    pdf_path = os.path.join(output_dir, "fitness_evolution_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 1 * inch, "Fitness Evolution Report")

    c.setFont("Helvetica", 12)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawCentredString(width / 2, height - 1.3 * inch, f"Generated on: {timestamp}")

    total_gens = len(grouped_fits)
    total_candidates = sum(len(fits) for fits in grouped_fits.values())
    c.drawString(1 * inch, height - 2 * inch, f"Total Generations: {total_gens}")
    c.drawString(1 * inch, height - 2.3 * inch, f"Total Candidates: {total_candidates}")

    if poly_coeffs is not None:
        poly_str = ' + '.join([f"{coeff:.3g}x^{i}" for i, coeff in enumerate(reversed(poly_coeffs))])
        c.drawString(1 * inch, height - 2.6 * inch, f"3rd Order Polynomial Fit: y = {poly_str}")

    try:
        from PIL import Image
        img = Image.open(plot_image_path)
        img_width, img_height = img.size
        aspect = img_height / img_width

        max_width = 6.5 * inch
        scaled_height = max_width * aspect
        x = 1 * inch
        y = height - 3.2 * inch - scaled_height
        c.drawImage(plot_image_path, x, y, width=max_width, height=scaled_height)

        # Draw histogram underneath
        img2 = Image.open(hist_image_path)
        img2_width, img2_height = img2.size
        aspect2 = img2_height / img2_width
        scaled_height2 = max_width * aspect2
        y2 = y - scaled_height2 - 0.3 * inch
        c.drawImage(hist_image_path, x, y2, width=max_width, height=scaled_height2)
    except Exception as e:
        c.drawString(1 * inch, height - 5 * inch, f"⚠️ Failed to load plot image: {e}")

    c.showPage()
    c.save()
    print(f"✅ PDF report generated at: {pdf_path}")

if __name__ == "__main__":
    num_workers = 16
    fitness_jsons = find_fitness_jsons(batch_paths)

    with Pool(processes=num_workers) as pool:
        fitness_entries = pool.map(load_fit_file, fitness_jsons)

    grouped_fits = group_by_generation(fitness_entries)
    output_dir = os.path.join(batch_paths[0], 'evolution_reports')
    os.makedirs(output_dir, exist_ok=True)

    plot_path, coeffs = plot_fitness_boxplots(grouped_fits, output_dir)
    hist_path = plot_histogram_with_gaussian(grouped_fits, output_dir)
    generate_pdf_report(output_dir, plot_path, hist_path, grouped_fits, coeffs)

    print("✅ Done!")
