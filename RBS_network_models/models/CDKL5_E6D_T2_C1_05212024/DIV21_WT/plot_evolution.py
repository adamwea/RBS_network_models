import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt
import numpy as np

batch_paths = [
    # '/global/homes/a/adammwea/pscratch/z_simulated_data/'
    # 'CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26/'
    
    # aw 2025-04-29 10:16:42
    '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-28/'
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
        if fit < 1000:  # exclude large values
            grouped.setdefault(gen, []).append(fit)
    return grouped

def plot_fitness_boxplots(grouped_fits, output_dir):
    gens = sorted(grouped_fits.keys())
    data = [grouped_fits[gen] for gen in gens]

    plt.figure(figsize=(12, 6))
    bplot = plt.boxplot(
        data, labels=gens, patch_artist=True, showfliers=False
    )

    # Define a color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(gens)))

    # Style boxes: transparent fill, red outline for gen 0
    for i, (box, color) in enumerate(zip(bplot['boxes'], colors)):
        box.set_facecolor((*color[:3], 0.3))  # RGBA: same hue, lower alpha
        box.set_edgecolor('red' if i == 0 else 'black')
        box.set_linewidth(2.5 if i == 0 else 1.0)

    for whisker in bplot['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1)
    for cap in bplot['caps']:
        cap.set_color('black')
        cap.set_linewidth(1)

    # Jittered scatter points by generation color
    for i, (gen, color) in enumerate(zip(gens, colors)):
        y = grouped_fits[gen]
        x = np.random.normal(loc=i + 1, scale=0.08, size=len(y))
        plt.scatter(
            x, y,
            alpha=0.7,
            s=25,
            edgecolor='k',
            linewidth=0.5,
            color=color,
            zorder=3,
            label=f"Gen {gen}" if i == 0 else None  # optional legend
        )

    plt.title('Candidate Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True, axis='y')
    plt.tight_layout()

    # Save plots
    pdf_path = os.path.join(output_dir, "fitness_evolution_plot.pdf")
    png_path = os.path.join(output_dir, "fitness_evolution_plot.png")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=300)
    print(f"✅ Saved plot to:\n  {pdf_path}\n  {png_path}")
    # plt.show()  # Enable to view inline

if __name__ == "__main__":
    num_workers = 16
    fitness_jsons = find_fitness_jsons(batch_paths)

    with Pool(processes=num_workers) as pool:
        fitness_entries = pool.map(load_fit_file, fitness_jsons)

    grouped_fits = group_by_generation(fitness_entries)
    output_dir = batch_paths[0]
    plot_fitness_boxplots(grouped_fits, output_dir)
    print("✅ Done!")
