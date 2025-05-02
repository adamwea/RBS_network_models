import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

# ─── import your evol parameter bounds ───────────────────────────────────────────
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.evol_params import params as evol_params

# ─── configure your batch paths here ────────────────────────────────────────────
batch_paths = [
    # '/global/homes/a/adammwea/pscratch/z_simulated_data/'
    # 'CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26/'
    
    # aw 2025-04-29 10:17:12
    '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-28/'
]

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
        else:
            print(f'❌ No match for: {fpath}')
    return sorted(entries, key=lambda x: (x[0], x[1]))

def load_cfg_file(entry):
    gen, cand, fpath = entry
    try:
        with open(fpath, 'r') as f:
            cfg = json.load(f).get('simConfig', {})
    except Exception as e:
        print(f'⚠️ Failed to load {fpath}: {e}')
        cfg = {}

    # filter only the keys in your evol_params dict
    filtered = {k: cfg.get(k, np.nan) for k in evol_params.keys() if k in cfg}
    return (gen, cand, filtered)

def group_by_param_and_gen(cfg_entries):
    # produces: { param_name: { gen: [val, val, ...], ... }, ... }
    grouped = {p: {} for p in evol_params.keys()}
    for gen, cand, cfg in cfg_entries:
        for p, val in cfg.items():
            if not np.isnan(val):
                grouped[p].setdefault(gen, []).append(val)
    return grouped

def plot_param_boxplots(grouped, output_dir):
    for param, gen_dict in grouped.items():
        gens = sorted(gen_dict.keys())
        data = [gen_dict[g] for g in gens]

        plt.figure(figsize=(12,6))
        bplot = plt.boxplot(data, labels=gens, patch_artist=True, showfliers=False)

        # color palette
        colors = plt.cm.tab10(np.linspace(0,1,len(gens)))

        low, high = evol_params[param]  # bound tuple

        # style each box
        for i, (box, vals, color) in enumerate(zip(bplot['boxes'], data, colors)):
            # fill
            box.set_facecolor((*color[:3], 0.3))
            # outline red if any val violates bounds
            if any(v < low or v > high for v in vals):
                box.set_edgecolor('red')
                box.set_linewidth(2.5)
            else:
                box.set_edgecolor('black')
                box.set_linewidth(1.0)

        # rest of styling
        for whisker in bplot['whiskers']:
            whisker.set_color('black'); whisker.set_linewidth(1)
        for cap in bplot['caps']:
            cap.set_color('black'); cap.set_linewidth(1)

        # scatter points
        for i, (gen, color) in enumerate(zip(gens, colors)):
            y = gen_dict[gen]
            x = np.random.normal(loc=i+1, scale=0.08, size=len(y))
            plt.scatter(x, y,
                        alpha=0.7, s=25,
                        edgecolor='k', linewidth=0.5,
                        color=color, zorder=3)

        plt.title(f"Parameter '{param}' Across Generations")
        plt.xlabel('Generation')
        plt.ylabel(param)
        plt.axhline(low,  color='gray', linestyle='--')
        plt.axhline(high, color='gray', linestyle='--')
        plt.grid(True, axis='y')
        plt.tight_layout()

        # save per-parameter
        pdf = os.path.join(output_dir, f"{param}_evolution.pdf")
        png = os.path.join(output_dir, f"{param}_evolution.png")
        plt.savefig(pdf)
        plt.savefig(png, dpi=300)
        plt.close()
        print(f"✅ Saved '{param}' plot to:\n  {pdf}\n  {png}")

if __name__ == "__main__":
    num_workers = 16
    cfg_entries = find_cfg_jsons(batch_paths)
    with Pool(processes=num_workers) as pool:
        cfg_entries = pool.map(load_cfg_file, cfg_entries)

    grouped = group_by_param_and_gen(cfg_entries)
    output_dir = batch_paths[0]
    plot_param_boxplots(grouped, output_dir)
    print("✅ All parameter‐drift plots complete!")
