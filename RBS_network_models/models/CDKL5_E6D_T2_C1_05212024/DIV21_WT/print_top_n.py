import os
import glob
import json
import numpy as np


def find_json_files(batch_dir, pattern='**/*_fitness.json'):
    return sorted([
        p for p in glob.glob(os.path.join(batch_dir, pattern), recursive=True)
        #if os.path.basename(p) != exclude_name
    ])

def filter_top_n_candidates(json_files, top_n=10):
    """
    Given job_args [(pdf_path, candidate_path, fit_value), ...],
    keep only the top N candidates based on best (lowest) fitness,
    but restore original generation/candidate ordering afterward.
    """
    if not json_files:
        return []

    fits = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            fits.append(data['fit'])
    
    sort_idx = np.argsort(fits)
    fits = np.array(fits)[sort_idx]
    
    # Sort by fitness value (ascending)
    #json_files_sorted_by_fit = sorted(json_files, key=lambda x: x[2])
    json_files_sorted_by_fit = np.array(json_files)[sort_idx]
    
    # Take top N
    top_fits = json_files_sorted_by_fit[:top_n]

    print(f"✅ Selected top {top_n} candidates based on fitness and sorted by gen/cand order.")
    #return top_jobs_sorted
    return top_fits

if __name__ == "__main__":
    batch_dir = ['/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26']
    json_files = find_json_files(batch_dir[0])
    top_n = 256
    top_fits = filter_top_n_candidates(json_files, top_n=top_n)
    
    # print each cfg file path like a string ready to be copied
    # based on json file path
    output_file = os.path.join(batch_dir[0], "top_candidates.md")
    with open(output_file, 'w') as f:
        f.write("# Top Candidates\n\n")
        for json_file in top_fits:
            cfg_file = json_file.replace('_fitness.json', '_cfg.json')
            f.write(f"'{cfg_file}',\n")
    print(f"✅ Top candidates saved to {output_file}")
    
   