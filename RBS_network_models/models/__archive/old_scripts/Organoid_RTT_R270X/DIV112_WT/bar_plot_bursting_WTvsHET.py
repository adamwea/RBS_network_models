# Imports =====================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
# Main =============================================================================
data_paths = [
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/network_metrics/Organoid_RTT_R270X_pA_pD_B1_d91_250107_M07297_Network_000028_network_metrics_well000.npy',
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/network_metrics/Organoid_RTT_R270X_pA_pD_B1_d91_250107_M07297_Network_000028_network_metrics_well001.npy',
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/network_metrics/Organoid_RTT_R270X_pA_pD_B1_d91_250107_M07297_Network_000028_network_metrics_well002.npy',
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/network_metrics/Organoid_RTT_R270X_pA_pD_B1_d91_250107_M07297_Network_000028_network_metrics_well003.npy',
    #'/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/network_metrics/Organoid_RTT_R270X_pA_pD_B1_d91_250107_M07297_Network_000028_network_metrics_well004.npy', #spiikesorting failed
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/network_metrics/Organoid_RTT_R270X_pA_pD_B1_d91_250107_M07297_Network_000028_network_metrics_well005.npy',
]

known_well_sources = {
    'well000': 'MUT',
    'well001': 'MUT',
    'well002': 'MUT',
    'well003': 'WT',
    'well004': 'WT',
    'well005': 'WT',    
}

# Load data
#data = []
data_by_well = {}
for data_path in data_paths:
    well = data_path.split('_')[-1].split('.')[0]
    #data.append(np.load(data_path, allow_pickle=True).item())
    data_by_well[well] = {}
    data_by_well[well]['source'] = known_well_sources[well]
    data_by_well[well]['npy'] = np.load(data_path, allow_pickle=True).item()
    
# get regular and hyper bursting data out of the data - parse by well
for well, d in data_by_well.items():
    data_by_well[well]['Number_Bursts'] = d['npy']['bursting_data']['bursting_summary_data']['Number_Bursts']
    data_by_well[well]['Number_Hyper_Bursts'] = d['npy']['mega_bursting_data']['bursting_summary_data']['Number_Bursts']

# regular_bursting = {}
# hyper_bursting = {}
# for i, d in enumerate(data):
#     regular_bursting[i] = d['bursting_data']['bursting_summary_data']['Number_Bursts']
#     hyper_bursting[i] = d['mega_bursting_data']['bursting_summary_data']['Number_Bursts']
    
# plot bar graph - group data by source
print('Plotting bar graph...')

# Organize data by WT and MUT groups
WT = np.array([[d['Number_Bursts'], d['Number_Hyper_Bursts']] for well, d in data_by_well.items() if d['source'] == 'WT'])
MUT = np.array([[d['Number_Bursts'], d['Number_Hyper_Bursts']] for well, d in data_by_well.items() if d['source'] == 'MUT'])

# Compute means and standard errors
wt_mean = WT.mean(axis=0)
mut_mean = MUT.mean(axis=0)
wt_sem = WT.std(axis=0, ddof=1) / np.sqrt(WT.shape[0])
mut_sem = MUT.std(axis=0, ddof=1) / np.sqrt(MUT.shape[0])

# Create separate plots for Regular and Hyper Bursts with circles for data points
fig, axs = plt.subplots(2, 1, figsize=(7, 10))

# Define colors
wt_color = '#AEC6CF'  # Pastel blue
mut_color = '#F4C2C2'  # Pastel red

# X locations for bars
barWidth = 0.35
r1 = np.array([0])  # WT position
r2 = np.array([barWidth])  # MUT position

# Regular Bursts Plot
axs[0].bar(r1, wt_mean[0], yerr=wt_sem[0], color=wt_color, width=barWidth, capsize=5, edgecolor='grey', label='WT')
axs[0].bar(r2, mut_mean[0], yerr=mut_sem[0], color=mut_color, width=barWidth, capsize=5, edgecolor='grey', label='MUT')

# Scatter plot of individual values with circles
axs[0].scatter(np.full(WT.shape[0], r1), WT[:, 0], color='b', alpha=0.5, edgecolor='black', label='WT Data', s=80)
axs[0].scatter(np.full(MUT.shape[0], r2), MUT[:, 0], color='r', alpha=0.5, edgecolor='black', label='MUT Data', s=80)

axs[0].set_title('Regular Bursts', fontweight='bold')
axs[0].set_ylabel('Number of Bursts', fontweight='bold')
axs[0].set_xticks([r1[0], r2[0]])
axs[0].set_xticklabels(['WT', 'MUT'])
axs[0].legend()

# Hyper Bursts Plot
axs[1].bar(r1, wt_mean[1], yerr=wt_sem[1], color=wt_color, width=barWidth, capsize=5, edgecolor='grey', label='WT')
axs[1].bar(r2, mut_mean[1], yerr=mut_sem[1], color=mut_color, width=barWidth, capsize=5, edgecolor='grey', label='MUT')

# Scatter plot of individual values with circles
axs[1].scatter(np.full(WT.shape[0], r1), WT[:, 1], color='b', alpha=0.5, edgecolor='black', label='WT Data', s=80)
axs[1].scatter(np.full(MUT.shape[0], r2), MUT[:, 1], color='r', alpha=0.5, edgecolor='black', label='MUT Data', s=80)

axs[1].set_title('Hyper Bursts', fontweight='bold')
axs[1].set_ylabel('Number of Bursts', fontweight='bold')
axs[1].set_xticks([r1[0], r2[0]])
axs[1].set_xticklabels(['WT', 'MUT'])
axs[1].legend()

# Adjust layout and show
plt.tight_layout()

path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/figures/Number_Bursts_by_Well_and_Source.png'
save_dir = os.path.dirname(path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.savefig(path)
print(f'Bar plot saved to: {path}')