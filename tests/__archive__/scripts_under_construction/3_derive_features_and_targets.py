# New Fitness Targets
from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import *

#load network metric npy
npy_path = (
    "/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/"
    "CDKL5_DIV21/_config/experimental_data_features/network_metrics/"
    "CDKL5-E6D_T2_C1_05212024_240611_M06844_Network_000076_network_metrics_well000.npy"
)

#
output_dir = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/_config/experimental_data_features'

'''script'''
'''network_metric_targets skeleton'''
network_metrics = np.load(npy_path, allow_pickle=True).item()
network_metrics_targets = build_network_metric_targets_dict(network_metrics)
save_network_metric_dict_with_timestamp(network_metrics, network_metrics_targets, output_dir)

  