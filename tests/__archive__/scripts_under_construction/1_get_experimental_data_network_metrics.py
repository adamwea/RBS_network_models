# This script is used to get the network metrics from the experimental data. It uses the helper.py script to import the necessary modules and functions.
from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import *

'''Args'''
'''Recordings to analyze'''
recording_paths = [
    #"/pscratch/sd/a/adammwea/xRBS_input_data/CDKL5-E6D_T2_C1_05212024/240611/M06844/Network/000076/data.raw.h5",
    #"/pscratch/sd/a/adammwea/workspace/xInputs/xRBS_input_data/CDKL5-E6D_T2_C1_05212024/240611/M06844/Network/000076/data.raw.h5"
    # Data shouldn't be stored in the repo so I'm going to use full paths here since downstream users will need to change this anyway.
    "/pscratch/sd/a/adammwea/workspace/xInputs/xRBS_input_data/CDKL5-E6D_T2_C1_05212024/240611/M06844/Network/000076/data.raw.h5",
]

'''Convolution Params''' #going to be tweaking this as we develop from now on, so I'll need to keep this up to date
#convolution_params_path = "/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/convolution_params/_standard_convolution_params.py"
#convolution_params_path = "/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/convolution_params/241202_convolution_params.py"
    #tweaking binsize to be smaller to get better resolution of network activity for shorter simulations.
convolution_params_path = "/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/_config/convolution_params/241202_convolution_params.py"
    #just moving things around

'''Output for sorting objects'''
#expect to find sorting objects in this directory, we aren't
# adding any new sorting objects here.
sorting_output_parent_path = "/pscratch/sd/a/adammwea/workspace/yThroughput/yRBS_spike_sorted_data"

'''Output for network metrics'''
output_path =(
    "/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21"
    "/_config/experimental_data_features/network_metrics"
)

'''options'''
stream_select = 0

'''main'''
paired_objects = load_recording_and_sorting_object_tuples(recording_paths, sorting_output_parent_path, stream_select) 
for pairs in paired_objects:
    stream_nums = [0, 1, 2, 3, 4, 5]
    if stream_select is not None: stream_nums = [stream_select]
    for stream_num in stream_nums:            
        kwargs = {
            'recording_object': pairs[0],
            'sorting_object': pairs[1],
            'details': pairs[2],
            'stream_num': stream_num,
            'output_path': output_path,
            'convolution_params_path': convolution_params_path,
            'plot': True,
        }        
        save_network_metrics(**kwargs)
print("Network metrics saved.")