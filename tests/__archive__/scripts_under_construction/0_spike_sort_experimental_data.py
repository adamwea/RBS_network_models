from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import *

def main (**kwargs):
    '''sort data'''
    sorting_objects = spike_sort_experimental_data(**kwargs)
    return sorting_objects

'''
UI
'''

'''Recordings to sort'''
recording_paths = [
    #"/pscratch/sd/a/adammwea/xRBS_input_data/CDKL5-E6D_T2_C1_05212024/240611/M06844/Network/000076/data.raw.h5",
    #"/pscratch/sd/a/adammwea/workspace/xInputs/xRBS_input_data/CDKL5-E6D_T2_C1_05212024/240611/M06844/Network/000076/data.raw.h5"
    # Data shouldn't be stored in the repo so I'm going to use full paths here since downstream users will need to change this anyway.
    "/pscratch/sd/a/adammwea/workspace/xInputs/xRBS_input_data/CDKL5-E6D_T2_C1_05212024/240611/M06844/Network/000076/data.raw.h5",
]

'''Spike Sorting Parameters Kilosort2 (default for now)'''
default_kilosort_2_params = get_default_kilosort2_params()
sorting_params = default_kilosort_2_params.copy()
sorting_params.update(
    {
        #update parameters here...
    }
)

'''Output for sorting objects'''
sorting_output_path = "/pscratch/sd/a/adammwea/workspace/yThroughput/yRBS_spike_sorted_data"

'''Args'''
stream_select = None  # Set to a list of well indices if needed, e.g., [0, 2, 4]
stream_select = 0

if __name__ == "__main__":
    kwargs = {
        "stream_select": stream_select,
        "recording_paths": recording_paths,
        'sorting_params': sorting_params,
        "sorting_output_path": sorting_output_path,
    }    
    sorting_objects=main(**kwargs) #sorting objects will be saved to the output path, and can be used downstream.