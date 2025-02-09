global PROGRESS_SLIDES_PATH, SIMULATION_RUN_PATH, REFERENCE_DATA_NPY, CONVOLUTION_PARAMS, DEBUG_MODE
#from RBS_network_models.utils.sim_helper import run_sensitivity_analysis
from RBS_network_models.sensitivity_analysis import run_sensitivity_analysis, plot_sensitivity_analysis
import os
import time
# ===================================================================================================
sim_data_path = (
    # '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/'
    # 'tests/outputs/test_run_a_simulation/test_run_5_data.pkl'
    
    # 2025-01-11 15:08:04 - moved 'tests' out of installed part of package
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/tests'
    '/outputs/test_run_a_simulation/test_run_5_data.pkl'    
    )
output_dir =(
    #original testing
    # '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/'
    # 'CDKL5_DIV21/tests/outputs/test_sensitivity_analysis'
    
    #level testing - 2025-01-10 10:14:16
    # '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/'
    # 'tests/outputs/test_sensitivity_analysis_levels'
    
    # aw 2025-01-11 15:10:04 - moved 'tests' out of installed part of package
    # '/pscratch/sd/a/adammwea/workspace/RBS_network_models/tests'
    # '/outputs/test_sensitivity_analysis_levels'
    
    # aw 2025-01-11 15:37:05 - renamed folder and reorganized data
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/tests'
    '/outputs/test_sensitivity_analysis_levels_mod' #NOTE: I renamed the above folder to this one. This one has all the data that would end up in the above folder.
                                                    #     # I've moved files wihtin the renamed folder to an organized structure that I think is better.
                                                    #     # TODO: update run_sensitivity_analysis to save data in the new structure.
                                                    #     # TODO: update plot_sensitivity_grid_plots to load data from the new structure.
                                                    #               # This is what I'm working on first, and why I renamed and reorganized the folder first. 
    )
reference_data_path = (
    # '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/CDKL5/'
    # 'DIV21/src/experimental_reference_data/CDKL5-E6D_T2_C1_05212024_240611_M06844_Network_000076_network_metrics_well000.npy'
    '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5'
    '/DIV21/network_metrics/CDKL5-E6D_T2_C1_05212024_240611_M06844_Network_000076_network_metrics_well000.npy'    
    )
# ===================================================================================================

def test_sensitivity_analysis(
    sim_data_path,
    output_dir,
    reference_data_path,
    run_analysis = True,
    plot_analysis = True,
    plot_grid = True,
    plot_heatmaps = True,
    levels = 6
    ):

    # Prepare paths, pre-run checks, assert paths are correct
    assert os.path.exists(sim_data_path), f"sim_data_path does not exist: {sim_data_path}"
    assert os.path.exists(reference_data_path), f"reference_data_path does not exist: {reference_data_path}"
    if os.path.exists(output_dir):
        print(f"output_dir exists: {output_dir}")
    else:
        os.makedirs(output_dir)
        print(f"output_dir created: {output_dir}")

    # run sensitivity analysis
    if run_analysis:
        start = time.time()        
        from RBS_network_models.CDKL5.DIV21.src.conv_params import conv_params
        from RBS_network_models.CDKL5.DIV21.src.fitness_targets import fitnessFuncArgs
        run_sensitivity_analysis(
            sim_data_path, 
            output_dir,
            plot = True,
            conv_params = conv_params,                              # NOTE: this is only needed if plotting is requested,
            fitnessFuncArgs = fitnessFuncArgs,                      # NOTE: this is only needed if plotting is requested,
            
            reference_data_path = reference_data_path,              # NOTE: this is only needed if plotting is requested, 
                                                                    #       plots are generated with comparison to reference data
            option='parallel',                                     # NOTE: this if commented out, the default is 'sequential' - which is better for debugging obvs
            num_workers=32,                                         # NOTE: this is only needed if option='parallel'. I'm choosing 2 workers for 2 simulations per node.
            duration_seconds=20,
            #levels=6,                                              # NOTE: this is the number of levels for the sensitivity analysis
                                                                    # ideally, use even numbers, so original value is in the middle
            levels=levels,                                          # NOTE: this is the number of levels for the sensitivity analysis
            upper_bound=5,                                          # NOTE: this is the upper bound for the sensitivity analysis
            lower_bound=0.1,                                        # NOTE: this is the lower bound for the sensitivity analysis
            #debug=True,                                             # NOTE: if true (if uncommented), will load old data and not run new simulations
            try_loading=False,                  # NOTE: if true (if uncommented), will load old data and not run new simulations
            )
        print('Sensitivity analysis successfully ran!')
        print(f"Time taken: {time.time() - start} seconds")

    # plot sensitivity analysis
    if plot_analysis:
        start = time.time()
        #simulation_output_dir = output_dir + '/simulations'
        plot_sensitivity_analysis(
            sim_data_path,
            output_dir,
            #num_workers=32,                # NOTE: only used to load network_metrics data...for now # aw 2025-01-11 17:01:52
            #num_workers=64,                 # these are fairly quick, so I think I can increase the number of workers without disturbing login node too much     
            levels=levels,                  # NOTE: this is only used to assert that permutation counting is working correctly
            plot_grid=plot_grid,            # NOTE: this plots summary plots as a grid for all permutations involved in the sensitivity analysis.
            plot_heatmaps=plot_heatmaps     # NOTE: this plots heatmaps for all params involved in the sensitivity analysis.                     
            )
        print('Sensitivity analysis plots successfully generated!')
        print(f"Time taken: {time.time() - start} seconds")
test_sensitivity_analysis(
    sim_data_path,
    output_dir,
    reference_data_path,
    run_analysis = False,       # NOTE: Default is True, uncomment to skip
                                # TODO: Need to fix/verify that changes to cfg are properly passed to netParams and net before running the simulation
    #plot_analysis = False,     # NOTE: Default is True, uncomment to skip
    plot_grid = False,          # NOTE: Default is True, uncomment to skip. This plots summary plots as a grid for all permutations involved in the sensitivity analysis.
    #plot_heatmaps = False,     # NOTE: Default is True, uncomment to skip. This plots heatmaps for all params involved in the sensitivity analysis.
    )
# ===================================================================================================
# python command to run this script and save the output to output and error log files
'''
cd */RBS_network_models
python \ 
    /RBS_network_models/RBS_network_models/tests/_4_test_sensitivity_analysis.py \
    > /RBS_network_models/tests/outputs/test_sensitivity_analysis_levels/_test_sensitivity_analysis_levels_out.log \
    2> /RBS_network_models/tests/outputs/test_sensitivity_analysis_levels/_test_sensitivity_analysis_levels_err.log
'''