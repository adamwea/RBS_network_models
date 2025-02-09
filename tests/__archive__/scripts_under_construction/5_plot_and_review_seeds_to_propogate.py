from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import *
#import get_simulated_network_activity_metrics
add_repo_root_to_sys_path()


## Current seed script(s) selected:
paths=[
    '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/_config/seed_selection_scripts/241126_Run2_improved_netparams.py'
]

# load each seed script like a module
all_seed_paths=[]
for path in paths:
    seeds = import_module_from_path(path)
    seeds = seeds.seed_paths
    for seed in seeds:
        seed_path = seed['path']
        all_seed_paths.append(seed_path)
    
for seed in all_seed_paths:
    print(seed)
    
#===================================================================================================
'''main script'''
# Configuration parameters
#global PROGRESS_SLIDES_PATH, SIMULATION_RUN_PATHS, REFERENCE_DATA_NPY, CONVOLUTION_PARAMS
# PROGRESS_SLIDES_PATH = (
#     # "/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/"
#     # "workspace/optimization_projects/CDKL5_DIV21/"
#     # "_3_analyze_plot_review/progress_slides"
    
#     "/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/"
#     "_3_plot_all_runs_and_select_seeds/network_summary_slides"
# )
# SIMULATION_RUN_PATHS = [
#     "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/"
#     "CDKL5_DIV21/241126_Run2_improved_netparams"
# ]
# REFERENCE_DATA_NPY = (
#     "/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace"
#     "/optimization_projects/CDKL5_DIV21/_1_derive_features_from_experimental_data/"
#     "network_metrics/network_metrics_well000.npy"
# )

# Main entry point
def main(
    analyze_in_parallel=False, 
    analyze_in_sequence=False, 
    generate_presentation_in_parallel=False, 
    generate_presentation_in_sequence=False,
    progress_slides_path = None,
    simulation_run_paths = None,
    reference_data_npy = None,
    convolution_params = None,
    seed_paths = None, 
    ):
    assert progress_slides_path is not None, "Progress slides path must be provided."
    assert simulation_run_paths is not None, "Simulation run paths must be provided."
    assert reference_data_npy is not None, "Reference data path must be provided."
    
    print("Starting simulation analysis...")
    print('')
    print(f"analysis in parallel: {analyze_in_parallel}")
    print(f"analysis in sequence: {analyze_in_sequence}")
    #print(f"generate presentation: {generate_presentation}")
    print(f"generate presentation in parallel: {generate_presentation_in_parallel}")
    print(f"generate presentation in sequence: {generate_presentation_in_sequence}")
    print('') 
    
    # Determine CPU usage for parallel tasks
    number_physical_cpus = int(os.cpu_count() / 2)
    
    
    
    # Collect fitness data
    fitness_data = collect_fitness_data(SIMULATION_RUN_PATHS)
    print(f"Collected fitness data for {len(fitness_data)} simulations.")

    # Analyze simulations - parallel or sequential
    #assert analyze_in_parallel or analyze_in_sequence, "At least one analysis method must be selected."
    assert not (analyze_in_parallel and analyze_in_sequence), "Select only one analysis method."
    assert not (generate_presentation_in_parallel and generate_presentation_in_sequence), "Select only one presentation method."
    
    #sequential
    if analyze_in_sequence:
        print("Analyzing simulations sequentially...")
        analyze_simulations(
            SIMULATION_RUN_PATHS,
            reference=True,
            reference_data = REFERENCE_DATA_NPY,
        )
        
    
    #parallel
    #analyze_in_parallel = False
    if analyze_in_parallel:
        max_workers = max(1, number_physical_cpus // 4)
        print(f"Analyzing simulations in parallel with {max_workers} workers...")
        analyze_simulations_parallel(
            SIMULATION_RUN_PATHS,
            #fitness_data,
            reference=True,
            reference_data = REFERENCE_DATA_NPY,
            convolution_params = CONVOLUTION_PARAMS,
            progress_slides_path = progress_slides_path,
            seed_paths = seed_paths, #TODO: implement later
            max_workers=max_workers,
        )
    else:
        print("Skipping simulation analysis step.")

    # Generate presentation and summary plots
    #generate_presentation = True
    
    #sequential
    if generate_presentation_in_sequence:
        # #TODO: I need to reimplement this function to work with the new data structure
        # implemented = False
        # assert implemented, "This function is not yet implemented."
        print("Generating network summary slides and plots sequentially...")
        collect_network_summary_plots_parallel(
            SIMULATION_RUN_PATHS, 
            fitness_data, 
            #file_extension="pdf",
            max_workers=1
        )
    
    #parallel
    if generate_presentation_in_parallel:
        print("Generating network summary slides and plots...")
        collect_network_summary_plots_parallel(
            SIMULATION_RUN_PATHS, 
            fitness_data, 
            #file_extension="pdf", 
            seed_paths = seed_paths,
            progress_slides_path=progress_slides_path,
            max_workers=number_physical_cpus // 2
        )
    
    print("Simulation analysis complete.")

'''notes and main script'''
# ===============================================================================================================================
# NOTE: This script is also meant to act as a very basic analysis log. Look for corresponding notes in aw obsidian notes.
# ===============================================================================================================================

#SIMULATION_RUN_PATHS = []
# ok, this is working now - 2024-12-1

#run: 241126_Run2_improved_netparams
#SIMULATION_RUN_PATHS.append("/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams")
#REFERENCE_DATA_NPY = ("/pscratch/sd/a/adammwea/workspace/RBS_neftwork_simulations/workspace/optimization_projects/CDKL5_DIV21/_1_derive_features_from_experimental_data/network_metrics/network_metrics_well000.npy")
#SLIDES_OUTPUT_PATH = "/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_3_analyze_plot_review/network_summary_slides"
#PROGRESS_SLIDES_PATH = "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/_network_summary_slides"   
#CONVOLUTION_PARAMS = "workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/convolution_params/241202_convolution_params.py"

#updated pathing for seed selection scripts
SEED_PATHS = all_seed_paths
REFERENCE_DATA_NPY = ("/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/_config/experimental_data_features/network_metrics/CDKL5-E6D_T2_C1_05212024_240611_M06844_Network_000076_network_metrics_well000.npy")
PROGRESS_SLIDES_PATH = "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/_seed_selection_summary_slides"
CONVOLUTION_PARAMS = "/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/_config/convolution_params/241202_convolution_params.py"
# SIMULATINO_RUN_PATHS = all_seed_paths

#convert seed_paths to list of their parent directories
SIMULATION_RUN_PATHS = []
for seed in all_seed_paths:
    SIMULATION_RUN_PATHS.append(os.path.dirname(os.path.dirname(seed)))
SIMULATION_RUN_PATHS = list(set(SIMULATION_RUN_PATHS))

# Entry point check
if __name__ == "__main__":
    
    #Recalculate the fitness data given potential changes to convolution params
    
    # Plot Network Summaries
    main(
        #comment out the options you don't want to run, as needed
        #analyze_in_sequence=False,
        #analyze_in_parallel=True,
        #analyze_in_sequence=True,
        #analyze_in_parallel=False,
        generate_presentation_in_parallel=True,
        #generate_presentation_in_sequence=True,
        progress_slides_path = PROGRESS_SLIDES_PATH,
        simulation_run_paths = SIMULATION_RUN_PATHS,
        reference_data_npy = REFERENCE_DATA_NPY,
        convolution_params = CONVOLUTION_PARAMS,
        seed_paths = SEED_PATHS,
    )

    # plot parameter line plots
    
    # histogram of parameter values - review nfactors, make sure all seeds are comparatively better than reference.
    # make sure we're moving in the right direction.