global PROGRESS_SLIDES_PATH, SIMULATION_RUN_PATH, REFERENCE_DATA_NPY, CONVOLUTION_PARAMS
from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import *
from fitness_helper import *
from fitting.calculate_fitness_vCurrent import fitnessFunc
add_repo_root_to_sys_path()  
# ===================================================================================================

'''main script'''
# Main entry point
def main():    

    # # Generate network summary slides and plots    
    # print("Analyzing simulations sequentially...")
    # kwargs = {
    #     'data_path': SIMULATION_RUN_PATH,
    #     #'reference': True, #this is just always true now
    #     'reference_data': REFERENCE_DATA_NPY,
    #     'convolution_params': CONVOLUTION_PARAMS,
    #     'conv_param_tunning': True,
    # }
    # generate_network_summary_slide_content(**kwargs)
    
    #
    import json
    sim_data_path = SIMULATION_RUN_PATH
    with open(sim_data_path, 'r') as f:
        sim_data = json.load(f)
    
    simData = sim_data.pop('simData')
    kwargs = sim_data
    fitness = fitnessFunc(simData=simData, mode='simulated data', **kwargs)

'''notes and main script'''
# ===============================================================================================================================
# NOTE: This script is also meant to act as a very basic analysis log. Look for corresponding notes in aw obsidian notes.
# ===============================================================================================================================

SIMULATION_RUN_PATH = (
    #"/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams"
    #data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_6_data.json
    
    #"/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/"
    #"CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_6_data.json"
    
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/'
    'CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_3/gen_3_cand_173_data.json'
    )
REFERENCE_DATA_NPY = (
    
    # "/pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/"
    # "_1_derive_features_from_experimental_data/network_metrics/network_metrics_well000.npy"
    
    "/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/_config/"
    "experimental_data_features/network_metrics/CDKL5-E6D_T2_C1_05212024_240611_M06844_Network_000076_network_metrics_well000.npy"
    )
# PROGRESS_SLIDES_PATH = (
#     # "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams"
#     # "/_network_summary_slides"
#     )   
CONVOLUTION_PARAMS = (
    #"workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/convolution_params/241202_convolution_params.py")
    
    "/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/"
    "CDKL5_DIV21/_config/convolution_params/241202_convolution_params.py"
    )

# Entry point check
if __name__ == "__main__":
    main()
