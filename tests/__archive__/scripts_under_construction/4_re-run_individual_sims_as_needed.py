global PROGRESS_SLIDES_PATH, SIMULATION_RUN_PATH, REFERENCE_DATA_NPY, CONVOLUTION_PARAMS
from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import *
from fitness_helper import *
from fitting.calculate_fitness_vCurrent import fitnessFunc
add_repo_root_to_sys_path()  
# ===================================================================================================

'''main script'''
import json
import os
# Main entry point
def main():
    #for debugging, print path
    import sys
    import pprint

    #pprint.pprint(sys.path)
    
    # get dir of this script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # declare cfg runtime kwargs    
    cfg_script_path = './batching/241211_cfg.py'
    param_script_path = './fitting/evol_parameter_spaces/241202_adjusted_evol_params.py'
    target_script_path = './fitting/experimental_data_features/fitness_args_20241205_022033.py'
    cfg_script_path = os.path.join(script_dir, cfg_script_path)
    param_script_path = os.path.join(script_dir, param_script_path)
    target_script_path = os.path.join(script_dir, target_script_path)
    cfg_kwargs = {
        'duration_seconds': 30,
        'param_script_path' : param_script_path,
        'target_script_path' : target_script_path,
    }
    batching_dir = os.path.join(script_dir, 'batching')
    
    #save .json file so cfg.py can read it
    with open(f'{batching_dir}/cfg_kwargs.json', 'w') as f:
        json.dump(cfg_kwargs, f, indent=4)    

    sim_data_path = SIMULATION_RUN_PATH
    sim_cfg_path = sim_data_path.replace('_data.json', '_cfg.json')
    sim_netparam_path = sim_data_path.replace('_data.json', '_netParams.json')
        
    with open(sim_data_path, 'r') as f:
        sim_data = json.load(f)
        
    net = sim_data['net']
    netParams = net['params']
            
    from netpyne import sim
    #sim.loadSimCfg(sim_data_path)
    #import batching.cfg as cfg
    # netParams_script_path = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/batching/netParams.py'
    #simConfig, netParams = sim.readCmdLineArgs(
    #     simConfigDefault=cfg_script_path, 
    #     netParamsDefault=netParams_script_path
    #     )
    
    #this only works if the cfg.py script is in the same directory as this script
    #cfgModule = sim.loadPythonModule(cfg_script_path)
    #old_cfg = cfgModule.cfg
    #sim.cfg = cfg
    
    # load netparams from sim_data_path
    #netParams = sim.loadNetParams(sim_data_path, setLoaded=False)
    
    #create sim w cfg
    #sim.create(netParams=netParams, simConfig=cfg)

    #sim.loadSimCfg(cfgModule)
    #sim.loadSimCfg(None, data = cfg)
    
    #load net from sim_data_path
    #sim.loadSimCfg(sim_data_path)
    cfg = sim.loadSimCfg(sim_cfg_path, setLoaded=False)
    #netParams = sim.loadNetParams(sim_data_path, setLoaded=False)
    netParams = sim.loadNetParams(sim_netparam_path, setLoaded=False)  
    
    #edit cfg as needed
    cfg.dump_coreneuron_model = False
    cfg.cvode_active = False
    cfg.coreneuron = False
    cfg.coreneuron = True
    cfg.use_fast_imem = False
    cfg.allowSelfConns = False
    cfg.oneSynPerNetcon = True
    cfg.validateNetParams = True
    cfg.printSynsAfterRule = True
    
    #cfg.coreneuron = True
    #cfg.dump_coreneuron_model = True
    # cfg.cache_efficient = True
    # #cfg.cvode_active = True
    # cfg.use_fast_imem = True
    # cfg.allowSelfConns = True
    # cfg.oneSynPerNetcon = False
    
    #Modify cell params to fix issues...
    # netParams.cellParams['E']['secs']['soma']['geom']['L'] = 5
    # netParams.cellParams['E']['secs']['soma']['geom']['diam'] = 5
    # netParams.cellParams['E']['secs']['soma']['geom']['Ra'] = 100

    # initialize with net
    sim.loadNet(sim_data_path)
    net = sim.net.__dict__.copy()
    #sim.gatherData() remove sim.attributes that come fram sim.gatherData()
    # Remove attributes that come from sim.gatherData()
    if hasattr(net, 'allCells'):
        del net.allCells
    if hasattr(net, 'allPops'):
        del net.allPops
    #sim.initialize(netParams=netParams, simConfig=cfg, net=net)
    sim.initialize(netParams=netParams, simConfig=cfg)
    sim.net.pops = net['pops']
    sim.net.cells = net['cells']
    simData = sim.setupRecording()
    print("Simulation initialized.")
    
    # recreate net, i.e. ignore current net
    #sim.create(netParams=netParams, simConfig=cfg)
    
    # # Convert SimConfig objects to dictionaries
    # old_cfg_dict = old_cfg.__dict__.copy()
    # new_cfg_dict = new_cfg.__dict__.copy()
    
    # # Find keys where new and old cfg differ
    # common_keys = old_cfg_dict.keys() & new_cfg_dict.keys()
    # diff_keys = {k: (old_cfg_dict[k], new_cfg_dict[k]) for k in common_keys if old_cfg_dict[k] != new_cfg_dict[k]}
    # pprint.pprint(diff_keys)
    
    import sys    
    sim.simulate()
    print("Simulation complete.")
    

'''notes and main script'''
# ===============================================================================================================================
# NOTE: This script is also meant to act as a very basic analysis log. Look for corresponding notes in aw obsidian notes.
# ===============================================================================================================================

SIMULATION_RUN_PATH = (    
    # '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/'
    # 'CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_3/gen_3_cand_173_data.json'
    
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/'
    'CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_11_data.json'
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
