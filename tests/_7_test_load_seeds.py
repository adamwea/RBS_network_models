import json
import os
import pickle
import re
#from DIV21.src.seeded_cfgs import seeded_cfg_paths
from RBS_network_models.developing.CDKL5.DIV21.src.seeded_cfgs import seeded_cfg_paths
from netpyne import sim
#===============================================================================
seed_dir = "/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/CDKL5/DIV21/seeds"
#===============================================================================
for run_name, run in seeded_cfg_paths.items():
    base_path = run['base_path']
    seed_paths = run['paths']
    for seed_path in seed_paths:
        def handle_seed_data(seed_path):
            
            #init
            pkl_path = None
            json_path = None
            
            # check possible file types
            is_json = seed_path.endswith('.json')
            is_pkl = seed_path.endswith('.pkl')
            assert is_json or is_pkl, "Seed path must end with '.json' or '.pkl'"
            json_path = seed_path if is_json else seed_path.replace('.pkl', '.json')
            pkl_path = seed_path if is_pkl else seed_path.replace('.json', '.pkl')
            assert json_path is not None or pkl_path is not None, "No seed data paths provided"
            seed_path = os.path.join(base_path, seed_path)
            json_path = os.path.join(base_path, json_path)
            pkl_path = os.path.join(base_path, pkl_path)
            json_exists = os.path.exists(json_path)
            pkl_exists = os.path.exists(pkl_path)
            assert json_exists or pkl_exists, f"Seed path does not exist: {seed_path}"
            
            #if seed_path.endswith('.json'):
            seed_json = None
            seed_pkl = None
            if json_exists:
                #seed_json = None
                def handle_json_seed(json_path):
                    print('loading json...')
                    with open(json_path, 'r') as f:
                        seed = json.load(f)
                    print('done')
                    return seed
                seed_json = handle_json_seed(json_path)
                cfg_json = seed_json['simConfig']
                net = seed_json['net']
                #import CDKL5.DIV21.src.cfg as cfg
                import RBS_network_models.developing.CDKL5.DIV21.src.cfg as template_cfg
                import RBS_network_models.developing.CDKL5.DIV21.src.netParams as template_netParams
                template_netParams = template_netParams.netParams
                #sim.create(netParams=net, simConfig=cfg_json)
                # netParams = sim.loadNetParams(
                #     #'/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/CDKL5/DIV21/src/netParams.py',
                #     seed_path,
                #     setLoaded=False,
                #     )
            
                template_cfg = template_cfg.cfg
                
                def translate_cfg(cfg_json, template_cfg, json_path=None, pkl_path=None, seed_dir=None):
                    
                    
                    run_name = os.path.basename(os.path.dirname(os.path.dirname(json_path)))
                    gen = os.path.basename(os.path.dirname(json_path))
                    cand = re.search(r'cand_\d+', json_path)[0]
                    
                    forced_keys = {
                        'allowSelfConns': True,
                        'simLabel': f'{run_name}_{gen}_{cand}',
                        'saveFolder': seed_dir,
                        'filename': os.path.join(seed_dir, f'{run_name}_{gen}_{cand}'),
                        'num_excite': template_cfg.num_excite,
                        'num_inhib': template_cfg.num_inhib,
                        'recordCells': template_cfg.recordCells,
                        'saveJson': False,
                        'savePickle': True, # NOTE: pkl files are just more robust and convenient than json files
                        'validateNetParams': True,
                        'analysis': template_cfg.analysis,
                    }
                    
                    def recursive_update(template_cfg, cfg_json, indent='', forced_keys=None):
                        for key, value in cfg_json.items():
                            if forced_keys is not None:
                                if hasattr(template_cfg, key) and key in forced_keys:
                                    print(f'{indent}setting:', key, forced_keys[key])
                                    setattr(template_cfg, key, forced_keys[key])
                                    continue
                                
                            if isinstance(value, dict):
                                if hasattr(template_cfg, key):
                                    print(f'{indent}recursing:', key)
                                    recursive_update(getattr(template_cfg, key), value, indent=indent + '  ', forced_keys=forced_keys)
                            else:                                
                                if hasattr(template_cfg, key):
                                    print(f'{indent}setting:', key, value)
                                    setattr(template_cfg, key, value)
                        return template_cfg
                    translated_cfg = recursive_update(
                        template_cfg, 
                        cfg_json, 
                        forced_keys=forced_keys
                        )
                    
                    #save cfg as json in seed_dir
                    # with open(os.path.join(seed_dir, f'{run_name}_{gen}_{cand}_cfg.json'), 'w') as f:
                    #     json.dump(translated_cfg, f)
                    
                    #save cfg as pkl in seed_dir
                    with open(os.path.join(seed_dir, f'{run_name}_{gen}_{cand}_cfg.pkl'), 'wb') as f:
                        pickle.dump(translated_cfg, f)
                        
                    return translated_cfg
                translated_cfg = translate_cfg(
                    cfg_json, 
                    template_cfg,
                    json_path=json_path,
                    #pkl_path=pkl_path,
                    seed_dir=seed_dir                    
                    )
                
                return translated_cfg
            elif pkl_exists:
                # def handle_pkl_seed(pkl_path):
                #     #import pickle
                #     print('loading pkl...')
                #     with open(pkl_path, 'rb') as f:
                #         seed = pickle.load(f)
                #     print('done')
                #     return seed
                # seed_pkl = handle_pkl_seed(pkl_path)
                # simConfig = sim.loadSimCfg(pkl_path, setLoaded=False)
                
                #
                #gen = os.path.basename(os.path.dirname(pkl_path))
                try:
                    #batch runs 
                    run_name = os.path.basename(os.path.dirname(os.path.dirname(pkl_path))) 
                    gen = re.search(r'gen_\d+', pkl_path)[0]
                    cand = re.search(r'cand_\d+', pkl_path)[0]
                    simLabel = f'{run_name}_{gen}_{cand}'
                    filename = os.path.join(seed_dir, f'{run_name}_{gen}_{cand}')
                except:
                    #non batch runs 
                    # run_name = gen
                    # cand = ''
                    run_name = os.path.basename(pkl_path)
                    # remove _data.pkl from run_name
                    run_name = run_name.replace('_data.pkl', '')
                    simLabel = run_name
                    filename = os.path.join(seed_dir, run_name)
                
                #load sim object
                sim.load(pkl_path)
                sim.cfg.simLabel = simLabel
                sim.cfg.filename = filename
                sim.cfg.saveFolder = seed_dir
                sim.saveData()
                print()
            print(seed_path)
        handle_seed_data(seed_path)


