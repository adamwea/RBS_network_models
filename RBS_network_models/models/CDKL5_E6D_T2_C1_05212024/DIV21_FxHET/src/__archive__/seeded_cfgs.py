seeded_cfg_paths = {
    
    # older paths manually pulled from 
    # /pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/
    # CDKL5_DIV21/src/old_analysis_params/seed_selection_scripts/
    # 241126_Run2_improved_netparams.py
    #'/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_26_data.json',
    'old_seed_selections': {
        #'base_path': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/outputs/',
        'base_path': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/CDKL5/DIV21/outputs/',
        'paths': [
            '241126_Run2_improved_netparams/gen_1/gen_1_cand_26_data.json',
            '241126_Run2_improved_netparams/gen_1/gen_1_cand_13_data.json',
            '241126_Run2_improved_netparams/gen_1/gen_1_cand_64_data.json',
            '241126_Run2_improved_netparams/gen_1/gen_1_cand_28_data.json',
            '241126_Run2_improved_netparams/gen_1/gen_1_cand_66_data.json',
            '241126_Run2_improved_netparams/gen_1/gen_1_cand_6_data.json',
            '241126_Run2_improved_netparams/gen_2/gen_2_cand_39_data.json',
            '241126_Run2_improved_netparams/gen_2/gen_2_cand_31_data.json',
            '241126_Run2_improved_netparams/gen_2/gen_2_cand_73_data.json',
            '241126_Run2_improved_netparams/gen_2/gen_2_cand_80_data.json',
            '241126_Run2_improved_netparams/gen_2/gen_2_cand_60_data.json',
            '241126_Run2_improved_netparams/gen_4/gen_4_cand_72_data.json',
            '241126_Run2_improved_netparams/gen_3/gen_3_cand_6_data.json',
            '241126_Run2_improved_netparams/gen_3/gen_3_cand_53_data.json',
            '241126_Run2_improved_netparams/gen_3/gen_3_cand_15_data.json',
        ]
    },
    
    #aw 2025-01-02 19:37:39 - from test_run_a_simulation.py runs
    'test_runs1': {
        #'base_path': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_run_a_simulation/',
        'base_path': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/tests/outputs/test_run_a_simulation/',
        'paths': [
            'test_run_5_data.pkl',
        ]
    },
                
    # aw 2025-01-02 20:10:47 - from test_sensitivity_analysis.py runs
    'test_sensitivity_analysis_runs1': {
        #'base_path': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_sensitivity_analysis/',
        'base_path': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/tests/outputs/test_sensitivity_analysis/',
        'paths': [
            'weightIE_increased_data.pkl',
            'probLengthConst_increased_data.pkl',
            'weightEI_reduced_data.pkl',
            'tau2_exc_increased_data.pkl',
            'probII_reduced_data.pkl',
            'probEI_reduced_data.pkl',
            # #Data path: /pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_sensitivity_analysis/weightIE_increased_data.pkl
            # '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_sensitivity_analysis/weightIE_increased_data.pkl',
            # #Data path: /pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_sensitivity_analysis/probLengthConst_increased_data.pkl
            # '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_sensitivity_analysis/probLengthConst_increased_data.pkl',
            # #Data path: /pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_sensitivity_analysis/weightEI_reduced_data.pkl
            # '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_sensitivity_analysis/weightEI_reduced_data.pkl',
            # #Data path: /pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_sensitivity_analysis/tau2_exc_increased_data.pkl
            # '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_sensitivity_analysis/tau2_exc_increased_data.pkl',
            # #Data path: /pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_sensitivity_analysis/probII_reduced_data.pkl
            # '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_sensitivity_analysis/probII_reduced_data.pkl',
            # #Data path: /pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_sensitivity_analysis/probEI_reduced_data.pkl
            # '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/tests/outputs/test_sensitivity_analysis/probEI_reduced_data.pkl',
        ],
    },
    
    # aw 2025-01-03 09:20:38 - selected from old runs. Available plots may be a little unreliable... but i dont think it's worthwhile to re-run the simulations
    'run8': {
        #'base_path': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/optimizing/CDKL5/CDKL5_DIV21/outputs/',
        'base_path': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/CDKL5/DIV21/outputs/',
        'paths': [
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_2/gen_2_cand_171_data.pkl',
            # data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_167_data.json
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_167_data.json',
            # data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_52_data.json
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_52_data.json',
            # data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_90_data.json
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_90_data.json',
            # data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_23_data.json
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_23_data.json',
            # data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_95_data.json
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_95_data.json',
            # data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_28_data.json
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_28_data.json',
            # data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_44_data.json
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_44_data.json',
            # data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_115_data.json
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_115_data.json',
            # data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_58_data.json
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_90_data.json',
            # data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_163_data.json
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_163_data.json',
            # data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_161_data.json
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_161_data.json',
            # gen_1_cand_72
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_72_data.json',
            # gen_1_cand_57
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_57_data.json',
            # gen_1_cand_148
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_148_data.json',
            # gen_1_cand_42
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_42_data.json',
            # gen_1_cand_73
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_73_data.json',
            # gen_1_cand_136
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_136_data.json',
            # gen_0_cand_144
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_144_data.json',
            # gen_0_cand_133
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_133_data.json',
            # gen_0_cand_141
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_141_data.json',
            # gen_0_cand_129
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_129_data.json',
            # gen_0_cand_159
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_159_data.json',
            # gen_0_cand_72 # TODO: Show this one to Roy
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_72_data.json',
            # gen_0_cand_176
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_176_data.json',
            # gen_0_cand_113
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_113_data.json',
            # gen_0_cand_168
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_168_data.json',
            # gen_0_cand_77 # TODO: Show this one to Roy
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_77_data.json',
            # gen_0_cand_11
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_11_data.json',
            # gen_0_cand_47
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_47_data.json',
            # gen_0_cand_111
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_111_data.json',
            # gen_0_cand_173
            '241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_173_data.json',
        ]
    },
    
    # 2025-01-03 22:26:18 - all paths above freshly added. Next batch run will derive cfgs and netparams from these paths
}