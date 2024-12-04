#run: 241126_Run2_improved_netparams
seed_paths = [
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_58_data.json',
        # bad fitness, too much constant activity - but both populations are firing and appear to oscillating
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_26_data.json',
        # not a lot of bursting. decent fit. probably better than it should be.
        # both populations are firing and appear to oscillating
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_13_data.json',
        # best activity so far. both populations are firing and appear to oscillating
        # bursts appear to vary in size and frequency
        # amplitudes are closer to the target
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_32_data.json',
        # good activity. both populations are firing and appear to oscillating
        # bursts are fairly consistent 
        # amplitudes is similar to the last seed
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_64_data.json',
        # good activity. both populations are firing and appear to oscillating
        # bursts are the most defined here
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_28_data.json',
        # good activity. both populations are firing and appear to oscillating
        # too much constant activity, no obvious bursting
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_5_data.json',
        # way too little excitatory activity. but they're bursting a little bit. oscillations are nice. 
        # too much constant activity, bursts have good frequency, but not good amplitude
        # best frequency of bursting so far I guess
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_73_data.json',
        # good oscillations. too much constant activity. not enough bursting.
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_35_data.json'
        # good oscillations. too much constant activity. not enough bursting. 
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_66_data.json'
        # almost identical to the last seed. good oscillations. too much constant activity. not enough bursting.
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_6_data.json'
        # really interesting activity. neruons apppear to fire in a way that is stochastic but also rhythmic.
        # amplitude is exactly where it should be, but the activity is still to constant, I think.
        # activity konks out partway through the similaution...which is not ideal.
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_19_data.json'
        #super consistent bursting.
        # too much constant activity.
        # amplitude is too high.
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_39_data.json'
        # probably the best activity so far.
        # E_rates are higher than I rates. activity looks erratic and more nautral than the others.
        # amplitdue is acceptable.
        # one of the best fits so far.
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_79_data.json'
        # nice clear oscillations. too much constant activity. bursts are super non-distinct from background activity.
        # amplitude is too high.
        # I think I won't use this one.
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_47_data.json'
        # only a little bettter than the last one.
        
    ' /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_31_data.json'
        # very active. too much constant activity. burst timing is good, but amplitude is too high.
        # bursting with respect to background might be ok?
        # not sure if I'll use this one.
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_41_data.json'
        # similar to the best one so far in essense...in my intuitive pattern recognition.
        # but its much worse at the same time. no recognizable bursting.
        # I think underlying spiking here is close to correct, but bursts are just not emerging.
        # I will probably use this one.
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_73_data.json'
        # same notes as above basically, but better bursting and genreal network activity shape.
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_80_data.json'
        # similar to the last two but not as good.
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_60_data.json'
        # probably the best activity so far.
        # E_rates are higher than I rates. activity looks erratic and more nautral than the others.
        # bursts are fairly distinct from background activity.
        # amplitude is acceptable only a little high.
        # very comparable to experimental data in terms of general shape.
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_4/gen_4_cand_51_data.json'
        # E_rates are not where they should be. I_rates are too high - but nice oscillation with higher than wanted amplitude.
        
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_4/gen_4_cand_72_data.json'
    
        #meh. not great. but maybe good enough to seed.
        
    #data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_3/gen_3_cand_6_data.json
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_3/gen_3_cand_6_data.json'
        # better E_rates.
        # I_rates are probably too high.
        # oscillations are distinct but far too regular...not very natural.
        # amplitude is too high.
        # I think I will use this one.
        
    
    #data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_3/gen_3_cand_53_data.json
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_3/gen_3_cand_53_data.json'
        # super interesting activity. more natural looking than most. 
        # but again it konks out partway through the simulation.
        # and when it is firing, its too constant. too noisy.
        
    #data_path: /pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_3/gen_3_cand_15_data.json
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_3/gen_3_cand_15_data.json'
        # super meh. not great. but maybe good enough to seed.
        # very regular oscillations. too much constant activity.   
]