# Run: 241126_Run2_improved_netparams
#NOTE: I initially used chatGPT to turn intial notes into this format. I then manually edit the comments and choose to seed or not seed based on analysis of the data.
# Run: 241126_Run2_improved_netparams
seed_paths = [
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_58_data.json",
        "comment": "Bad fitness, too much constant activity - but both populations are firing and appear to oscillate.",
        "seed": False
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_26_data.json",
        "comment": "Not a lot of bursting. Decent fit, probably better than it should be. Both populations are firing and appear to oscillate.",
        "seed": True
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_13_data.json",
        "comment": "Best activity so far. Both populations are firing and appear to oscillate. Bursts vary in size and frequency. Amplitudes are closer to the target.",
        "seed": True
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_32_data.json",
        "comment": "Good activity. Both populations are firing and appear to oscillate. Bursts are fairly consistent. Amplitudes are similar to the last seed.",
        #"seed": True
        "seed": False # switched to false after further review. E rates are too low. I rates are too high. Oscillations are too regular. Amplitude is too high.
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_64_data.json",
        "comment": "Good activity. Both populations are firing and appear to oscillate. Bursts are the most defined here.",
        "seed": True
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_28_data.json",
        "comment": "Good activity. Both populations are firing and appear to oscillate. Too much constant activity, no obvious bursting.",
        "seed": False
        #"seed": True   # switched to true after further review. E rates are higher than I rates. This is still a pretty bad simulation. Constant activity is too high. 
                        # no real bursting.
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_5_data.json",
        "comment": "Way too little excitatory activity. They're bursting a little. Oscillations are nice, bursts have good frequency but not amplitude. Best frequency of bursting so far.",
        "seed": False
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_73_data.json",
        "comment": "Good oscillations. Too much constant activity. Not enough bursting.",
        "seed": False
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_35_data.json",
        "comment": "Good oscillations. Too much constant activity. Not enough bursting.",
        "seed": False
        #"seed": True # consider this on third pass...
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_66_data.json",
        "comment": "Almost identical to the last seed. Good oscillations. Too much constant activity. Not enough bursting.",
        "seed": False
        #"seed": True # consider this on third pass...
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_6_data.json",
        "comment": "Really interesting activity. Neurons appear to fire in a way that is stochastic but also rhythmic. Amplitude is exactly where it should be, but the activity is still too constant. Activity konks out partway through the simulation.",
        #"seed": False
        "seed": True # no I want to include this for its naturalistic activity. E/I ratio is good. Only issue is the konking out.
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_1/gen_1_cand_19_data.json",
        "comment": "Super consistent bursting. Too much constant activity. Amplitude is too high.",
        "seed": False
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_39_data.json",
        "comment": "Probably the best activity so far. E rates are higher than I rates. Activity looks erratic and more natural than others. Amplitude is acceptable. One of the best fits so far.",
        "seed": True
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_79_data.json",
        "comment": "Nice clear oscillations. Too much constant activity. Bursts are super non-distinct from background activity. Amplitude is too high. I think I won't use this one.",
        "seed": False
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_47_data.json",
        "comment": "Only a little better than the last one.",
        "seed": False
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_31_data.json",
        "comment": "Very active. Too much constant activity. Burst timing is good, but amplitude is too high. Bursting with respect to background might be okay. Not sure if I'll use this one.",
        "seed": False
        #"seed": True # consider this on third pass...
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_41_data.json",
        "comment": "Similar to the best one so far in essence. But it's much worse at the same time. No recognizable bursting. I think underlying spiking here is close to correct, but bursts are just not emerging. I will probably use this one.",
        "seed": True
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_73_data.json",
        "comment": "Same notes as above basically, but better bursting and general network activity shape.",
        "seed": True
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_80_data.json",
        "comment": "Similar to the last two but not as good.",
        #"seed": False
        'seed': True # keep this one, it's among the naturalistic ones.
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_2/gen_2_cand_60_data.json",
        "comment": "Probably the best activity so far. E rates are higher than I rates. Activity looks erratic and more natural than the others. Bursts are fairly distinct from background activity. Amplitude is acceptable, only a little high. Very comparable to experimental data in terms of general shape.",
        "seed": True
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_4/gen_4_cand_51_data.json",
        "comment": "E rates are not where they should be. I rates are too high - but nice oscillation with higher than wanted amplitude.",
        "seed": False
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_4/gen_4_cand_72_data.json",
        "comment": "Meh. Not great, but maybe good enough to seed.",
        #"seed": True
        "seed": False # switched to false after further review. too active. too much constant activity. bursts are very regular. amplitude is too high.
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_3/gen_3_cand_6_data.json",
        "comment": "Better E rates. I rates are probably too high. Oscillations are distinct but far too regular, not very natural. Amplitude is too high. I think I will use this one.",
        "seed": True
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_3/gen_3_cand_53_data.json",
        "comment": "Super interesting activity. More natural-looking than most. But it konks out partway through the simulation. When it is firing, it's too constant, too noisy.",
        # "seed": False
        "seed": True # keep this one, it's also among the naturalistic ones. despite the konking out.
    },
    {
        "path": "/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams/gen_3/gen_3_cand_15_data.json",
        "comment": "Super meh. Not great, but maybe good enough to seed. Very regular oscillations. Too much constant activity.",
        "seed": True
    }
]

'''functions'''

def plot_candidate_params(seed_params):
    """
    Plot parameters for all candidates on shared lines, normalizing values for visualization.
    Each parameter spans the same length, with actual (non-normalized) values displayed.
    """
    import matplotlib.pyplot as plt
    import os

    # Prepare figure
    all_params = list({param for value in seed_params.values() for param in value['param_space']})
    fig, ax = plt.subplots(figsize=(10, len(all_params) * 0.7))  # Adjust height for compact spacing

    # Normalize min, max, and values for plotting
    param_ranges = {}
    for param_key in all_params:
        values = []
        for seed_data in seed_params.values():
            params = seed_data['param_space']
            if param_key in params:
                param_value = params[param_key]
                if isinstance(param_value, list):
                    values.extend(param_value[:2])  # Min and max
                else:
                    values.append(param_value)
        param_ranges[param_key] = (min(values), max(values)) if values else (0, 1)

    # Plot parameters
    y_ticks = list(range(len(all_params)))
    y_tick_labels = all_params

    for param_idx, param_key in enumerate(all_params):
        y_position = param_idx  # Shared y-axis position for this parameter
        min_range, max_range = param_ranges[param_key]

        for seed_key, seed_data in seed_params.items():
            params = seed_data['param_space']
            seed_bool = seed_data['seed']
            if param_key in params:
                param_value = params[param_key]

                if isinstance(param_value, list):  # Min, Max, Value provided
                    min_val, max_val, val = param_value
                    if max_range != min_range:
                        norm_min = (min_val - min_range) / (max_range - min_range)
                        norm_max = (max_val - min_range) / (max_range - min_range)
                        norm_val = (val - min_range) / (max_range - min_range)
                    else:
                        norm_min = norm_max = norm_val = 0.5  # Centered normalization for equal ranges

                    ax.plot([0, 1], [y_position, y_position], 'k-', lw=1, label='_nolegend_')  # Line for normalized range
                    
                    #plot seeded values in red
                    if not seed_bool: ax.plot(norm_val, y_position, 'bo', label='_nolegend_')  # Blue dot for normalized value
                    else: ax.plot(norm_val, y_position, 'ro', label='_nolegend_')  # Red dot for normalized value

                    # Annotate with only min and max values
                    ax.text(0, y_position - 0.15, f"{min_val:.2f}", fontsize=8, ha='right', va='center')
                    ax.text(1, y_position - 0.15, f"{max_val:.2f}", fontsize=8, ha='left', va='center')
                else:  # Only Value provided
                    val = param_value
                    if max_range != min_range:
                        norm_val = (val - min_range) / (max_range - min_range)
                    else:
                        norm_val = 0.5  # Centered normalization for equal ranges

                    ax.plot([0, 1], [y_position, y_position], 'k-', lw=1, label='_nolegend_')  # Line for normalized range
                    # Annotate single value in the center and slightly above the line
                    ax.text(0.5, y_position - 0.15, f"{val:.2f}", fontsize=8, ha='center', va='center', color='red')

    # Configure the plot
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel("Normalized Parameter Value")
    ax.set_title("Candidate Parameter Review (Normalized)")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.invert_yaxis()  # Invert y-axis for better readability
    plt.tight_layout()

    # Save the plot
    seed_params_any_path = list(seed_params.values())[0]['path']
    run_path = os.path.dirname(os.path.dirname(seed_params_any_path))
    run_seed_review_dir = os.path.join(run_path, "seed_review")
    
    if not os.path.exists(run_seed_review_dir):
        os.makedirs(run_seed_review_dir)
    
    plt.savefig(os.path.join(run_seed_review_dir, "seed_params_normalized.pdf"))
    plt.savefig(os.path.join(run_seed_review_dir, "seed_params_normalized.png"))

'''main script'''
if __name__ == "__main__":   
    #setup environment
    from setup_environment import set_pythonpath
    set_pythonpath()
    import os
    import json

    # get current parameter space
    #from workspace.RBS_network_simulations.workspace.optimization_projects.CDKL5_DIV21 import parameter_spaces
    from workspace.optimization_projects.CDKL5_DIV21_dep.parameter_spaces._0_init_evol_parameter_space import params
    #param_space = CDKL5_DIV21.parameter_spaces
    #import _0_init_evol_parameter_space as params
    # from workspace.RBS_network_simulations.workspace.optimization_projects.CDKL5_DIV21.parameter_spaces import *
    # import _0_init_evol_parameter_space as params

    # review selected param space
    sim_config_dict = {}
    seed_params = {}
    for seed_path in seed_paths:
        #print a space for readability
        print("\n")
        
        # load the data
        seed_path["path"] = os.path.abspath(seed_path["path"])
        with open(seed_path["path"], "r") as f:
            seed_path["data"] = json.load(f)
        
        # print the path and comment    
        print(f"Loaded {seed_path['path']}")
        #print(f"Comment: {seed_path['comment']}")
        
        # add the sim_config to the sim_config dict
        sim_config = seed_path["data"]["simConfig"]
        sim_label = seed_path["data"]["simConfig"]["simLabel"]
        sim_config_dict[sim_label] = sim_config
        seed_path["sim_config"] = sim_config
        print(f"Sim config for {sim_label} collected.")
        #print(f"Netparams: {sim_config}")
        
        #cross reference params and sim_config, get three values for each param [min, max, value]
        #min and max are in the params dict, each key has a two item list, [0] is min, [1] is max
        #value is in the sim_config dict, each key has a value
        param_cross_dict={}
        param_cross_dict['param_space'] = {}
        for key in sim_config.keys():
            if key in params.keys():
                if type(params[key]) == list:
                    param_cross_dict['param_space'][key] = [params[key][0], params[key][1], sim_config[key]]
                else:
                    param_cross_dict['param_space'][key] = sim_config[key]
        
        param_cross_dict['path'] = seed_path["path"]
        param_cross_dict['comment'] = seed_path["comment"]
        param_cross_dict['seed'] = seed_path["seed"]
        seed_path["seed_params"] = param_cross_dict.copy()
        seed_params[sim_label] = param_cross_dict.copy()
        print(f"Seed params for {sim_label} collected.")
        
    # plot the parameter data for review
    plot_cand_params = True
    if plot_cand_params:
        plot_candidate_params(seed_params)
        


    
    
    
    
    
    
    
