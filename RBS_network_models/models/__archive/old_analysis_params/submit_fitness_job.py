import sys
from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.fitness_helper import fitnessFunc

if __name__ == "__main__":
    # Extract arguments (e.g., paths to simData and kwargs)
    sim_data_path = sys.argv[1]
    kwargs_path = sys.argv[2]

    # Load simData and kwargs
    import json
    with open(sim_data_path, 'r') as f:
        simData = json.load(f)
    with open(kwargs_path, 'r') as f:
        kwargs = json.load(f)

    # Run fitness function
    fitness = fitnessFunc(**kwargs)

    # # Save fitness result
    # with open(kwargs['fitness_save_path'], 'w') as f:
    #     json.dump({"fitness": fitness}, f)

    print(f"Fitness calculation complete. Result: {fitness}")
