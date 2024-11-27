import os
import json
import re
import analysis_functions as af

simulation_run_paths = [
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241126_Run2_improved_netparams',
]

fitness_data = []

# Collect fitness data
for simulation_run_path in simulation_run_paths:
    for root, dirs, files in os.walk(simulation_run_path):
        for file in files:
            if file.endswith('_fitness.json'):
                fitness_path = os.path.join(root, file)
                with open(fitness_path, 'r') as f:
                    fitness_content = json.load(f)
                    average_fitness = fitness_content.get('average_fitness', float('inf'))
                    fitness_data.append((average_fitness, root))

# Sort by average fitness (least to greatest)
fitness_data.sort()

# Analyze simulations in order of fitness
for average_fitness, simulation_path in fitness_data:
    for root, dirs, files in os.walk(simulation_path):
        for file in files:
            if file.endswith('_data.json'):
                data_path = os.path.join(root, file)
                try:
                    af.analyze_simulation_data(data_path)
                except Exception as e:
                    print(f'Error analyzing data for file: {file}, Error: {e}')
                    
#how to run on interacive node in perlmutter command line:
#salloc --nodes=1 --ntasks-per-node=256 -C cpu -q interactive -t 04:00:00
#module load conda
#conda activate netsims_env
#srun -n 256 python generate_update_slides_v3.py
