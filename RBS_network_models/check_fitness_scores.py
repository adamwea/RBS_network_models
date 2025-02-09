import os
import json

path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/batch_runs/batch_2025-02-06'

#get all fitness scores
fitness_scores = {}
for root, dirs, files in os.walk(path):
    if '/gen_' in root:
        for file in files:
            if file.endswith('_fitness.json'):
                # fitness_scores.append(os.path.join(root, file))
                #get everything before _fitness.json to use as key. This is candidate and gen info
                key = os.path.splitext(file)[0] #remove .json
                key = key.replace('_fitness', '') #remove _fitness
                gen_number = key.split('_')[1]
                cand_number = key.split('_')[3]
                json_path = os.path.join(root, file)
                data = json.load(open(json_path))
                if gen_number not in fitness_scores:
                    fitness_scores[gen_number] = {}
                    if cand_number not in fitness_scores[gen_number]:
                        fitness_scores[gen_number][cand_number] = {}                
                fitness_scores[gen_number][cand_number] = {
                    'data': data,
                    'average_fitness': data['average_fitness'],
                    'pickle_path': json_path.replace('_fitness.json', '_data.pkl'),
                }
                
                
# just add a place to pause and look at the fitness scores
# sort fitness_scores by gen and cand in key names
fitness_scores = {k: fitness_scores[k] for k in sorted(fitness_scores.keys())}

#convert keys expressed as keys to ints
for gen in fitness_scores:
    fitness_scores[gen] = {int(k): fitness_scores[gen][k] for k in sorted(fitness_scores[gen].keys())}

# sort by numeric order
for gen in fitness_scores:
    fitness_scores[gen] = {k: fitness_scores[gen][k] for k in sorted(fitness_scores[gen].keys())}
    
# print warning if there are any numbers missing from the candidate numbers in sequence
for gen in fitness_scores:
    for cand in fitness_scores[gen]:
        max_cand = max(fitness_scores[gen].keys()) + 1
        if cand not in range(0, max_cand):
            print(f'WARNING: Candidate {cand} is missing from generation {gen}')
        # print warning if 'average_fitness' is not an int or float
        print(fitness_scores[gen][cand]['average_fitness'])
        if not isinstance(fitness_scores[gen][cand]['average_fitness'], (int, float)):
            print(f'WARNING: average_fitness for candidate {cand} in generation {gen} is not a number')

# for gen in fitness_scores:
#     fitness_scores[gen] = {k: fitness_scores[gen][k] for k in sorted(fitness_scores[gen].keys())}


#check for corrupted pickle files
import dill
for gen in fitness_scores:
    for cand in fitness_scores[gen]:
        pickle_path = fitness_scores[gen][cand]['pickle_path']
        print(f'Checking {pickle_path}...')
        if not os.path.exists(pickle_path):
            print(f'WARNING: Pickle file {pickle_path} does not exist')
        # attempt to load, print warning if it fails
        try:
            with open(pickle_path, 'rb') as f:
                data = dill.load(f)
        except Exception as e:
            print(f'WARNING: Pickle file {pickle_path} is corrupted: {e}')



#print(fitness_scores)
                
                