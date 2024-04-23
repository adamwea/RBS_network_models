import os
import netpyne
import subprocess

#get output directory
output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')

#os walk through output path and find all .json files with containing fitness in the name
for root, dirs, files in os.walk(output_path):
    for file in files:
        if '.json' in file and 'Fitness' in file:
            #print(file)
            #with open(os.path.join(root, file), 'r') as f:
            #    print(f.read())
            #print('\n')
            #replace '_Fitness' with '_data' in the file name
            data_file_path = os.path.join(root, file.replace('_Fitness', '_data'))
            #check if data file_path exists
            if os.path.exists(data_file_path): pass
            else: continue
                #print(data_file_path)
            #create glob path
            glob_path = os.path.join(root, file)
            
            #load the data using git lfs as needed
            subprocess.run(['git', 'lfs', 'fetch', f'--include=[data_file_path]'])

            #load the data file using netpyne loadall
            data = netpyne.sim.loadAll(data_file_path)
            print(data)
            