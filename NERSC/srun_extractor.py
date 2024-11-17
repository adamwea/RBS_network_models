import sys
import re
import os

def append_args_to_file(args, output_file):
    # Open the file in append mode
    with open(output_file, "a+") as file:
        # Read the contents of the file
        file.seek(0)
        payload = file.read()

        # Extract the current gen from the payload
        current_gen = re.search(r'gen_(\d+)', payload)
        if current_gen:
            current_gen = int(current_gen.group(1))
        else:
            current_gen = None
        print("Current gen:", current_gen)

        # Extract the incoming gen from the arguments
        incoming_gen = None
        for arg in args:
            #if arg.startswith('srun'):
            gen_number = re.search(r'gen_(\d+)', arg)
            if gen_number:
                incoming_gen = int(gen_number.group(1))
                print("Incoming gen:", incoming_gen)
                break

        # Clear the payload if the incoming gen is different
        if incoming_gen is not None and incoming_gen != current_gen:
            file.seek(0)
            file.truncate()

        # Write each argument to a new line in the file
        for arg in args:
            if arg.startswith('srun'):
                file.write(' '.join(arg.split()))  # Add spaces between args
                file.write(" ")  # Add a new line after each srun command
            else:
                file.write(arg)
                file.write(" ")  # Add a new line after each srun command
            #get current candidate
            if 'cand_' in arg:
                cand_num = arg.split('cand_')[1].split('_')[0]
                #cand = f'cand_{cand_num}'
                cand = cand_num
        #at the end of the line add bash to write the output and errors to gen_x_cand_y.out and gen_x_cand_y.err
        #file.write(f'> gen_{incoming_gen}_cand_{cand}.out 2> gen_{incoming_gen}_cand_{cand}.err')
        #get path by getting root directory of *_cfg.json, a file which should be one of the args in the srun command
        for arg in args:
            if '_cfg.json' in arg:
                path = os.path.dirname(arg)
                
                #set path for output and error files
                file.write(f'-u') # tell output to buffer in real time
                file.write(" ")  # Add a new line after each srun command
                output = f'{path}/gen_{incoming_gen}_cand_{cand}.run'
                output = output.split('=')[1] #split at = in string and take part after it
                error = f'{path}/gen_{incoming_gen}_cand_{cand}.err'
                error = error.split('=')[1]
                file.write(f'>> {output} 2> {error}') 
                break
        file.write("\n")  # Add a new line after each srun command

    print("Arguments have been appended to", output_file)

def check_if_data_exists(args):
    #find the arg that's a file path with _cfg.json in the name
    for arg in args:
        if '_cfg.json' in arg:
            #print(arg)
            #remove 'simConfig=' from the beginning of the string
            arg = arg.split('simConfig=')[1]
            #print(arg)
            data_path = os.path.dirname(arg)
            #print(data_path)

            #check if each _cfg.json file has a corresponding _data.json and _Fitness.json file
            all_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
            cfg_files = [f for f in all_files if '_cfg.json' in f]
            data_files = [f for f in all_files if '_data.json' in f]
            fitness_files = [f for f in all_files if '_Fitness.json' in f]
            #print(len(cfg_files) == len(data_files) == len(fitness_files))
            #sys.exit()
            try: assert len(cfg_files) == len(data_files) == len(fitness_files), 'Data files are missing, this generation is incomplete'
            except: return False
            return True

if __name__ == "__main__": 
    # Get all args after 'python srun_extractor.py' and pass them to a .txt file
    args = sys.argv[1:]
    # Create a new text file to store the arguments
    output_file = "commands.txt"
    #check that generation is complete
    generation_complete_status = check_if_data_exists(args)
    # Append the arguments to the text file only if the generation is incomplete
    if generation_complete_status is False: append_args_to_file(args, output_file)


