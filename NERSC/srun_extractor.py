import sys
import re

# Get all args after 'python srun_extractor.py' and pass them to a .txt file
args = sys.argv[1:]

# Create a new text file to store the arguments
output_file = "input.txt"

# Open the file in append mode
with open(output_file, "a+") as file:
    # # Read the contents of the file
    # file.seek(0)
    # payload = file.read()

    # # Extract the current gen from the payload
    # current_gen = re.search(r'gen_(\d+)', payload)
    # if current_gen:
    #     current_gen = int(current_gen.group(1))
    # else:
    #     current_gen = 0
    # print("Current gen:", current_gen)

    # # Extract the incoming gen from the arguments
    # incoming_gen = None
    # for arg in args:
    #     #if arg.startswith('srun'):
    #     gen_number = re.search(r'gen_(\d+)', arg)
    #     if gen_number:
    #         incoming_gen = int(gen_number.group(1))
    #         print("Incoming gen:", incoming_gen)
    #         break

    # # Clear the payload if the incoming gen is different
    # if incoming_gen is not None and incoming_gen != current_gen:
    #     file.seek(0)
    #     file.truncate()

    # Write each argument to a new line in the file
    for arg in args:
        if arg.startswith('srun'):
            gen_number = re.search(r'gen_(\d+)', arg)
            if gen_number:
                file.write(f"gen: {gen_number.group(1)}\n")
            file.write(' '.join(arg.split()))  # Add spaces between args
            file.write(" ")  # Add a new line after each srun command
        else:
            file.write(arg)
            file.write(" ")  # Add a new line after each srun command
    file.write("\n")  # Add a new line after each srun command

print("Arguments have been appended to", output_file)
