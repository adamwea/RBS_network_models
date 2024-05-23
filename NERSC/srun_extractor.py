import sys
import re

# Get all args after 'python srun_extractor.py' and pass them to a .txt file
args = sys.argv[1:]

# Create a new text file to store the arguments
output_file = "commands.txt"

# Open the file in append mode
with open(output_file, "a+") as file:

    # Write each argument to a new line in the file
    for arg in args:
        if arg.startswith('srun'):
            #gen_number = re.search(r'gen_(\d+)', arg)
            #if gen_number:
                #file.write(f"gen: {gen_number.group(1)}\n")
            file.write(' '.join(arg.split()))  # Add spaces between args
            file.write(" ")  # Add a new line after each srun command
        else:
            file.write(arg)
            file.write(" ")  # Add a new line after each srun command
    file.write("\n")  # Add a new line after each srun command

print("Arguments have been appended to", output_file)
