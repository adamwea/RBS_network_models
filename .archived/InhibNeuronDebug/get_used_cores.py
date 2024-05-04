import subprocess
import re

# Get the list of all running processes
processes = subprocess.check_output(['ps', '-e', '-o', 'pid=']).decode().split()
process_names = subprocess.check_output(['ps', '-e', '-o', 'comm=']).decode().split()
print(process_names)

used_cores = set()

# For each running process
for pid in processes:
    process_name = process_names[processes.index(pid)]
    if "systemd" in process_name:
        continue
    try:
        # Get the CPU affinity of the process
        output = subprocess.check_output(['taskset', '-p', pid]).decode()
        
        # The output will be something like: "pid 123's current affinity mask: 3"
        # The affinity mask is a bitmask where each bit represents a CPU core
        # If a bit is set, it means the process can run on that core
        # We can convert the mask to binary and then check which bits are set
        mask = int(re.search('affinity mask: (.*)', output).group(1), 16)
        cores = [i for i in range(mask.bit_length()) if mask & (1 << i)]
        
        if len(cores) < 16:
            print(pid, process_name, cores)
            pass
        
        # Add the cores to the set of used cores
        used_cores.update(cores)
    except:
        pass

# Convert the set of used cores to a comma-separated string
used_cores = ','.join(map(str, used_cores))

print(used_cores)