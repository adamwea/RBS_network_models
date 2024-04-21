# generate_sbatch.py
import os
import subprocess
from USER_INPUTS import *
import datetime

datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define the sbatch options using the variables from user_inputs.py
sbatch_options = f"""#!/bin/bash
#SBATCH --job-name={USER_JobName}
#SBATCH -A {USER_allocation}
#SBATCH -t {USER_walltime}
#SBATCH --nodes={USER_nodes}
#SBATCH --output=NERSC/output/job_outputs/job_output_%j_{USER_JobName}.txt
#SBATCH --error=NERSC/output/job_outputs/job_error_%j_{USER_JobName}.txt
#SBATCH --mail-user={USER_email}
#SBATCH --mail-type=ALL
#SBATCH -q {USER_queue}
#SBATCH -C cpu
"""

# Define the rest of the shell script
shell_script = f"""{sbatch_options}
export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose
export FI_LOG_LEVEL=debug

module load python
module load conda
module unload cray-mpich
module unload cray-libsci
module use /global/common/software/m3169/perlmutter/modulefiles
module load openmpi

conda activate 2DSims

mkdir -p NERSC/output/job_outputs

python3 NERSC/batchRun.py {USER_JobName} 30
"""

# Write the shell script to a file
job_inputs_dir = "NERSC/job_inputs"
if not os.path.exists(job_inputs_dir):
    os.makedirs(job_inputs_dir)
with open(f"NERSC/job_inputs/{datetime_str}_sbatch_jobscript.sh", "w") as f:
    f.write(shell_script)

# Submit the shell script using sbatch
#subprocess.run(["sbatch", "sbatch_script.sh"])