# generate_sbatch.py
import os
import subprocess
from USER_INPUTS import *
import datetime

# get sys arguments
#option = sys.argv[1]
options = ['mpidirect', 'mpi_bulletin'] #, 'option must be either "mpidirect" or "mpibulletin"'
option = 'mpi_bulletin'

if option == 'mpi_bulletin':
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    

    #prepare mpiexec command
    # mtl_base_verbose = "--mca mtl_base_verbose 100"
    # report_bindings = "--report-bindings"
    # display_map = "--display-map"
    # display_topo = "--display-topo"
    # display_devel_map = "--display-devel-map"
    # hw_threads = '--use-hwthread-cpus'
    # mpiexec_flags = f"{mtl_base_verbose} {report_bindings} {display_map} {display_topo} {display_devel_map}"
    mpiexec_command = [
        f"mpiexec --map-by ppr:{Perlmutter_cores_per_node}:node"
        f" -np --display-map {USER_total_cores}"
        f" nrniv -mpi batchRun.py {USER_JobName} {USER_seconds}"
        ]
    #remove /n from mpiexec_command
    mpiexec_command = ' '.join(mpiexec_command)
    # mpiexec --map-by ppr:128:node --display-map -np 256 nrniv -mpi batchRun.py interactive_debug 5

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
#SBATCH --exclusive
    """

    # Define the rest of the shell script
    #To run use: mpiexec -np [num_cores] nrniv -mpi batchRun.py
    #    export OMP_PROC_BIND=spread
    # export KMP_AFFINITY=verbose
    # export FI_LOG_LEVEL=debug
    shell_script = f"""{sbatch_options}
module load conda
conda activate neuron_env
module load openmpi
mkdir -p NERSC/output/job_outputs

# Set the library path for NEURON and Python libraries
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set the path to include the NEURON binaries
export PATH=$HOME/neuron/bin:$PATH

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
cd NERSC
{mpiexec_command}
    """

    # Write the shell script to a file
    job_inputs_dir = "NERSC/job_inputs"
    if not os.path.exists(job_inputs_dir):
        os.makedirs(job_inputs_dir)
    with open(f"NERSC/job_inputs/{datetime_str}_sbatch_jobscript_{USER_JobName}.sh", "w") as f:
        f.write(shell_script)

    # Submit the shell script using sbatch
    #subprocess.Popen(["sbatch", f'NERSC/job_inputs/{datetime_str}_sbatch_jobscript_{USER_JobName}.sh'])
elif option == 'mpidirect':
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
    #SBATCH --exclusive
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

    python3 NERSC/batchRun.py {USER_JobName} {USER_seconds}
    """

    # Write the shell script to a file
    job_inputs_dir = "NERSC/job_inputs"
    if not os.path.exists(job_inputs_dir):
        os.makedirs(job_inputs_dir)
    with open(f"NERSC/job_inputs/{datetime_str}_sbatch_jobscript_{USER_JobName}.sh", "w") as f:
        f.write(shell_script)

    # Submit the shell script using sbatch
    subprocess.Popen(["sbatch", f'NERSC/job_inputs/{datetime_str}_sbatch_jobscript_{USER_JobName}.sh',f' 2>&1 | tee NERSC/output/job_outputs/{datetime_str}_bashoutput_{USER_JobName}.txt'])