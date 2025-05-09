#!/bin/bash
#SBATCH -A m2043                            
#SBATCH -q regular                          # regular or shared queue
#SBATCH -C cpu                              # cpu or gpu nodes
#SBATCH -t 12:00:00                         # walltime
#SBATCH --nodes=8                          # number of nodes
#SBATCH --ntasks-per-node=256               # max tasks per node
#SBATCH --cpus-per-task=1                   # logical CPUs per task
#SBATCH --image=adammwea/netsims_docker:v1
#SBATCH --threads-per-core=2                
#SBATCH --hint=socket
#SBATCH --mail-type=ALL

# Optional: For debugging
# echo "SLURM_NTASKS: $SLURM_NTASKS"
# echo "SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"
# echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

# Bind CPUs properly
export SLURM_CPU_BIND="cores"

# Launch your application
bash /global/homes/a/adammwea/workspace/repos/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_WT/config_and_run.sh

# to run the script, use:
# sbatch /global/homes/a/adammwea/workspace/repos/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_WT/queue_job.sh