#!/bin/bash
#In Slurm each hyper thread is considered a "cpu" so the --cpus-per-task option must be 
#adjusted accordingly. Generally best performance is obtained with 1 OpenMP thread per 
#physical core. Additional details about affinity settings.

# Load necessary modules
module unload openmpi
module load python/3.9
module load conda
module load openmpi

# Activate the conda environment
source activate 2DSims

# '''
# Clarifying Affinity and Execution:
# Affinity Settings: When you specify affinity settings, you're essentially guiding the 
# operating system on how to distribute the threads across the available CPU cores. This is 
# especially important in systems with many cores to optimize performance by minimizing memory 
# access latency and maximizing cache use.
#
# Actual Thread Execution: Despite the potential to be scheduled on many different cores 
# (as defined by the affinity settings), each individual thread will execute on only one 
# core at a time. The operating system's scheduler handles the actual placement of threads 
# based on these settings.
# '''
# Set OpenMP environment variables
export OMP_PROC_BIND=spread
# export OMP_PLACES=threads
# export OMP_NUM_THREADS=64
# export OMP_DISPLAY_AFFINITY=true
# export OMP_AFFINITY_FORMAT="host=%H, pid=%P, thread_num=%n, thread affinity=%A"
# export KMP_AFFINITY=verbose,none

# Execute the application and sort the output
# in the case of netpyne sims, tasks_per_node is apparently best set to cores_per_node
# (according to the hpcslurm jobscript in the netpyne repo)
cores_per_node=128
tasks_per_node=$((cores_per_node))
#cpu_per_task=$((2*(128/tasks_per_node)))
#srun -n $((tasks_per_node)) -c $((cpu_per_task)) check-hybrid.gnu.pm | sort -k4,6 #> output.log 2>&1
srun -n $((tasks_per_node)) check-hybrid.gnu.pm | sort -k4,6 #> output.log 2>&1
#mpirun -n $((tasks_per_node)) check-hybrid.gnu.pm | sort -k4,6 #> output.log 2>&1