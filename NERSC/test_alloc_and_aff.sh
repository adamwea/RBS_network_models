#!/bin/bash
#In Slurm each hyper thread is considered a "cpu" so the --cpus-per-task option must be 
#adjusted accordingly. Generally best performance is obtained with 1 OpenMP thread per 
#physical core. Additional details about affinity settings.

# Load necessary modules
module unload openmpi
module load python/3.9
module load conda
module load openmpi

module load PrgEnv-gnu
#module swap gcc gcc/11.2.0
ftn -fopenmp -o hybrid-hello hybrid-hello.f90
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
export OMP_PLACES=threads
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=spread

# export OMP_DISPLAY_AFFINITY=true
# export OMP_AFFINITY_FORMAT="host=%H, pid=%P, thread_num=%n, thread affinity=%A"
export KMP_AFFINITY=verbose
export CRAY_OMP_CHECK_AFFINITY=TRUE

# Execute the application and sort the output
# in the case of netpyne sims, tasks_per_node is apparently best set to cores_per_node
# (according to the hpcslurm jobscript in the netpyne repo)
cores_per_node=128
tasks_per_node=$((cores_per_node))
tasks_per_node=4
#cpu_per_task=$((2*(128/tasks_per_node)))
#srun -n $((tasks_per_node)) -c $((cpu_per_task)) check-hybrid.gnu.pm | sort -k4,6 #> output.log 2>&1
#srun -n $((tasks_per_node)) -c 16 check-hybrid.gnu.pm | sort -k4,6 #> output.log 2>&1
#srun -n $((tasks_per_node)) check-hybrid.gnu.pm ./a.out : -n $((tasks_per_node)) check-hybrid.gnu.pm ./b.out
#mpirun -n $((tasks_per_node)) check-hybrid.gnu.pm | sort -k4,6 #> output.log 2>&1
#srun -n $((tasks_per_node)) -c 16 ./hybrid-hello | sort -k2,3