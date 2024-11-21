#!/bin/bash
#SBATCH --job-name=24n24hr
#SBATCH -A m2043
#SBATCH -t 24:00:00
#SBATCH -N 24
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --exclusive
#SBATCH --output=./NERSC/output/latest_job_init_error.txt
#SBATCH --error=./NERSC/output/latest_job_init_output.txt
#SBATCH --image=adammwea/netpyneshifter:v5

#### SRUN PAYLOAD METHOD USING GNU PARALLEL MODULE AT NERSC - HIGHLY PARALLELIZED AND SCALABLE
# Run this script with the following commands (modulate nodes and time as needed):
# Q0S JOB: 
#   sbatch slurm_payload.sh
# INTERACTIVE NODE:
#   salloc --nodes=2 -C cpu -q interactive -t 04:00:00 --exclusive --image=adammwea/netpyneshifter:v5
# bash /pscratch/sd/a/adammwea/RBS_network_simulations/sbatch_scripts/slurm_payload.sh

### OPTIONS
# Batch_Run_Label=${SLURM_JOB_NAME:-"login"} # Label for the batch run. Default to 'login' if empty.
# Duration_Seconds=1 # Simulation Duration in seconds
nodes=${SLURM_NNODES:-1} # Number of nodes, default to 1 if not set
# max_gens=10 # Number of generations to run
# gen_pop=10 # Population per generation

### PARALLELIZATION PARAMS
OMP_threads_per_process=1 # Recommended 1 CPU per process
export OMP_NUM_THREADS=$OMP_threads_per_process

## Load modules and activate conda environment
echo "Loading modules and setting environment variables..."
module load conda
conda activate preshifter
module load parallel

### RUN
# ## Copy this bash script to the run_path
# full_sh_path=$(realpath $0)
# script_name=$(basename $0)
# cp ${full_sh_path} ${run_path}/${script_name}
# echo "Bash script copied to run_path"

## Report allocation specs
echo "SLURM Environment"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Physical cores per node: $(lscpu | grep '^Core(s) per socket:' | awk '{print $4}')"
echo "Logical CPUs per node: $(lscpu | grep '^CPU(s):' | awk '{print $2}')"
echo "SLURM CPUs per node: $SLURM_CPUS_ON_NODE"
echo "Sockets per node: $(lscpu | grep 'Socket(s):' | awk '{print $2}')"
echo "Cores per socket: $(lscpu | grep 'Core(s) per socket:' | awk '{print $4}')"

## Report processes allocation based on environment
echo "Allocating 1 simulation per socket for optimal memory usage (i.e. 2 per node)..."
njobs=$((nodes * 2)) # Generally 2 jobs per node, 1 per socket
cores_per_node=$(lscpu | grep '^Core(s) per socket:' | awk '{print $4}')
ntasks=$(echo "$cores_per_node * 0.75" | bc)
ntasks=${ntasks%.*} # Remove the decimal part, if any
echo "Allocating $ntasks parallel cell populations per simulation in each socket..."
echo "Initializing $gen_pop simulation(s) cfgs per generation, $Duration_Seconds seconds each..."

## Initialize run
echo "Initializing parallelized simulations commands..."
cd /pscratch/sd/a/adammwea/RBS_network_simulations/
echo '' > commands.txt # Clear commands.txt
(
python -u ./simulate/batchRun_evolutionary_algorithm.py \
    & export PID=$! # Run batchRun_payload.py in the background
) >> "${Batch_Run_Label}_output.run" 2>"${Batch_Run_Label}_error.err"
gen_count=$(ls -l ${run_path} | grep -c -P '^d.*gen_\d+$') # Check for folders named gen_# in run_path, get the number of the latest generation
current_gen=$((gen_count - 1)) # Latest generation is the current generation (if negative, set to 0 inside wait_for_current_gen.sh)
current_gen=$((current_gen < 0 ? 0 : current_gen)) # If current_gen is -1 then it will be set to 0 in wait_for_current_gen.sh
echo "Waiting for collection of commands for generation: $current_gen"
bash ./sbatch_scripts/wait_for_commands.sh $gen_pop &>> "${Batch_Run_Label}_wait_output.run" # Wait for batchRun_*.py to finish sending the next srun commands to commands.txt
echo "Commands collected in commands.txt"

## Loop through generations
gen_count=0
max_gens=10
while ((gen_count < max_gens)); do
    echo "Delivering Payload (${njobs} in parallel)..."
    parallel --jobs ${njobs} echo :::: commands.txt
    echo "Running parallel simulations..."
    parallel --jobs ${njobs} :::: commands.txt
    echo '' > commands.txt # Clear commands.txt
    echo "Parallel commands finished"
    bash ./sbatch_scripts/wait_for_commands.sh $gen_pop &>> "${Batch_Run_Label}_wait_output.run" # Wait for batchRun_*.py to finish sending the next srun commands to commands.txt
    echo "Commands added to commands.txt"
    gen_count=$((gen_count + 1))
done