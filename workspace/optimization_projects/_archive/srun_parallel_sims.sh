#!/bin/bash
#SBATCH --job-name=debug_job
#SBATCH -A m2043
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --exclusive
#SBATCH --output=./NERSC/output/latest_job_init_error.txt
#SBATCH --error=./NERSC/output/latest_job_init_output.txt
#SBATCH --image=adammwea/netpyneshifter:v5

## Load modules and activate conda environment
load_modules() {
    echo "Loading modules and setting environment variables..."
    module load conda
    conda activate preshifter
    module load parallel
    nodes=${SLURM_NNODES:-1} # Number of nodes, default to 1 if not set
    OMP_threads_per_process=1 # Recommended 1 CPU per process
    export OMP_NUM_THREADS=$OMP_threads_per_process

}

## Report allocation specs
report_allocation_specs() {
    echo "SLURM Environment"
    echo "Nodes: $SLURM_JOB_NUM_NODES"
    echo "Physical cores per node: $(lscpu | grep '^Core(s) per socket:' | awk '{print $4}')"
    echo "Logical CPUs per node: $(lscpu | grep '^CPU(s):' | awk '{print $2}')"
    echo "SLURM CPUs per node: $SLURM_CPUS_ON_NODE"
    echo "Sockets per node: $(lscpu | grep 'Socket(s):' | awk '{print $2}')"
    echo "Cores per socket: $(lscpu | grep 'Core(s) per socket:' | awk '{print $4}')"
}

## Report processes allocation based on environment
report_processes_allocation() {
    echo "Allocating 1 simulation per socket for optimal memory usage (i.e. 2 per node)..."
    njobs=$((nodes * 2)) # Generally 2 jobs per node, 1 per socket
    cores_per_node=$(lscpu | grep '^Core(s) per socket:' | awk '{print $4}')
    ntasks=$(echo "$cores_per_node * 0.75" | bc)
    ntasks=${ntasks%.*} # Remove the decimal part, if any
    echo "Allocating $ntasks parallel cell populations per simulation in each socket..."
    echo "Initializing $gen_pop simulation(s) cfgs per generation, $Duration_Seconds seconds each..."
}

## Source USER_ variables from temp_user_args.py
source_USER_vars() {
    # Load Python USER_ variables from temp_user_args.py
    if [ -f temp_user_args.py ]; then
        echo "Sourcing variables from temp_user_args.py..."
        while IFS= read -r line; do
            if [[ $line == USER_* ]]; then
                eval "export $line"
            fi
        done < <(python -c "exec(open('temp_user_args.py').read()); \
            [print(f'{key}={value!r}') for key, value in locals().items() if key.startswith('USER_')]")
    else
        echo "Error: temp_user_args.py not found."
        exit 1
    fi

    # Verify that variables are set
    env | grep USER_
}

## Wait for commands to be collected in commands.txt
wait_for_commands() {
    gen_pop=$1
    count=0
    timeout=15
    start_time=$(date +%s)

    while true; do 
        inotifywait --event modify --timeout $timeout "commands.txt" && {
            count=$(wc -l < "commands.txt")
            echo "line added to commands.txt"
            if (( count >= gen_pop )); then
                break
            fi
        } || {
            current_time=$(date +%s)
            if (( current_time - start_time >= timeout )); then
                echo "Timeout reached without modification"
                count=$(wc -l < "commands.txt")
                echo "Current count: $count"
                if (( count >= gen_pop )); then
                    break
                fi
            fi
        }
    done
}

## Initialize run
initialize_run() {
    # args
    batch_script=$1
    
    # change to the directory where the simulation scripts are located
    cd /pscratch/sd/a/adammwea/RBS_network_simulations/

    # Initialize parallelized simulations commands
    echo "Initializing parallelized simulations commands..."
    echo '' > commands.txt # Clear commands.txt

    # Initialize temporary output files
    echo '' > "temp_output.run"
    echo '' > "temp_error.err"

    # delete temp_user_args.py
    rm -f temp_user_args.py

    # Run batchRun_evolutionary_algorithm.py in the background
    echo "Running batch_script (${batch_script}) in the background..."
    (
    python -u $batch_script\
        & export PID=$! # Run batchRun_payload.py in the background
    ) >> "temp_output.run" 2>"temp_error.err"

    #wait forr temp_user_args.py to be created - this step is mostly here to make sure this function does
    #not progress before temp_user_args.py is created and USER_vars are set for the current run
    echo "Waiting for temp_user_args.py to be created..." 
    while [ ! -f temp_user_args.py ]; do
        sleep 1
    done
    
    # Verify that needed USER_vars have been set by batchRun_*.py
    echo "Sourcing USER_ variables..."
    source_USER_vars

    # Assert that USER_run_path is set
    echo "Checking for USER_run_path..."
    if [ -z "${USER_run_path}" ]; then
        echo "Error: USER_run_path is not set."
        exit 1
    fi
    run_path=${USER_run_path}
    echo "USER_run_path: $run_path"
    
    # Check for folders named gen_# in run_path, get the number of the latest generation
    echo "Checking for folders named gen_# in run_path..."
    gen_count=$(ls -l ${run_path} | grep -c -P '^d.*gen_\d+$') # Check for folders named gen_# in run_path, get the number of the latest generation
    current_gen=$((gen_count - 1)) # Latest generation is the current generation (if negative, set to 0 inside wait_for_current_gen.sh)
    current_gen=$((current_gen < 0 ? 0 : current_gen)) # If current_gen is -1 then it will be set to 0 in wait_for_current_gen.sh
    echo "Current generation: $current_gen"

    # Wait for collection of commands for generation generated by batchRun_*.py in the background
    echo "Waiting for collection of commands for generation: $current_gen"
    gen_pop=${USER_pop_size:-10} # Population size for each generation
    echo "gen_pop: $gen_pop"
    wait_for_commands $gen_pop &>> "temp_wait_output.run" # Wait for batchRun_*.py to finish sending the next srun commands to commands.txt
    #bash ./sbatch_scripts/wait_for_commands.sh $gen_pop &>> "wait_output.run" # Wait for batchRun_*.py to finish sending the next srun commands to commands.txt
    echo "Commands collected in commands.txt"
}

## Loop through generations
loop_through_generations() {
    gen_count=0
    max_gens=${USER_max_gens:-10} # Number of generations to run
    #njobs=2 #override njobs to 2 for now
    while ((gen_count < max_gens)); do
        
        # Report generation count
        echo "Delivering Payload (${njobs} in parallel)..."
        srun -N 1 parallel --jobs ${njobs} echo :::: commands.txt
        
        # Run parallel simulations
        echo "Running parallel simulations..."
        srun -N 1 parallel --jobs ${njobs} :::: commands.txt
        echo "Parallel commands finished"

        # Copy generation commands to USER_run_path
        echo "Copying generations to USER_run_path..."
        cp commands.txt ${USER_run_path}/commands_${gen_count}.txt # Save commands.txt for reference
        
        # Clear commands.txt
        echo '' > commands.txt # Clear commands.txt
        
        # Wait for collection of commands for next generation generated by batchRun_*.py in the background
        echo "Waiting for collection of commands for generation: $((gen_count + 1))"
        echo $gen_pop
        wait_for_commands $gen_pop &>> "temp_wait_output.run" # Wait for batchRun_*.py to finish sending the next srun commands to commands.txt
        #bash ./sbatch_scripts/wait_for_commands.sh $gen_pop &>> "temp_wait_output.run" # Wait for batchRun_*.py to finish sending the next srun commands to commands.txt
        echo "Commands added to commands.txt"

        # Increment generation count
        gen_count=$((gen_count + 1))
    done
}

## Copy temporary output files to USER_run_path
copy_move_and_cleanup_temp_files() {
    echo "Copying temporary output files to USER_run_path..."
    cp temp_output.run "${USER_run_path}/output.run"
    cp temp_error.err "${USER_run_path}/error.err"
    cp temp_wait_output.run "${USER_run_path}/wait_output.run"
    mv pids.pid "${USER_run_path}/pids.pid"
    rm -f temp_output.run temp_error.err temp_wait_output.run pids.pid temp_user_args.py commands.txt
}

## Run standard plotting scripts
run_standard_plotting() {
    echo "Running standard plotting scripts..."
    python -u ./plotting/plot_generation.py
    python -u ./plotting/plot_population.py
    python -u ./plotting/plot_simulation.py
}

## Main function to run all steps
main() {
    batch_script=$1
    cd /pscratch/sd/a/adammwea/RBS_network_simulations
    load_modules
    report_allocation_specs
    report_processes_allocation
    initialize_run $batch_script
    loop_through_generations
    copy_move_and_cleanup_temp_files
    #run_standard_plotting
}

#### SRUN PAYLOAD METHOD USING GNU PARALLEL MODULE AT NERSC - HIGHLY PARALLELIZED AND SCALABLE
# Run this script with the following commands (modulate nodes and time as needed):
# Q0S JOB: 
#   sbatch slurm_payload.sh
# INTERACTIVE NODE:
#   salloc --nodes=2 -C cpu -q interactive -t 04:00:00 --exclusive --image=adammwea/netpyneshifter:v5
# bash /pscratch/sd/a/adammwea/RBS_network_simulations/sbatch_scripts/slurm_payload.sh

# Execute main function
export batch_script='/pscratch/sd/a/adammwea/RBS_network_simulations/optimization_scripts/batchRun_evol.py'
main $batch_script