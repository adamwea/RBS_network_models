#!/bin/bash
#SBATCH --job-name=50nodes
#SBATCH -A m2043
#SBATCH -t 10:00:00
#SBATCH -N 50
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --exclusive
#SBATCH --output=./NERSC/output/latest_job_init_error.txt
#SBATCH --error=./NERSC/output/latest_job_init_output.txt
#SBATCH --image=adammwea/netpyneshifter:v5

#### SRUN PAYLOAD METHOD AT NERSC - HIGHLY PARALLELIZED AND SCALABLE
# Run this script with the following commands (modulate nodes and time as needed):
# sbatch srun_payload_qos.sh

###OPTIONS
Duration_Seconds=300 #Simulation Duration in seconds
Batch_Run_Label=$SLURM_JOB_NAME #Label for the batch run
nodes=$SLURM_NNODES #hpc, number of nodes as requested by salloc
#nodes=1 #override as needed. If running on laptop or server, number of nodes is 1. #NOTE: if running on a laptop or server, you will probably need to run in docker container with srun and slurm installed.
max_gens=3000 #number of generations to run, generally a high number to allow for convergence
#max_gens=1 #override as needed
njobs=$((nodes*2)) #generally 2 jobs per node, 1 per socket
#njobs=2 #override as needed
gen_pop=200 #population per generation

### PARALLELIZATION PARAMS
OMP_threads_per_process=1 #recomended 1 cpu per process NERSC, https://docs.nersc.gov/development/languages/python/parallel-python/#numpy-and-nested-threading
export OMP_NUM_THREADS=$OMP_threads_per_process

### RUN
##init batchRun folder
run_path=$(python3 NERSC/USER_init_new_batch.py ${Batch_Run_Label})
echo "batchRun folder path created:" $run_path

##copy this bashScript to the run_path
full_sh_path=$(realpath $0)
echo "bashScript Path: ${full_sh_path}"
script_name=$(basename $0) #get the name of this bashScript file
full_sh_path=${full_sh_path} 
cp ${full_sh_path} ${run_path}/${script_name} #save copy of this bashScript in the run_path
echo "bashScript copied to run_path"

##load modules and activate conda env
echo "loading modules and setting enviornment variables..."
module load conda
conda activate preshifter
module load parallel
#rm -rf /app/tmp
#export TMPDIR=/app/tmp
#export PMIX_MCA_gds=hash

##init run
echo "Running a simulations..."
cd NERSC
echo '' > commands.txt #clear commands.txt
python batchRun_payload.py -rp ${run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label} & export PID=$! #run batchRun_payload.py in the background
#bash wait_for_latest_gen.sh $run_path #wait for commands in commands.txt to pertain to the latest generation, enable continuation.
bash wait_for_commands.sh $gen_pop #wait for the batchRun_payload.py to finish sending the next srun commands to commands.txt
echo "Commands added to commands.txt"

##Loop through generations
gen_count=0
while (($gen_count < $max_gens)); do
    # Run commands in parallel and clear commands.txt
    echo "Running commands in parallel"
    parallel --jobs ${njobs} echo :::: commands.txt
    parallel --jobs ${njobs} :::: commands.txt
    echo '' > commands.txt
    echo "Parallel commands finished"    
    # re-init
    bash wait_for_commands.sh $gen_pop
    echo "Commands added to commands.txt"
    gen_count=$((gen_count+1))    
done







