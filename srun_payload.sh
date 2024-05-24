#### SRUN PAYLOAD METHOD AT NERSC - HIGHLY PARALLELIZED AND SCALABLE
# Run this script with the following commands (modulate nodes and time as needed):
# salloc --nodes=2 -C cpu -q interactive -t 04:00:00 --exclusive --image=adammwea/netpyneshifter:v5
# bash srun_payload.sh

###OPTIONS
Duration_Seconds=300 #Simulation Duration in seconds
Batch_Run_Label='Figure_Sims' #Label for the batch run folder
nodes=$SLURM_NNODES #hpc, number of nodes as requested by salloc
#nodes=1 #override as needed. If running on laptop or server, number of nodes is 1. #NOTE: if running on a laptop or server, you will probably need to run in docker container with srun and slurm installed.
max_gens=300 #number of generations to run, generally a high number to allow for convergence
max_gens=1 #override as needed
njobs=$((nodes*2)) #generally 2 jobs per node, 1 per socket
njobs=2 #override as needed
gen_pop=2 #population per generation

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

## test 5 - this solution works!:
# - trying loop method
# - putting $ in python command makes it run in the background...sick

##init run
echo "Running a simulations..."
cd NERSC
echo '' > commands.txt #clear commands.txt
python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label} & export PID=$! #run batchRun_payload.py in the background
bash wait_for_commands.sh $gen_pop #wait for the first command to finish sending the next srun commands to commands.txt
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

#test 4:
# - trying ::::
# - takes a few minutes, but this runs the first python command at least.
# - yea it get's stuck after the first command. probably waiting for the first command to finish....which it never will.
#echo "python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}" > commands.txt
#parallel --jobs ${njobs} echo :::: commands.txt
#parallel --jobs ${njobs} :::: commands.txt

#test3:
# - now not even the first command is running.
# - setting 12 minute time limit.
# - Nope this doesnt work. Nothing runs.
#echo "srun -N 1 -n 1 -c 1 python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}" > commands.txt
#srun parallel --jobs ${njobs} < commands.txt

#test2:
# - batchRun_payload.py runs. Not sure if parallel sruns are working.
# - I'm giving it 12 minutes to show evidence of running in parallel.
# - doesnt seem to start after running batchRun_payload.py. Probably because I didnt srun so it only does 1 job at a time.
#echo "srun -N 1 -n 1 -c 1 python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}" > commands.txt
#parallel --jobs ${njobs} < commands.txt

#test 1: /usr/bin/bash: argument_srun: command not found
#echo "srun -N 1 -n 1 -c 1 python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}" > commands.txt
#parallel --jobs ${njobs} argument_{} :::: commands.txt







