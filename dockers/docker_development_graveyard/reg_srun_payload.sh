#!/bin/bash
#SBATCH --job-name=big_run
#SBATCH -A m2043
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH --exclusive
#SBATCH --output=./NERSC/output/latest_job_init_error.txt
#SBATCH --error=./NERSC/output/latest_job_init_output.txt
#SBATCH --image=adammwea/netpyneshifter:v5

###options
Duration_Seconds=15
#Batch_Run_Label='payload_test_at_scale'
Batch_Run_Label=$SLURM_JOB_NAME
nodes=$SLURM_NNODES #hpc
#nodes=2 #laptop or server
OMP_threads_per_process=1 #recomended 1 cpu per task NERSC, https://docs.nersc.gov/development/languages/python/parallel-python/#numpy-and-nested-threading
cpus_per_task=$OMP_threads_per_process
cores_per_node=128 # 128 physical. 256 logical. perlmutter
#cores_per_node=4 # 8 physical. 16 logical. laptop
#cores_per_node=48 # 24 physical. 48 logical. server
total_cores=$((nodes*cores_per_node))
#hyper_threads=1 #threads per core, max 2
ntasks=16 #need to find a nuumber of tasks that works well. Ideally some multiple of pop.
cores_per_task=$((total_cores/ntasks))
cpus_per_task=$((cores_per_task*2))

###check
#make sure ntasks is a multiple of nodes
if [ $((total_cores%ntasks)) -ne 0 ]; then
    echo "cores_per_node must be a multiple of tasks_per_node"
    exit 1
fi

###prep batchRun
container_run_path=$(python3 NERSC/USER_init_new_batch.py ${Batch_Run_Label})
echo "Container path:" $container_run_path
full_sh_path=$(realpath $0)
echo "Batch script path: ${full_sh_path}"
script_name=$(basename $0) #get the name of this batch script file
full_sh_path_container=${full_sh_path} # set full_sh_path_container to the value of full_sh_path
cp ${full_sh_path_container} ${container_run_path}/${script_name} #save copy of this batch_script in the run_path
echo "Running a shifter for each simulation..."

### srun the batch script ###
#module load mpich/4.1.1
echo "Nodes: ${nodes}"
echo "MPI tasks per node: ${ntasks}"
#echo "Cores per task: ${cores_per_task}"
echo "CPUs per task: ${cpus_per_task}"
echo "OMP threads per task: ${OMP_threads_per_process}"
echo "Running..."

#Test batchRun local
module load conda
conda activate preshifter
export OMP_NUM_THREADS=1
cd NERSC
module load parallel
rm -f commands.txt
max_gens=3000
gen_pop=200 #one job and one pop per socket
#njobs=$((SLURM_NNODES)) #until I figure out how to specifically ask for sockets, 1 job per node
njobs=$((SLURM_NNODES*2)) #--sockets-per-node=1 should allow two parallel jobs per node, good use of memory
gen_count=0
python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label} & export PID=$!
#init
echo '' > commands.txt
bash wait_for_commands.sh $gen_pop
echo "Commands added to commands.txt"

while (($gen_count < $max_gens)); do
    # Run Init commands
    # echo '' > commands.txt
    # bash wait_for_commands.sh $gen_pop
    #echo "Commands added to commands.txt"

    # Run commands in parallel and clear commands.txt
    echo "Running commands in parallel"
    parallel --jobs ${njobs} echo :::: commands.txt
    parallel --jobs ${njobs} :::: commands.txt
    echo '' > commands.txt
    
    # re-init
    bash wait_for_commands.sh $gen_pop
    echo "Parallel commands finished"
    echo "Commands added to commands.txt"
    gen_count=$((gen_count+1))
    
    # # Optional: add a delay to avoid rapid looping
    # echo "Sleeping for 1 second"
    # sleep 1
    # #break
done