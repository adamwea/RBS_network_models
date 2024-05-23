###options
Duration_Seconds=15
Batch_Run_Label='payload_test'
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
#rm -rf /app/tmp
#export TMPDIR=/app/tmp
#export PMIX_MCA_gds=hash
#python3 batchRun_srun.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}
module load parallel
# njobs=$SLURM_NNODES
# echo ${njobs}
#delete input.txt if it exists
rm -f commands.txt
#parallel --citation
#parallel --jobs ${njobs} ./payload.sh argument_{} ::: input.tx

#parallel --jobs ${njobs} python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}
#python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}

#make sure batch script is only running on one core
echo "srun -N 1 -n 1 -c 1 python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}" > commands.txt
#python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}
njobs=$((SLURM_NNODES+1))
#njobs=$((4+1))
echo ${njobs}
#srun -N 1 -n 1 -c 1 python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}

#test 1: /usr/bin/bash: argument_srun: command not found
#parallel --jobs ${njobs} argument_{} :::: commands.txt

#test2:
# - batchRun_payload.py runs. Not sure if parallel sruns are working.
# - I'm giving it 12 minutes to show evidence of running in parallel.
# - doesnt seem to start after running batchRun_payload.py. Probably because I didnt srun so it only does 1 job at a time.
#parallel --jobs ${njobs} < commands.txt

#test3:
# - now not even the first command is running.
# - setting 12 minute time limit.
# - 
srun parallel --jobs ${njobs} < commands.txt





# srun \
#     --ntasks ${ntasks} \
#     --cpus-per-task ${cores_per_task} \
#     --cpu_bind=cores \
#     shifter --image=adammwea/netpyneshifter:v6 \
#     nrniv -mpi -python batchRun_mpi.py \
#     -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}

# shifter /bin/bash -c "\
#     mpiexec -n ${total_cores} &&\
#     nrniv -mpi -python batchRun_mpi.py \
#     -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}"

