#To run this, run these lines:
#salloc --nodes=1 -C cpu -q interactive -t 00:30:00 --exclusive --image=adammwea/netpyneshifter:v5
#salloc --nodes=1 -C cpu -q interactive -t 04:00:00 --exclusive --image=kpkaur28/neuron:v3
#bash test_shifter_interactive.sh
    
### parameters
Duration_Seconds=5
Batch_Run_Label=$SLURM_JOB_NAME
JOB_ID=$SLURM_JOB_ID
cores_per_node=4 # 128 physical cores on perlmutter node
nodes=$SLURM_NNODES
#num_MPI_task=$((nodes*cores_per_node)) # 128 physical cores on the laptop
num_MPI_task=$((cores_per_node)) # 128 physical cores on the laptop
proc_per_task=1

##modules
module load mpich/4.1.1

### Uncomment for testing MPI
# Start the Docker container and get its ID
#shifter mpiexec -np ${np} python3 testmpi.py #test mpi
#srun -N ${nodes} -n ${num_MPI_task} -c ${proc_per_task} shifter --image=adammwea/netpyneshifter:v5 python3 testmpi.py --display-allocation
#srun --mpi=pmi2 -n 4 -c 1 shifter --image=adammwea/netpyneshifter:v5 python3 testmpi.py

# Run commands inside the Docker container
echo "Starting shifter..."
container_run_path=$(shifter --image=adammwea/netpyneshifter:v5 python3 NERSC/USER_init_new_batch.py ${Batch_Run_Label})
# Remove the first character (the zero) from the variable
container_run_path=${container_run_path:1}
echo "Container path:" $container_run_path
#full_sh_path_container=$(shifter realpath --relative-to=$container_run_path $0)
full_sh_path_container=$(shifter realpath $0)
echo "Batch script path in container: ${full_sh_path_container}"
script_name=$(basename $0) #get the name of this batch script file
cp ${full_sh_path_container} ${container_run_path}/${script_name} #save copy of this batch_script in the run_path
echo "Running batch script inside the Docker container..."

# ## Run the batch script inside the Docker container
#shifter /bin/bash
cd NERSC
srun shifter --image=adammwea/netpyneshifter:v5 \
    nrniv -mpi batchRun.py -rp ${container_run_path} -d ${Duration_Seconds} #\
#srun -N ${SLURM_NNODES} shifter --image=adammwea/netpyneshifter:v5 nrniv -mpi -python batchRun.py -rp ${container_run_path} -d ${Duration_Seconds} #\
#     > ${container_run_path}/mpi_output.txt \
#     2> ${container_run_path}/mpi_error.txt
# echo "Batch script finished running inside the Docker container"
# echo "Stopping the Docker container"