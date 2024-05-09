#To run this, run these lines:
#salloc --nodes=1 -C cpu -q interactive -t 00:30:00 --exclusive --image=adammwea/netpyneshifter:v3
#salloc --nodes=1 -C cpu -q interactive -t 04:00:00 --exclusive --image=kpkaur28/neuron:v3
#bash test_shifter_interactive.sh
#SBATCH 
    
### parameters
Duration_Seconds=5
Batch_Run_Label=$SLURM_JOB_NAME
JOB_ID=$SLURM_JOB_ID
cores_per_node=128 # 128 physical cores on perlmutter node
nodes=$SLURM_NNODES
np=$((nodes*cores_per_node)) # 128 physical cores on the laptop

### Uncomment for testing MPI
# Start the Docker container and get its ID
#shifter mpiexec -np ${np} python3 testmpi.py #test mpi
#srun --mpi=pmi2 -n 4 -c 1 shifter --image=adammwea/netpyneshifter:1.0 python3 testmpi.py

# Run commands inside the Docker container
echo "Starting the Docker container"
container_id=$(docker run -it -d -v $(pwd):/app test_image /bin/bash)
container_run_path=$(docker exec $container_id bash -c "python3 NERSC/USER_init_new_batch.py ${Batch_Run_Label}")
echo "Container path: ${container_run_path}"
full_sh_path_container=$(docker exec $container_id bash -c "realpath $0")
echo "Batch script path in container: ${full_sh_path_container}"
script_name=$(basename $0) #get the name of this batch script file
docker exec $container_id bash -c "cp ${full_sh_path_container} ${container_run_path}/${script_name}" #save copy of this batch_script in the run_path
echo "Running batch script inside the Docker container..."
docker exec $container_id bash -c "\
    cd NERSC &&\
    mpiexec -np ${np}\
    nrniv -mpi batchRun.py -rp ${container_run_path} -d ${Duration_Seconds}\
    > ${container_run_path}/mpi_output.txt 2> ${container_run_path}/mpi_error.txt"
echo "Batch script finished running inside the Docker container"
echo "Stopping the Docker container"
docker stop $container_id # Stop the Docker container when you're done