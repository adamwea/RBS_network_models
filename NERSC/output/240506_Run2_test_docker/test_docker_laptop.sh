#specify the duration of the simulation and the label of the batch run
Duration_Seconds=7
Batch_Run_Label=test_docker

#initialize
run_path=$(python3 NERSC/USER_init_new_batch.py ${Batch_Run_Label}) # Initialize the batch file and store the return value in run_path
echo "Run path: ${run_path}"

#save copy of this batch_script in the run_path
full_path=$(realpath $0)
cp ${full_path} ${run_path}/test_docker_laptop.sh

# Run the MPI command
cd NERSC # Change directory
echo "Running the simulation"
docker run -it --rm -v $(pwd) test_image \
mpiexec -np 1 \
nrniv -mpi batchRun.py ${run_path} 5 \
> ${run_path}/mpi_output.txt \
2> ${run_path}/mpi_error.txt