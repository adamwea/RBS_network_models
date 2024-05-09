source activate 2DSims_labserver
export PYTHONWARNINGS="ignore:DeprecationWarning"

#specify the duration of the simulation and the label of the batch run
Duration_Seconds=10
Batch_Run_Label=labserver_debug
run_path=$(python3 NERSC/USER_init_new_batch.py ${Batch_Run_Label}) # Initialize the batch file and store the return value in run_path
echo "Run path: ${run_path}"
#save copy of this batch_script in the run_path
full_path=$(realpath $0)
cp ${full_path} ${run_path}/run_in_labserver.sh

#Max 48 if absolutely no one else is using server
NODES=1
PPN=10 
NP=$((NODES * PPN)) #Max 48 if absolutely no one else is using server

# Check versions
nrniv --version
mpiexec --version
mpicc --version

# Run the MPI command
cd NERSC # Change directory
echo "Running the simulation"
mpiexec --display-map --map-by ppr:${PPN}:node -np ${NP} \
nrniv -mpi batchRun.py ${run_path} ${Duration_Seconds} \
> ${run_path}/mpi_output.txt \
2> ${run_path}/mpi_error.txt