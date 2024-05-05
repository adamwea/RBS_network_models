source activate 2DSims_labserver
export PYTHONWARNINGS="ignore:DeprecationWarning"

#Max 48 if absolutely no one else is using server
NODES=1
PPN=10 
NP=$((NODES * PPN)) #Max 48 if absolutely no one else is using server

# Change directory
cd NERSC

# Check versions
nrniv --version
mpiexec --version
mpicc --version
echo "Running the simulation"

# Run the MPI command
#mpiexec --display-map --map-by ppr:1:core -np ${NP} nrniv -mpi batchRun.py interactive_node_run 10
echo "mpiexec --display-map --map-by ppr:${PPN}:node -np ${NP} nrniv -mpi batchRun.py interactive_node_run 10"
mpiexec --display-map --map-by ppr:${PPN}:node -np ${NP} nrniv -mpi batchRun.py labserver_run 10