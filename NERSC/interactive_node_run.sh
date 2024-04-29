module load conda
conda activate neuron_env
#conda activate 2DSims
module load openmpi
#module load openmpi/5.0.0rc12

mkdir -p NERSC/output/job_outputs

# Set the library path for NEURON and Python libraries
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set the path to include the NEURON binaries
export PATH=$HOME/neuron/bin:$PATH

# Execute the application and capture all output
# mpiexec_command = f"mpiexec {mpiexec_flags} --use-hwthread-cpus -np {USER_total_cores} nrniv -mpi NERSC/batchRun.py {USER_JobName} {USER_seconds}"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
cd NERSC
#mpiexec --use-hwthread-cpus -np 128 nrniv -mpi -bind-to hwthread batchRun.py interactive_debug 5
#mpiexec --display-map -np 128 nrniv -mpi batchRun.py interactive_debug 5
# cores_per_node = 128
mpiexec --map-by ppr:128:node --display-map -np 256 nrniv -mpi batchRun.py interactive_debug 5
#mpiexec --display-map -np 256 nrniv -mpi batchRun.py interactive_debug 5
#mpiexec --report-bindings -np 64 nrniv -mpi batchRun.py interactive_debug 5
#mpiexec --use-hwthread-cpus --display-map -mca mpi_abort_print_stack 1 -np 256 nrniv -mpi -bind-to hwthread debug.py interactive_debug 5