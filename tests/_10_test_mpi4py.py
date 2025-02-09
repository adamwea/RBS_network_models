import os
import sys
from mpi4py import MPI
import logging
# ===================================================================================================
# Use interactive node to test
'''
salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=2 --image=adammwea/netsims_docker:v1
'''

# Build standard MPI environment recommended by NERSC (or load docker/shifter/conda where this is already setup)
'''
module load python
conda create -n my_mpi4py_env python=3.8
conda activate my_mpi4py_env
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
'''
#OR # NOTE: this is the simplified conda environment I've setup for myself
''' 
module load conda
conda activate netpyne_mpi 
'''

# How to test MPI allocation
'''
shifter --image=adammwea/netsims_docker:v1 /bin/bash
srun -n 4 python /pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/tests/_10_test_mpi_alloc.py
mpirun -n 4 python /pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/tests/_10_test_mpi_alloc.py
'''
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_mpi4py_installation():
    """Check if mpi4py is installed and accessible."""
    try:
        import mpi4py
        logging.info("mpi4py is installed and imported successfully.")
    except ImportError:
        logging.error("mpi4py is not installed. Please install it using 'pip install mpi4py'.")
        sys.exit(1)

def validate_mpi_environment():
    """Validate the basic MPI environment."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logging.info(f"MPI Environment Initialized: Rank {rank} of {size} processes.")

    if size < 2:
        logging.warning("MPI size is less than 2. Some tests may not be meaningful.")
    
    return comm, rank, size

def test_working_directory():
    """Test if the script can correctly set and retrieve the current working directory."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    cwd = os.getcwd()

    assert cwd == script_dir, f"Working directory mismatch: expected {script_dir}, got {cwd}"
    logging.info(f"Working directory validated: {cwd}")

def test_rank_consistency(comm, rank, size):
    """Test rank consistency across processes."""
    # Gather all ranks to the root process
    all_ranks = comm.gather(rank, root=0)
    if rank == 0:
        logging.info(f"Ranks gathered at root: {all_ranks}")
        assert sorted(all_ranks) == list(range(size)), "Rank inconsistency detected!"

def test_inter_process_communication(comm, rank, size):
    """Test basic inter-process communication."""
    if size < 2:
        logging.warning("Skipping communication test due to insufficient processes.")
        return

    if rank == 0:
        # Root sends a message to all other processes
        for i in range(1, size):
            comm.send(f"Hello from Rank {rank}", dest=i)
        logging.info("Messages sent from Rank 0.")
    else:
        # Other ranks receive messages from root
        message = comm.recv(source=0)
        logging.info(f"Rank {rank} received message: {message}")

def main():
    """Main function to test mpi4py installation and functionality."""
    # Check mpi4py installation
    check_mpi4py_installation()

    # Initialize MPI environment and validate
    comm, rank, size = validate_mpi_environment()

    try:
        # Test working directory setup
        if rank == 0:  # Only rank 0 tests the working directory
            test_working_directory()

        # Test rank consistency
        test_rank_consistency(comm, rank, size)

        # Test inter-process communication
        test_inter_process_communication(comm, rank, size)

        logging.info(f"All tests passed on Rank {rank}.")
    except AssertionError as e:
        logging.error(f"Assertion failed on Rank {rank}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error on Rank {rank}: {e}")
    finally:
        MPI.Finalize()
        logging.info(f"MPI Finalized on Rank {rank}.")

def simple_test():
    #from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"Hello from rank {rank}")


if __name__ == "__main__":
    main()
    #simple_test()
