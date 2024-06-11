from mpi4py import MPI
import numpy as np
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Master process
        data = np.zeros((size-1, 10))  # Prepare a data array to gather results
        print("Master process started, waiting for data...")
        for i in range(1, size):
            data[i-1] = comm.recv(source=i, tag=77)
            print(f"Received data from process {i}: {data[i-1]}")
    else:
        # Worker processes
        local_data = np.random.rand(10)  # Generate some data
        print(f"Process {rank} sending data to master...")
        comm.send(local_data, dest=0, tag=77)
        time.sleep(5)  # Simulate some work/wait

if __name__ == "__main__":
    main()