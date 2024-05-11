from mpi4py import MPI
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
print(mpi_rank, mpi_size)
print(MPI.get_vendor())
print("MPI version:", MPI.Get_version())
print("MPI library version:", MPI.Get_library_version())