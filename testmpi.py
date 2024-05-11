from mpi4py import MPI

print("MPI version:", MPI.Get_version())
print("MPI library version:", MPI.Get_library_version())
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
print(mpi_rank, mpi_size)
print(MPI.get_vendor())
#print(MPI.Get_library_version())

# from mpi4py import MPI
# from neuron import h
# pc = h.ParallelContext()
# id = int(pc.id())
# nhost = int(pc.nhost())
# print("I am {} of {}".format(id, nhost))