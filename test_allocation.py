import multiprocessing as mp
import os
import psutil

nthreads = psutil.cpu_count(logical=True)
ncores = psutil.cpu_count(logical=False)
nthreads_per_core = nthreads // ncores
nthreads_available = len(os.sched_getaffinity(0))
ncores_available = nthreads_available // nthreads_per_core

assert nthreads == os.cpu_count()
assert nthreads == mp.cpu_count()

print(f'{nthreads}')
print(f'{ncores}')
print(f'{nthreads_per_core}')
print(f'{nthreads_available}')
print(f'{ncores_available}')