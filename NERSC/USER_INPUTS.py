import sys
from batch_helper_functions import get_walltime_per_sim

## Run Name
try: USER_run_label = sys.argv[1] ### Change this to a unique name for the batch run
except: USER_run_label = 'debug_run' ### Change this to a unique name for the batch run

##Simulation Duration
try: USER_seconds = int(sys.argv[2]) ### Change this to the number of seconds for the simulation
except: USER_seconds = 5

## Available Methods
USER_method = 'evol'
#USER_method = 'grid'

## Fitness Params
#Set paramaeters for convolving raster data into network activity plot
USER_raster_convolve_params = {
    'binSize': .03*1000, 
    'gaussianSigma': .12*1000, 
    'thresholdBurst': 1.0
    }
USER_plot_fitness_bool = False

## Evol Params
USER_frac_elites = 0.1 # must be 0 < USER_frac_elites < 1. This is the fraction of elites in the population.
USER_pop_size = 8
USER_max_generations = 5
USER_time_sleep = 10 #seconds between checking for completed simulations
USER_maxiter_wait_minutes = 20 #Maximum minutes to wait before new simulation starts before killing generation

## Parallelization
options = ['local', 'mpi_direct', 'hpc_slurm']
# 0 - local
# 1 - NERSC
option = options[1]
if option == 'local':
    USER_mpiCommand = 'mpirun'
    USER_nodes = 1 #This should be set to the number of nodes available
    USER_runCfg_type = 'mpi_bulletin'
    USER_cores_per_node_per_sim = 4 #This should be set to the number of cores desired for each simulation
    USER_cores_per_node = USER_cores_per_node_per_sim    
    USER_walltime = None
    USER_email = None
    USER_custom = None
elif option == 'mpi_direct':
    USER_queue = 'debug' #Options: debug, regular, premium
    USER_mpiCommand = 'mpirun --mca mtl_base_verbose 100'
    USER_allocation = 'm2043' #project ID
    Perlmutter_cores_per_node = int(128/USER_pop_size) #128 physical cores, 256 hyperthreads
    USER_nodes = 1 #This should be set to the number of nodes available    
    USER_runCfg_type = 'mpi_direct'
    USER_cores_per_node_per_sim = Perlmutter_cores_per_node #This should be set to the number of cores desired for each simulation
    #USER_cores_per_node_per_sim = int((Perlmutter_cores_per_node*USER_nodes)/USER_pop_size) #This should be set to the number of cores desired for each simulation
    #assert USER_cores_per_node_per_sim <= (Perlmutter_cores_per_node*USER_nodes)/USER_pop_size, 'USER_cores_per_node_per_sim must be less than or equal to Perlmutter_cores_per_node'    
    USER_cores_per_node = USER_cores_per_node_per_sim
    USER_walltime = None    
    USER_email = None
    #USER_custom_slurm = f'srun -n {Perlmutter_cores_per_node*USER_nodes} check-hybrid.gnu.pm | sort -k4,6 #> output.log 2>&1'
    USER_custom_slurm = f''
    USER_maxiter_wait_minutes = 5 #Maximum minutes to wait before new simulation starts before killing generation
elif option == 'hpc_slurm':
    USER_mpiCommand = 'mpirun'
    USER_allocation = 'm2043' #project ID
    Perlmutter_cores_per_node = 128 #128 physical cores, 256 hyperthreads
    USER_nodes = 1 #This should be set to the number of nodes available    
    USER_runCfg_type = 'hpc_slurm'
    #1 Sim per Node
    USER_cores_per_node_per_sim = Perlmutter_cores_per_node #This should be set to the number of cores desired for each simulation
    #USER_cores_per_node_per_sim = int((Perlmutter_cores_per_node*USER_nodes)/USER_pop_size) #This should be set to the number of cores desired for each simulation
    #assert USER_cores_per_node_per_sim <= (Perlmutter_cores_per_node*USER_nodes)/USER_pop_size, 'USER_cores_per_node_per_sim must be less than or equal to Perlmutter_cores_per_node'    
    USER_cores_per_node = USER_cores_per_node_per_sim
    USER_walltime_per_gen = '01:30:00' # set this value to the maxiumum walltime allowed to charge
    USER_walltime_per_sim = get_walltime_per_sim(USER_walltime_per_gen, USER_pop_size, USER_nodes)
    USER_walltime_per_sim = '00:06:00'
    print(f'USER_walltime_per_sim: {USER_walltime_per_sim}')
    USER_walltime = USER_walltime_per_sim    
    USER_email = 'amwe@ucdavis.edu'
    USER_custom_slurm = f'''
##Custom SLURM Options
#SBATCH -q regular
#SBATCH -C cpu

module load python
module load conda

#
# cray-mpich and cray-libsci conflict with openmpi so unload them
#
module unload cray-mpich
module unload cray-libsci
module use /global/common/software/m3169/perlmutter/modulefiles
module load openmpi

conda activate 2DSims

touch ~/.bashrc
##Custom SLURM Options
'''
# export OMP_PLACES=cores
# export OMP_PROC_BIND=spread
# export OMP_NUM_THREADS={USER_cores_per_node_per_sim}
# export OMP_DISPLAY_AFFINITY=true
# export OMP_AFFINITY_FORMAT="host=%H, pid=%P, thread_num=%n, thread affinity=%A"

# '''    
else: 
    print('Invalid Parallelization Option')
    sys.exit()

## Overwrite and Continue
USER_overwrite_run = False
USER_continue_run = False

