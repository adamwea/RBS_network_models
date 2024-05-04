import sys
import os

'''Batch Inputs'''
i = 0
try: USER_run_label = sys.argv[-2] ### Change this to a unique name for the batch run
except: USER_run_label = 'debug_run' ### Change this to a unique name for the batch run

'''SBATCH Inputs'''
USER_email = 'amwe@ucdavis.edu'
USER_JobName = USER_run_label

'''Simulation Inputs'''
script_path = os.path.dirname(os.path.realpath(__file__))
##Simulation Duration
try: USER_seconds = int(sys.argv[-1]) ### Change this to the number of seconds for the simulation
except: USER_seconds = 5
## Simulation method
USER_method = 'evol' #'evol', 'grid', 'asd'
USER_init_script = f'{script_path}/init.py'
USER_cfgFile = f'{script_path}/cfg.py'
## Network Params
USER_netParamsFile = f'{script_path}/netParams.py'
assert os.path.exists(USER_init_script), f'initFile does not exist: {USER_init_script}'
assert os.path.exists(USER_cfgFile), f'cfgFile does not exist: {USER_cfgFile}'
assert os.path.exists(USER_netParamsFile), f'netParamsFile does not exist: {USER_netParamsFile}'

'''Overwrite Inputs'''
USER_skip = False #Skip running the simulation if data already exist
USER_overwrite = False #Overwrite existing batch_run with same name
USER_continue = False #Continue from last completed simulation
if USER_continue: USER_skip = True #continue doesnt really work with out skip

'''Evol Params'''
USER_pop_size = 10
USER_frac_elites = 0.1 # must be 0 < USER_frac_elites < 1. This is the fraction of elites in the population.
USER_max_generations = 3000
USER_time_sleep = 10 #seconds between checking for completed simulations
maxiter_wait_minutes = 2*60 #Maximum minutes to wait before starting new Generation
USER_maxiter_wait = maxiter_wait_minutes*60/USER_time_sleep
USER_num_elites = int(USER_frac_elites * USER_pop_size) if USER_frac_elites > 0 else 1
USER_mutation_rate = 0.7
USER_crossover = 0.5


'''Plotting Inputs'''
##Plotting Params
USER_plot_fitness_bool = True
USER_plot_NetworkActivity = False
USER_plotting_path = 'NERSC/plots/'
#USER_ploting_path = None #prevent plotting even if USER_plot_fitness_bool = True
USER_figsize = (10, 10)
# Network Activity Plotting Params
USER_plotting_params = {
    'fresh_plots': True,
    'saveFig': USER_plotting_path,
    'figsize': USER_figsize,
    'NetworkActivity': {
        'figsize': USER_figsize,
        #limits (will override modifiers)
        'ylim': None,
        'xlim': None,
        #range mods
        'yhigh100': 1.05, #high modifier limit for y axis
        'ylow100': 0.95, #low modifier limit for y axis
        'saveFig': USER_plotting_path,
        'title_font': 11
        }
    }

'''Fitness Inputs'''
## Fitness Params
#Set paramaeters for convolving raster data into network activity plot
USER_raster_convolve_params = {
    #'binSize': .03*1000,
    'binSize': .03*250,
    #'gaussianSigma': .12*1000, 
    'gaussianSigma': .12*250,
    'thresholdBurst': 1.0
    }
USER_raster_crop = None

''' Parallelization Inputs'''
options = ['mpi_bulletin_Laptop', 
           'mpi_bulletin_Server', 
           'mpi_bulletin_NERSC', 
           'mpi_direct', 
           'hpc_slurm']
option = options[0]
if option == 'benshalom-labserver':
    USER_runCfg_type = 'mpi_bulletin'    
    USER_nodes = 1 #1 server = 1 node
    Server_cores_per_node = 10 #I think there are like 48 cores available, but they are shared with other users
    USER_cores_per_node = Server_cores_per_node
    USER_total_cores = Server_cores_per_node*USER_nodes
    #USER_JobName = f'mpiexec_test_{USER_nodes}x{USER_cores_per_node}'
    #USER_MPI_run_keep = True
    #USER_walltime = None
    #USER_email = None
    #USER_custom_slurm = None
    #USER_allocation = None
    #USER_mpiCommand = None
elif option == 'mpi_bulletin_Laptop':
    USER_pop_size = 4
    USER_runCfg_type = 'mpi_bulletin'    
    USER_nodes = 1 #This should be set to the number of nodes available
    Laptop_cores_per_node = 8 #8 physical cores, 16 hyperthreads
    USER_cores_per_node = Laptop_cores_per_node
    USER_total_cores = Laptop_cores_per_node*USER_nodes
    USER_JobName = f'mpiexec_test_{USER_nodes}x{USER_cores_per_node}'
    USER_MPI_run_keep = True
    USER_walltime = None
    USER_email = None
    USER_custom_slurm = None
    USER_allocation = None
    USER_mpiCommand = None
elif option == 'mpi_bulletin_Server':
    USER_runCfg_type = 'mpi_bulletin'    
    USER_nodes = 1 #This should be set to the number of nodes available
    Server_cores_per_node = 40 #???8 physical cores, 16 hyperthreads TODO: Find out the number of cores per node
    USER_cores_per_node = Server_cores_per_node
    USER_total_cores = Server_cores_per_node*USER_nodes
    USER_JobName = f'mpiexec_test_{USER_nodes}x{USER_cores_per_node}'
    USER_MPI_run_keep = True
    USER_walltime = None
    USER_email = None
    USER_custom_slurm = None
    USER_allocation = None
    USER_mpiCommand = None
elif option == 'mpi_bulletin_NERSC':
    #USER_pop_size = 128 # Population sizes
    USER_pop_size = 100
    USER_queue = 'debug' #Options: debug, regular, premium
    USER_runCfg_type = 'mpi_bulletin'    
    USER_allocation = 'm2043' #project ID
    USER_walltime = "00:30:00"    
    USER_email = "amwe@ucdavis.edu"
    USER_nodes = 1 #This should be set to the number of nodes available
    #Perlmutter_cores_per_node = 256 #128 physical cores, 256 hyperthreads
    Perlmutter_cores_per_node = 128 #128 physical cores, 256 hyperthreads
    USER_cores_per_node = Perlmutter_cores_per_node
    #USER_cpus_per_task = 2
    USER_total_cores = Perlmutter_cores_per_node*USER_nodes
    USER_JobName = f'4Nodes_{USER_nodes}x{USER_cores_per_node}'
    #USER_MPI_run_keep = True
    USER_custom_slurm = None
    USER_mpiCommand = None
elif option == 'mpi_direct':
    USER_queue = 'debug' #Options: debug, regular, premium
    USER_runCfg_type = 'mpi_direct'    
    USER_allocation = 'm2043' #project ID
    USER_walltime = "00:30:00"    
    USER_email = "amwe@ucdavis.edu"

    USER_nodes = 1 #This should be set to the number of nodes available
    Perlmutter_cores_per_node = 256 #128 physical cores, 256 hyperthreads
    #Perlmutter_cores_per_node = 128 #128 physical cores, 256 hyperthreads
    USER_MPI_processes_per_sim = Perlmutter_cores_per_node//USER_pop_size
    #assert that USER_MPI_processes_per_sim is a perfect square
    #assert USER_MPI_processes_per_sim**.5 % 1 == 0, 'USER_MPI_processes_per_sim should be a perfect square. Adjust population size.'
    #square_root = int(USER_MPI_processes_per_sim**.5)
    #if 16 process per sim per node
    #USER_MPI_processes_per_node = 4
    #USER_OMP_threads_per_process_per_node = 16 #ranks
    #USER_OMP_threads_per_process = 256
    USER_MPI_processes_per_node = 128//USER_pop_size
    USER_MPI_processes_per_node = 8
    #USER_MPI_processes_per_node = 32
    # USER_MPI_processes_per_node = square_root
    # USER_OMP_threads_per_process_per_node = square_root
    #USER_OMP_threads_per_process = USER_OMP_threads_per_process_per_node*USER_nodes # process
    USER_OMP_threads_per_process_per_core = 2
    USER_MPI_rank_per_unit = USER_OMP_threads_per_process_per_core
    USER_MPI_rank_per_unit = 1
    #unit = "slot"
    ppr_unit = "core"
    bind_unit = "core"
    #unit = "node"
    #USER_OMP_threads_per_process = '16'
    #    taskset -c $unused_cores
    USER_mpiCommand = f'''
    mpirun --mca mtl_base_verbose 100
    --use-hwthread-cpus
    --nooversubscribe 
    --map-by ppr:{USER_MPI_rank_per_unit}:{ppr_unit}    
    --bind-to {bind_unit}
    --report-bindings
    --display-map
    --display-topo
    --display-devel-map
    '''
    #remove returns from USER_mpiCommand
    USER_mpiCommand = ' '.join(USER_mpiCommand.split())
    #assert USER_MPI_processes_per_node*USER_OMP_threads_per_process_per_node == Perlmutter_cores_per_node, 'USER_MPI_processes_per_node*USER_OMP_threads_per_process must should be equal to Perlmutter_cores_per_node'
    # assert [USER_nodes*Perlmutter_cores_per_node] == [
    #     USER_MPI_processes_per_node*USER_OMP_threads_per_process_per_node*USER_pop_size
    #     ], 'node/core/process/thread allocation has some error'
    USER_JobName = f'unused_cores_test_{USER_nodes}x{USER_MPI_processes_per_node}'
    #USER_cores_per_node_per_sim = int(Perlmutter_cores_per_node/USER_pop_size) #128 physical cores, 256 hyperthreads
    #USER_threads_process = 
    #USER_cores_per_sim  = USER_cores_per_node_per_sim * USER_nodes
    
    #USER_cores_per_node_per_sim = int((Perlmutter_cores_per_node*USER_nodes)/USER_pop_size) #This should be set to the number of cores desired for each simulation
    #assert USER_cores_per_node_per_sim <= (Perlmutter_cores_per_node*USER_nodes)/USER_pop_size, 'USER_cores_per_node_per_sim must be less than or equal to Perlmutter_cores_per_node'    

    #USER_custom_slurm = f'srun -n {Perlmutter_cores_per_node*USER_nodes} check-hybrid.gnu.pm | sort -k4,6 #> output.log 2>&1'
    USER_custom_slurm = '''
export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose
export FI_LOG_LEVEL=debug
export OMPI_MCA_pml=ob1

# Get the list of used cores
export $(python NERSC/get_used_cores.py)

# Get the total number of cores
total_cores=$(nproc)

# Generate the list of unused cores
unused_cores=""
for ((i=0; i<$total_cores; i++)); do
    if [[ $used_cores != *"$i"* ]]; then
        unused_cores="$unused_cores,$i"
    fi
done
unused_cores=${unused_cores#,}
echo "Total cores: $total_cores"
echo "Used cores: $used_cores"
echo "Unused cores: $unused_cores"

'''
    #USER_maxiter_wait_minutes = 5 #Maximum minutes to wait before new simulation starts before killing generation
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
    #USER_walltime_per_sim = get_walltime_per_sim(USER_walltime_per_gen, USER_pop_size, USER_nodes)
    USER_walltime_per_sim = '00:06:00'
    #print(f'USER_walltime_per_sim: {USER_walltime_per_sim}')
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

