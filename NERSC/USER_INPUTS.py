import sys

## Run Name
try: USER_run_label = sys.argv[1] ### Change this to a unique name for the batch run
except: USER_run_label = 'debug_run' ### Change this to a unique name for the batch run

##Simulation Duration
try: USER_seconds = int(sys.argv[2]) ### Change this to the number of seconds for the simulation
except: USER_seconds = 1

## Available Methods
USER_method = 'evol'
#method = 'grid'

## Batch Params
USER_pop_size = 8
USER_max_generations = 5
USER_time_sleep = 10 #seconds between checking for completed simulations
USER_maxiter_wait_minutes = 20 #Maximum minutes to wait before new simulation starts before killing generation

## Parallelization
USER_cores_per_node = 16 #This should be set to the number of cores available on the node
USER_nodes = 1 #This should be set to the number of nodes available

## Evol Params
USER_frac_elites = 0.1 # must be 0 < USER_frac_elites < 1. This is the fraction of elites in the population.

## Overwrite and Continue
USER_overwrite_run = False
USER_continue_run = False