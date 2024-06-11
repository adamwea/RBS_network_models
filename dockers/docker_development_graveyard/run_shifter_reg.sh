#!/bin/bash
#SBATCH --job-name=128proc
#SBATCH -A m2043
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --exclusive
#SBATCH --output=./NERSC/output/latest_job_init_error.txt
#SBATCH --error=./NERSC/output/latest_job_init_output.txt
    
#specify the duration of the simulation and the label of the batch run
Duration_Seconds=15
Batch_Run_Label=$SLURM_JOB_NAME

##modules
module load mpich/4.1.1
module load conda

### Uncomment for testing MPI
# Start the Docker container and get its ID
#shifter mpiexec -np ${np} python3 testmpi.py #test mpi
#srun -N ${nodes} -n ${num_MPI_task} -c ${proc_per_task} shifter --image=adammwea/netpyneshifter:v5 python3 testmpi.py --display-allocation
#srun --mpi=pmi2 -n 4 -c 1 shifter --image=adammwea/netpyneshifter:v5 python3 testmpi.py

# ## Run the batch script inside the Docker container
#Test one simulation 
# srun -n 128 shifter --image=adammwea/netpyneshifter:v5 \
#     nrniv -python -mpi /pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/init.py\ 
#     simConfig=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run10_local_debug/gen_0/gen_0_cand_9_cfg.json\
#     netParams=/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/output/240511_Run10_local_debug/240511_Run10_local_debug_netParams.py

###prep batchRun
## parameters
Duration_Seconds=15
#Batch_Run_Label=process_per_sim_inttest
JOB_ID=$SLURM_JOB_ID
cores_per_node=4 # 128 physical cores on perlmutter node
nodes=$SLURM_NNODES
#num_MPI_task=$((nodes*cores_per_node)) # 128 physical cores on the laptop
num_MPI_task=$((cores_per_node)) # 128 physical cores on the laptop
proc_per_task=1

container_run_path=$(python3 NERSC/USER_init_new_batch.py ${Batch_Run_Label})
echo "Container path:" $container_run_path
full_sh_path=$(realpath $0)
echo "Batch script path: ${full_sh_path}"
script_name=$(basename $0) #get the name of this batch script file
full_sh_path_container=${full_sh_path} # set full_sh_path_container to the value of full_sh_path
cp ${full_sh_path_container} ${container_run_path}/${script_name} #save copy of this batch_script in the run_path
echo "Running a shifter for each simulation..."

##Test Batch
cd NERSC
conda activate preshifter
# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=close
python3 batchRun.py -rp ${container_run_path} -d ${Duration_Seconds} ${Batch_Run_Label} # \
# > ${container_run_path}/mpi_output.txt \
# 2> ${container_run_path}/mpi_error.txt
# echo "Batch script finished running inside the Docker container"
# echo "Stopping the Docker container"