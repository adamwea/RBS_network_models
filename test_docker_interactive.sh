#To run this, run these lines:
#salloc --nodes=1 -C cpu -q interactive -t 00:30:00 --exclusive --image=adammwea/netpyneshifter:v5
#salloc --nodes=1 -C cpu -q interactive -t 04:00:00 --exclusive --image=kpkaur28/neuron:v3
#bash test_shifter_interactive.sh

##modules
# module load mpich/4.1.1
# module load conda

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
Batch_Run_Label=mpi_batch_test_nersc
JOB_ID=$SLURM_JOB_ID


#num_MPI_task=$((nodes*cores_per_node)) # 128 physical cores on the laptop
#num_MPI_task=$((cores_per_node)) # 128 physical cores on the laptop
#proc_per_task=1

container_run_path=$(python3 NERSC/USER_init_new_batch.py ${Batch_Run_Label})
echo "Container path:" $container_run_path
full_sh_path=$(realpath $0)
echo "Batch script path: ${full_sh_path}"
script_name=$(basename $0) #get the name of this batch script file
full_sh_path_container=${full_sh_path} # set full_sh_path_container to the value of full_sh_path
cp ${full_sh_path_container} ${container_run_path}/${script_name} #save copy of this batch_script in the run_path
echo "Running a shifter for each simulation..."

##Test Batch Architectures

# conda activate preshifter
# OpenMP settings:
# export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=close

### srun each simulation ###
# python3 batchRun.py -rp ${container_run_path} -d ${Duration_Seconds} # \
# > ${container_run_path}/mpi_output.txt \
# 2> ${container_run_path}/mpi_error.txt

### srun the batch script ###
# nodes=$SLURM_NNODES
nodes=1 #laptop
#cores_per_node=4 # 128 physical cores on perlmutter node
# tasks_per_node=128 #NESRC
tasks_per_node=4 #laptop
num_MPI_task=$((nodes*tasks_per_node)) # 128 physical cores on perlmutter node
echo "Nodes: ${nodes}"
echo "Number of MPI tasks: ${num_MPI_task}"
mpiexec -bootstrap fork -n ${num_MPI_task} python testmpi.py
#srun -n ${num_MPI_task} --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 python3 testmpi.py
mpiexec -bootstrap fork -n 4 nrniv -mpi test0.hoc
#srun -n ${num_MPI_task} --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -mpi test0.hoc
cd NERSC
mpiexec -np ${num_MPI_task}\  #shifter --image=adammwea/netpyneshifter:v5 \ 
nrniv -mpi -python batchRun_mpi.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}
#srun -n ${num_MPI_task} --cpu_bind=cores shifter --image=adammwea/netpyneshifter:v5 nrniv -mpi -python batchRun_mpi.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}
#\
# > ${container_run_path}/mpi_output.txt \
# 2> ${container_run_path}/mpi_error.txt