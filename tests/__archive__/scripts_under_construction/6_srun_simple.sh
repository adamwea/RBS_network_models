#!/bin/bash

#SBATCH --job-name=CDKL5_DIV21_campaign_1
#SBATCH -A m2043
#SBATCH -t 10:00:00
#SBATCH -N 12
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --exclusive
#SBATCH --image=docker:adammwea/netsims_docker:v1

# Optional: Debugging output
export FI_LOG_LEVEL=warn
#export FI_PROVIDER="ofi_rxm"

# Restore system defaults and load necessary modules
# source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh
# module purge

#shifterimg pull adammwea/netsims_docker:v1
#shifterimg images -u adammwea
#shifterimg pull adammwea/netpyneshifter:v5
module load conda
source activate netsims_env
module load PrgEnv-gnu   # Replace with PrgEnv-intel or PrgEnv-cray if needed
module load cray-mpich   # Cray MPICH implementation

# Log loaded modules
module list

#srun shifter --image=adammwea/netpyneshifter:v5 nproc

# srun -N 4 -n 128 shifter --image=adammwea/netpyneshifter:v5 \
#   nrniv -python -mpi /pscratch/sd/a/adammwea/RBS_network_simulations/modules/simulation_config/init.py \
#   simConfig=/pscratch/sd/a/adammwea/zRBS_network_simulation_outputs/241126_Run20_improved_netparams/gen_0/gen_0_cand_3_cfg.json \
#   netParams=/pscratch/sd/a/adammwea/zRBS_network_simulation_outputs/241126_Run20_improved_netparams/241126_Run20_improved_netparams_netParams.py


# Run your Python script
#python /pscratch/sd/a/adammwea/RBS_network_simulations/optimization_scripts/batchRun_evol_srun_direct.py

# export OMP_NUM_THREADS=128
# export SLURM_CPUS_PER_TASK=128

#srun -N 1 -n 1 --cpus-per-task=128 --hint=nomultithread --cpu-bind=cores \
# python /pscratch/sd/a/adammwea/RBS_network_simulations/optimization_scripts/batchRun_evol_srun_direct.py
# python /pscratch/sd/a/adammwea/workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/_2_batchrun_optimization/batch_run_evol.py
python /pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/5_batch_run_evol_vCurrent.py