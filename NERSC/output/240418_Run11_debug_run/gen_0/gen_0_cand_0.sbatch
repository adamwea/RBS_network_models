#!/bin/bash
#SBATCH --job-name=gen_0_cand_0
#SBATCH -A default
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH -o /mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240418_Run11_debug_run/gen_0/gen_0_cand_0.run
#SBATCH -e /mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240418_Run11_debug_run/gen_0/gen_0_cand_0.err
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=end


module load python
module load conda
module load openmpi
conda activate 2DSims
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=256
export OMP_DISPLAY_AFFINITY=true
export OMP_AFFINITY_FORMAT="host=%H, pid=%P, thread_num=%n, thread affinity=%A"


source ~/.bashrc
cd .
mpirun -n 256 nrniv -python -mpi /mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/init.py simConfig=/mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240418_Run11_debug_run/gen_0/gen_0_cand_0_cfg.json netParams=/mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240418_Run11_debug_run/240418_Run11_debug_run_netParams.py 
wait
    