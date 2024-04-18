#!/bin/bash
#SBATCH --job-name=gen_0_cand_0
#SBATCH -A default
#SBATCH -t 00:07:30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32.0
#SBATCH -o /mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240417_Run11_debug_run/gen_0/gen_0_cand_0.run
#SBATCH -e /mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240417_Run11_debug_run/gen_0/gen_0_cand_0.err
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=end


source ~/.bashrc
cd .
mpirun -n 32 nrniv -python -mpi /mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/init.py simConfig=/mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240417_Run11_debug_run/gen_0/gen_0_cand_0_cfg.json netParams=/mnt/c/Users/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240417_Run11_debug_run/240417_Run11_debug_run_netParams.py 
wait
    