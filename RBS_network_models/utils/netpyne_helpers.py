import os
from pathlib import Path
# import sim
# from specs import SimConfig
from netpyne import sim, specs
import glob
import json
import warnings
warnings.filterwarnings("always", category=UserWarning)  # Ignore user warnings
import numpy as np
import numbers
from copy import deepcopy
import subprocess
from time import sleep
import pickle

'''
# TODO LIST: =====================================================================
Hypothetical functions/procedures to write at some point:
1. - Space Saver Procedure:
    - extract and save netParams files next to all _data.pkl and/or _data.json files found recursively in a target directory.
    - delete all _data.pkl and/or _data.json files found recursively in a target directory.
    ** reruns should only require loading netParams and simConfig from the saved files. **
    - required funcs and subfunctions:
        - find all _data.pkl and/or _data.json files in a target directory recursively.
        - extract netParams and simConfig from each file/simulation/batch as needed.
        - save netParams and simConfig to a file next to netParams.json/.pkl and simConfig.json/pkl as needed. Probably pkl since it's more robust and can handle more complex objects.
        - delete all _data.pkl and/or _data.json files found recursively in a target directory.
        - 2025-07-01 14:43:06 - completed all of these I think.

1.5 - related but seperate:
    - use rsync commands to copy all data in pscratch to benshalom cfs
    - use globus to sync all data in cfs back to benshalom nas
    ** this should probably go in seperate module actually...**
    
2. - Modify Any netParam or SimConfig attribute.
    ** This will likely require a bunch of specific functions to modify specific attributes. **
    ** Hopefully I can just parse by data type and structure.**
        2025-06-28 11:11:10 - done, I think. What I have seems to work for now, may need tweaking as I go.
        
2025-07-01 14:43:21
3. - Remove unecessery '.' characters from all _data.pkl and _data.json files prior to running space saver protocol. in the future.
'''

# def jobPathAndNameForSim(index):
#     if batch.method in ['optuna', 'sbi']:
#         jobName = "trial_" + str(ngen)
#     else:
#         jobName = "gen_" + str(ngen) + "_cand_" + str(candidate_index)
#     return jobName, genFolderPath + '/' + jobName

# def config_perlmutter_env():
#     bash = """
# module load conda
# conda activate my_mpi4py_env
# module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
# module load cray-mpich
# export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH
# export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)
# """
#     subprocess.run(bash, shell=True, executable="/bin/bash")

# def validate_perlmutter_env():
#     bash = """#!/bin/bash
# nrniv -mpi -python - <<EOF
# from neuron import h
# print("MPI load OK, h =", h)
# EOF
# """
#     subprocess.run(bash, shell=True, executable="/bin/bash")


def load_sim_data(data_file: str) -> list:
    """
    Load simulation data from a specified file.
    
    Parameters:
    data_file (str): Path to the simulation data file.
    
    Returns:
    dict: Loaded simulation data.
    """
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"Simulation data file {data_file} does not exist.")
    
    try:
        sim.clearAll()  # Clear any existing simulation data
    except:
        pass
    
    # Check if the file is a JSON or a pickle file
    if data_file.endswith('.json') or data_file.endswith('.pkl'):
        sim.load(data_file)  # Load the simulation data
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .pkl file.")
    
    return sim.allSimData, sim.net.allPops, sim.net.allCells

def _wait_for_jobs_to_finish(sim_dict, output_dir, pids, **kwargs):
    """
    """
    print(f"Waiting for jobs...")
    
    #total_jobs = len(pids)
    #total_perms = len(permutations)
    total_jobs = len(sim_dict)  # total number of jobs to wait for
    # assert total_jobs == total_perms, \
    #     f"Total jobs ({total_jobs}) does not match total permutations ({total_perms}). Please check the permutations and job IDs."
    num_iters = 0
    jobs_completed = 0
    # print "PID's: %r" %(pids)
    # start fitness calculation
    completed_jobs=[]
    while jobs_completed < total_jobs:
        #unfinished = [i for i, x in enumerate(fitness) if x is None]
        #for candidate_index in unfinished:
        for simLabel, sim_paths in sim_dict.items():
            if simLabel in completed_jobs:
                continue # skip already completed jobs, dont reload data for them.
            try:  # load simData and evaluate fitness
                #_, jobPath = jobPathAndNameForCand(candidate_index)
                jobPath = os.path.join(output_dir, simLabel, simLabel)
                dataPath = jobPath + '_data.json'
                simData = None
                
                # check if simData exists in json or pkl format and if it can be loaded
                if os.path.isfile(dataPath):
                    with open(dataPath) as file:
                        simData = json.load(file)['simData']
                else:
                    dataPath = jobPath + '_data.pkl'
                    #print(f'\tChecking for data at {dataPath}')
                    if os.path.isfile(dataPath):
                        with open(dataPath, 'rb') as file:
                            simData = pickle.load(file)['simData']
                if simData:
                    #fitness[candidate_index] = fitnessFunc(simData, **fitnessFuncArgs)
                    #if collectSummaryStats:
                    #    sum_statistics = summaryStatsFunc(simData, **summaryStatsArgs)
                    jobs_completed += 1
                    completed_jobs.append(simLabel)
                    #print('  Candidate %d fitness = %.1f' % (candidate_index, fitness[candidate_index]))
                    print(f"  Candidate {simLabel} completed with data at {dataPath}")
            except Exception as e:
                #err = "There was an exception evaluating candidate %d:" % (candidate_index)
                #print(("%s \n %s" % (err, e)))
                #print()
                print(f"  Candidate {simLabel} failed with error: {e}")
        #num_iters += 1
        print('completed: %d' % (jobs_completed))
        # if num_iters >= args.get('maxiter_wait', 5000):
        #     print(
        #         "Max iterations reached, the %d unfinished jobs will be canceled and set to default fitness"
        #         % (len(unfinished))
        #     )
        #     for canditade_index in unfinished:
        #         fitness[canditade_index] = indexToRerun  # rerun those that didn't complete;
        #         if collectSummaryStats:
        #             sum_statistics = [
        #                 -1 * indexToRerun for _ in range(summaryStatsLength)
        #             ]  # -indexToRerun (i.e. -maxFitness) for size of summ stats
        #         jobs_completed += 1
        #         try:
        #             if 'scancelUser' in kwargs:
        #                 os.system('scancel -u %s' % (kwargs['scancelUser']))
        #             else:
        #                 os.system(
        #                     'scancel %d' % (jobids[candidate_index])
        #                 )  # terminate unfinished job (resubmitted jobs not terminated!)
        #         except:
        #             pass
        #sleep(args.get('time_sleep', 1))
        sleep(kwargs.get('time_sleep', 1))  # wait for a second before checking again

def run_sims(sim_dict, output_dir, 
              parallel=True, init_path=None, mpi=False, slurm=False, 
              tasks_per_node=1,
              overwrite=False):
    """
    Run the simulations in parallel or sequentially.
    
    input:
    - sim_dict: dict, where keys are simulation labels and values are dictionaries with 'netParams' and 'cfg' paths.
    - output_dir: str, directory to save the simulation results.
    - parallel: bool, if True, run simulations in parallel using multiprocessing or MPI.
    - mpi: bool, if True, run simulations using MPI.
    - slurm: bool, if True, run simulations using SLURM.
    - overwrite: bool, if True, overwrite existing simulation data.
    """
    pids = [] # only used for MPI parallel runs, to keep track of the job IDs.
    for simLabel, sim_paths in sim_dict.items():
        print(f"Running simulation: {simLabel}")
        print(f"Configuration path: {sim_paths['cfg']}")
        print(f"Network parameters path: {sim_paths['netParams']}")             
        
        if parallel:
            if mpi:
                
                print("Running simulations in parallel using MPI...")
                assert init_path is not None, "init_path must be provided for MPI simulations."
                jobPath = os.path.join(output_dir, simLabel)
                kwargs={
                    'folder': jobPath, # folder from which to run the simulation commands - where batch files will save I think.   
                }
                if slurm:
                    print("Using SLURM for parallel execution...")
                    pid = mpi_run_sim(
                        cfg_path=sim_paths['cfg'],
                        net_params_path=sim_paths['netParams'],
                        init_path=init_path,
                        jobName=simLabel,
                        jobPath=jobPath,
                        slurm=True,
                        tasks_per_node=tasks_per_node,
                        **kwargs,                        
                    )
                    pids.append(pid)                    
                else:
                    raise NotImplementedError("non-SLURM mpi parallel simulation runs are not implemented yet.")
            else:
                raise NotImplementedError("Parallel simulation runs using multiprocessing are not implemented yet.")
        else:
            raise NotImplementedError("Sequential simulation runs are not implemented yet.")
        
    if mpi:
        # if running in parallel with MPI, return the list of job IDs
        #print(f"Submitted {len(pids)} jobs to the queue.")
        if len(pids) == 0:
            print("No jobs were submitted. Please check the configuration and parameters.")
        if len(pids)>0:
            print(f"Submitted {len(pids)} jobs to the queue.")
            try:
                assert jobPath is not None, "jobPath must be specified to save the job IDs."
                sleep(0.1)
                # read = proc.stdout.read()
                #with open('./pids.pid', 'a') as file:
                with open(os.path.join(jobPath, 'pids.pid'), 'a') as file:
                    file.write(str(pids))
            except Exception as e:
                print(f"Error saving job IDs to file: {e}")
        kwargs={
            # wait 1 * 6000 seconds before canceling all simulations
            # this is to ensure that the simulations have enough time to finish
            # 6000s = 100 minutes, which is a reasonable time for most simulations
            # 'time_sleep': 1, # 5 seconds sleep time between checks for job completion
            # 'maxiter_wait': 6000,  # max number of iterations to wait for jobs to finish
        }
        _wait_for_jobs_to_finish(sim_dict, output_dir, pids, **kwargs)
        
        return pids

def _jobStringMPIDirect(custom, folder, command):
    return f"""#!/bin/bash
{custom}
cd {folder}
{command}
    """

def mpi_run_sim(
    cfg_path: str,
    net_params_path: str, 
    init_path: str, 
    #mpi: bool = True,
    jobName: str = 'mpi_simulation',
    jobPath: str = None, 
    slurm: bool = False,
    tasks_per_node: int = 1,
    submit_type: str = 'mpi_direct',
    executor: str = 'sh', 
    #overwrite: bool = False,
    custom: str = '',
    **kwargs
    ) -> None:
    '''
    Run a simulation using MPI with the specified configuration and network parameters.
        NOTE: Much of this function is copied from netpyne.batchtools.utils.py
        
    Parameters:
        cfg_path (str): Path to the simulation configuration file.
        net_params_path (str): Path to the network parameters file.
        init_path (str): Path to the initialization script for the simulation.
        jobName (str): Name of the job to be submitted.
        jobPath (str): Path where the job script will be saved.
        slurm (bool): If True, use SLURM for job submission.
        tasks_per_node (int): Number of tasks per node for the simulation.
        submit_type (str): Type of submission, e.g., 'mpi_direct'.
        executor (str): Command to execute the job script.
        custom (str): Custom commands to include in the job script.
        **kwargs: Additional keyword arguments for flexibility.
    Returns:
        pids (list): List of process IDs of submitted jobs.
    '''
    pids = []  # List to store process IDs of submitted jobs
    
    if slurm:        
        mpiCommand="srun"
        numproc=tasks_per_node
        nrnCommand="nrniv"
        script=init_path
        cfgSavePath=cfg_path
        netParamsSavePath=net_params_path
        command = '%s -n %d %s -python -mpi %s simConfig=%s netParams=%s ' % (
            mpiCommand,
            numproc,
            nrnCommand,
            script,
            cfgSavePath,
            netParamsSavePath,
        )
    else:
        raise NotImplementedError("Non-SLURM MPI parallel simulation runs are not implemented yet.")
    
    assert jobPath is not None, "jobPath must be specified."
    
    #HACK # add jobName to jobPath
    jobPath = os.path.join(jobPath, jobName)
    
    folder = kwargs.get('folder', '.')
    
    # ----------------------------------------------------------------------
    # run on local machine with <nodes*coresPerNode> cores
    # ----------------------------------------------------------------------
    if submit_type == 'mpi_direct':
        #executer = '/bin/bash'
        executer = executor
        jobString = _jobStringMPIDirect(custom, folder, command)
        
    # ----------------------------------------------------------------------
    # save job and run
    # ----------------------------------------------------------------------
    print('Submitting job ', jobName)
    print(jobString)
    print('-' * 80)
    # save file
    batchfile = '%s.sbatch' % (jobPath)
    with open(batchfile, 'w') as text_file:
        text_file.write("%s" % jobString)
        
    if submit_type == 'mpi_direct':
        with open(jobPath + '.run', 'a+') as outf, open(jobPath + '.err', 'w') as errf:
            pids.append(subprocess.Popen([executer, batchfile], stdout=outf, stderr=errf,
                                        start_new_session=True).pid)
    else: #mpi_bulletin # NOTE: aw 2025-07-02 10:53:01 idk if I ever care to use this, but leaving it here for now.
        with open(jobPath + '.jobid', 'w') as outf, open(jobPath + '.err', 'w') as errf:
            pids.append(subprocess.Popen([executer, batchfile], stdout=outf, stderr=errf,
                                        start_new_session=True).pid)
    
    print(f"Submitted job {jobName} with PID {pids[-1]} to {jobPath}.")        
    return pids[0]

def import_evol_params(evol_path):
    """Import evolution parameters from a specified path.
    Parameters:
    evol_path (str): Path to the evolution parameters file.
    Returns:
    dict: Evolution parameters.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("evol_params", evol_path)
    evol_params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evol_params)
    
    # Assuming the evolution parameters are stored in a variable named 'evol_params' in the file.
    evol_params = evol_params.params if hasattr(evol_params, 'params') else None
    
    if evol_params is None:
        raise ValueError("evol_params is None, please check the evolution parameters file.")
    
    return evol_params

def _mutate_mean_keep_stdev(values, old_mu, new_mu, lower_bound=None, upper_bound=None):
    """ Adjust the mean of a set of values while keeping the standard deviation constant.
    
    Args:
        values (list or np.ndarray): The original data values.
        old_mu (float): The original mean.
        new_mu (float): The desired new mean.
        lower_bound (float, optional): Lower bound for the transformed data. Defaults to None.
        upper_bound (float, optional): Upper bound for the transformed data. Defaults to None.
        
    Returns:
        list: The transformed data with the new mean.
    """
    
    # Original data
    #data = np.random.normal(loc=100, scale=20, size=1000)
    data = values

    # Compute current mean and std deviation
    mu_old = np.mean(data)
    sigma_old = np.std(data)
    
    # check
    # TODO: same problem as stdev function below, need to investigate.
    #assert mu_old == old_mu, f"Expected old mean {old_mu}, but got {mu_old}."
    
    # mu difference
    mu_diff = new_mu - mu_old

    # Desired new mean
    #mu_new = new_mu
    mu_new = mu_old + mu_diff  # Adjusting the new mu to account for the difference
    
    # if mu_new < 0:
    #     mu_new = 0  
        
    if mu_new < lower_bound:
        mu_new = lower_bound
        
    if mu_new > upper_bound:
        mu_new = upper_bound
    
    # Transform the data
    #transformed_data = [(x - mu_old) / sigma_old * sigma_new + mu_old for x in data]
    transformed_data = [(x - mu_old) / sigma_old * (mu_new - mu_old) + mu_new for x in data]
    
    mu_new = np.mean(transformed_data)
    #assert mu_new == new_mu, f"Expected new mean {new_mu}, but got {mu_new}."
    
    return transformed_data

    # Verify the result
    # print("Original mean:", mu_old)
    # print("Original std dev:", sigma_old)
    # print("New mean:", np.mean(transformed_data))
    # print("New std dev:", np.std(transformed_data))
    
def _mutate_stdev_keep_mean(values, old_sigma, new_sigma, lower_bound=None, upper_bound=None):
    """ 
    Adjust the standard deviation of a set of values while keeping the mean constant.
    Args:
        values (list or np.ndarray): The original data values.
        old_sigma (float): The original standard deviation.
        new_sigma (float): The desired new standard deviation.
        lower_bound (float, optional): Lower bound for the transformed data. Defaults to None.
        upper_bound (float, optional): Upper bound for the transformed data. Defaults to None.
    Returns:
        list: The transformed data with the new standard deviation.
    """
    # Original data
    #data = np.random.normal(loc=100, scale=20, size=1000)
    data = values

    # Compute current mean and std deviation
    mu_old = np.mean(data)
    sigma_old = np.std(data)
    
    # check
    #TODO: seems like sigma measured here, and implemented during netparams build is different... need to investigate.
    # - for now I'm just going to shift the standard by the expected shift - and not focus on the exact value.
    #assert sigma_old == old_sigma, f"Expected old std dev {old_sigma}, but got {sigma_old}."
    
    # sigma difference
    sigma_diff = new_sigma - old_sigma

    # Desired new std deviation
    #sigma_new = new_sigma
    sigma_new = sigma_old + sigma_diff  # Adjusting the new sigma to account for the difference
    
    if sigma_new < 0:
        sigma_new = 0
        
    if sigma_new < lower_bound:
        sigma_new = lower_bound
    
    if sigma_new > upper_bound:
        sigma_new = upper_bound

    # Transform the data
    transformed_data = [(x - mu_old) / sigma_old * sigma_new + mu_old for x in data]
    
    sigma_new = np.std(transformed_data)
    #assert sigma_new == new_sigma, f"Expected new std dev {new_sigma}, but got {sigma_new}."
    
    return transformed_data

def save_sim_cfg_and_net_params(net_params: specs.NetParams, sim_cfg: specs.SimConfig, simLabel: str, saveFolder: str = None, overwrite: bool = False, load: bool = False) -> None:
    """
    Save the simulation configuration and network parameters to JSON files.
    
    Args:
        net_params (specs.NetParams): The network parameters to save.
        sim_cfg (specs.SimConfig): The simulation configuration to save.
        simLabel (str): A label for the simulation, used in the filename.
        saveFolder (str, optional): The folder where the files will be saved. Defaults to the current working directory.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        load (bool, optional): Whether to try to load existing files before saving. Defaults to False.
    """
    # Validate inputs
    assert saveFolder is not None, "saveFolder must be specified."
    assert simLabel is not None, "simLabel must be specified."
    
    #remove mapping from netParams #TODO: figure out how to actually take advantage of this
    net_params.mapping = {}
    
    #sim.initialize(netParams=net_params, simConfig=sim_cfg)  # Create the simulation with the provided parameters
    #sim.cfg.simLabel = simLabel  # Set the simulation label
    #sim.cfg.saveFolder = os.path.join(saveFolder, simLabel)
    sim_cfg.simLabel = simLabel  # Set the simulation label
    saveFolder = os.path.join(saveFolder, simLabel)  # Ensure the save folder is specific to the simulation label
    sim_cfg.saveFolder = saveFolder  # Set the save folder for the simulation configuration
    #sim_cfg.saveFolder = os.path.join(saveFolder, simLabel)  # Set the save folder

    netParams_path = os.path.join(saveFolder, f"{simLabel}_netParams.json")
    sim_cfg_path = os.path.join(saveFolder, f"{simLabel}_cfg.json")
    
    # sim.saveNetParams(netParams_path)  # Save the network parameters
    # sim.saveSimCfg(sim_cfg_path)  # Save the simulation configuration
    net_params.save(netParams_path)  # Save the network parameters
    sim_cfg.save(sim_cfg_path)  # Save the simulation configuration
    
    return netParams_path, sim_cfg_path

    #print(f"Saved netParams to {netParams_path} and simConfig to {sim_cfg_path}.")

def validate_sim_cfg_and_net_params(sim_data_path, extract=True) -> None:
    '''
    Given a sim_data_path, validate that the simConfig and netParams files exist.
    If they do not exist, extract them from the sim_data_path and save them to the same directory as the sim_data_path.
    
    If extract is false, and the files do not exist, raise an error.
    '''
    if not (sim_data_path.endswith('_data.json') or sim_data_path.endswith('_data.pkl')):
        raise ValueError(f"Expected a file ending with '_data.json' or '_data.pkl', got '{sim_data_path}'.")
    
    expected_cfg_path = sim_data_path.replace("_data.json", "_cfg.json") if sim_data_path.endswith('_data.json') else sim_data_path.replace("_data.pkl", "_cfg.json")
    expected_net_params_path = sim_data_path.replace("_data.json", "_netParams.json") if sim_data_path.endswith('_data.json') else sim_data_path.replace("_data.pkl", "_netParams.json")
    
    if not os.path.exists(expected_cfg_path) or not os.path.exists(expected_net_params_path):
        if extract:
            if not os.path.exists(expected_cfg_path):
                print(f"SimConfig file {expected_cfg_path} does not exist. Extracting from {sim_data_path}.")
                sim_cfg = extract_cfg_from_data(sim_data_path)
                sim_cfg.save(expected_cfg_path) # NOTE: cant save these files with additional '.' in the name so netpyne can parse .ext correctly.
                assert os.path.exists(expected_cfg_path), f"SimConfig file {expected_cfg_path} was not saved correctly."
            if not os.path.exists(expected_net_params_path):
                print(f"NetParams file {expected_net_params_path} does not exist. Extracting from {sim_data_path}.")
                net_params = load_net_params_from_data(sim_data_path)
                net_params.save(expected_net_params_path)
                assert os.path.exists(expected_net_params_path), f"NetParams file {expected_net_params_path} was not saved correctly."
        else:
            raise FileNotFoundError(f"SimConfig or NetParams files do not exist for {sim_data_path} and extraction is disabled.")

def space_saver_protocol(target_dirs: list[str] = None, recursive: bool = True, overwrite_sim_cfg: bool = False,
         overwrite_net_params: bool = False, **kwargs: dict
         ) -> None:
    if target_dirs is None:
        raise ValueError("No target directories provided. Please provide a list of directories to search for data files.")
    
    space_saved = 0
    space_used = 0   
    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            print(f"Target directory {target_dir} does not exist. Skipping.")
            continue
        
        # Find all _data.pkl and/or _data.json files in the target directory recursively
        json_files = get_sim_json_paths(target_dir, recursive=recursive)
        pkl_files = get_sim_pkl_paths(target_dir, recursive=recursive)        
        print(f"Found {len(json_files)} JSON files and {len(pkl_files)} PKL files in {target_dir}.")
        
        # Process each JSON file
        print("Processing JSON files...")
        for json_file in json_files:
            su, ss = _space_saver_loop(json_file, overwrite_sim_cfg=overwrite_sim_cfg,
                                          overwrite_net_params=overwrite_net_params)
            
            # Update space used and saved
            space_used += su
            space_saved += ss
        
        # Process each PKL file
        print("Processing PKL files...")
        for pkl_file in pkl_files:
            su, ss = _space_saver_loop(pkl_file, overwrite_sim_cfg=overwrite_sim_cfg,
                                          overwrite_net_params=overwrite_net_params)
            
            # Update space used and saved
            space_used += su
            space_saved += ss
            
    print(f"Total space used: {space_used / (1024 ** 2):.2f} MB")
    print(f"Total space saved: {space_saved / (1024 ** 2):.2f} MB")
    print(f"Net space saved: {(space_saved - space_used) / (1024 ** 2):.2f} MB")  

def _space_saver_loop(data_file: str, overwrite_sim_cfg: bool = False, 
                       overwrite_net_params: bool = False, 
                       space_used: int = 0, space_saved: int = 0) -> None:
    '''
    '''

    cfg_success = False
    net_params_success = False
    
    # check data_file extension
    if not (data_file.endswith('_data.json') or data_file.endswith('_data.pkl')):
        raise ValueError(f"Expected a file ending with '_data.json' or '_data.pkl', got '{data_file}'.")
    if data_file.endswith('_data.json'):
        expected_cfg_path = data_file.replace("_data.json", "_cfg.json")
        expected_net_params_path = data_file.replace("_data.json", "_netParams.json")
    elif data_file.endswith('_data.pkl'):
        expected_cfg_path = data_file.replace("_data.pkl", "_cfg.json")
        expected_net_params_path = data_file.replace("_data.pkl", "_netParams.json")
        
    # check if data_file has any information in it
    if data_file.endswith('_data.json'):
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            # if not data:
            #     raise ValueError(f"Data file {data_file} is empty or does not contain valid JSON.")
            assert data, f"Data file {data_file} is empty or does not contain valid JSON."
        except json.JSONDecodeError as e:
            #raise ValueError(f"Error decoding JSON from {data_file}: {e}")
            warnings.warn(f"Error decoding JSON from {data_file}: {e}. This file may be empty or not contain valid JSON.", UserWarning)

            # no data to be saved here anyway, I guess.
            print(f"\tNo data to be saved from {data_file}. It is either empty or does not contain valid JSON.")
            print(f"\tSkipping extraction for {data_file}.")
            return 0, 0
        
    # elif data_file.endswith('_data.pkl'):
    #     try:
    #         with open(data_file, 'rb') as f:
    #             data = sim.loadSimData(f)  # Use the internal method to load the simulation data file
    #         # if not data:
    #         #     raise ValueError(f"Data file {data_file} is empty or does not contain valid PKL.")
    #         assert data, f"Data file {data_file} is empty or does not contain valid PKL."
    #     except Exception as e:
    #         raise ValueError(f"Error loading PKL from {data_file}: {e}")        
        
    # simConfig
    try:
        exists = os.path.exists(expected_cfg_path)
        if exists and not overwrite_sim_cfg:
            print(f"\tSimConfig file {expected_cfg_path} already exists. Skipping extraction.")
            #continue
            pass
        else:
            if exists and overwrite_sim_cfg:
                print(f"\tOverwriting existing SimConfig file {expected_cfg_path}.")
                
                # account for space change after overwriting
                space_saved += os.path.getsize(expected_cfg_path)
            elif not exists:
                print(f"\tSimConfig file {expected_cfg_path} does not exist.")
                print(f"\tExtracting SimConfig from {data_file} to {expected_cfg_path}.")
            sim_cfg = extract_cfg_from_data(data_file)
            sim_cfg.save(expected_cfg_path)
            
            # assert the simConfig file was saved correctly, and exists
            if not os.path.exists(expected_cfg_path):
                raise FileNotFoundError(f"SimConfig file {expected_cfg_path} was not saved correctly.")
            
            # account for space used by the simConfig file
            space_used += os.path.getsize(expected_cfg_path)
        
        # flag success
        cfg_success = True
    except Exception as e:
        print(f"\tError processing {data_file} for simConfig: {e}")
        
        # flag failure
        cfg_success = False
        pass
    
    # netParams
    try:
        expected_net_params_path = data_file.replace("_data.json", "_netParams.json")
        exists = os.path.exists(expected_net_params_path)
        if exists and not overwrite_net_params:
            print(f"\tNetParams file {expected_net_params_path} already exists. Skipping extraction.")
            #continue
            pass
        else:
            if exists and overwrite_net_params:
                print(f"\tOverwriting existing NetParams file {expected_net_params_path}.")
                
                # account for space change after overwriting
                space_saved += os.path.getsize(expected_net_params_path)
            elif not exists:
                print(f"\tNetParams file {expected_net_params_path} does not exist.")
                print(f"\tExtracting NetParams from {data_file} to {expected_net_params_path}.")
            net_params = load_net_params_from_data(data_file)
            net_params.save(expected_net_params_path)
            
            # assert the netParams file was saved correctly, and exists
            assert os.path.exists(expected_net_params_path), f"NetParams file {expected_net_params_path} was not saved correctly."
            
            # account for space used by the netParams file
            space_used += os.path.getsize(expected_net_params_path)
        
        #flag success    
        net_params_success = True
    except Exception as e:
        print(f"\tError processing {data_file} for netParams: {e}")
        
        # flag failure
        net_params_success = False
        pass
    
    # If both simConfig and netParams were successfully extracted or validated, delete the original data file
    if cfg_success and net_params_success:
        try:
            print(f"\tDeleting original data file {data_file}.")
            
            # account for space saved by deleting the data file
            space_saved += os.path.getsize(data_file)
            
            os.remove(data_file)
        except Exception as e:
            print(f"\tError deleting {data_file}: {e}")
            pass
        print(f"Successfully extracted simConfig and netParams from {data_file}.")
    else:
        print(f"Skipping deletion of {data_file} due to previous errors in extraction.")
        
    # Return the space used and saved
    return space_used, space_saved
        
def load_net_params_from_data(sim_data_path: str) -> specs.NetParams:
    """
    Load the network parameters from a JSON or PKL simulation data file.
    
    Args:
        sim_data_path (str): Path to the simulation data file, which must end with '_data.json' or '_data.pkl'.
        
    Returns:
        specs.NetParams: The loaded network parameters.
        
    Raises:
        ValueError: If the provided path does not end with '_data.json' or '_data.pkl'.
    """
    if not (sim_data_path.endswith('_data.json') or sim_data_path.endswith('_data.pkl')):
        raise ValueError(f"Expected a file ending with '_data.json' or '_data.pkl', got '{sim_data_path}'.")
    
    # Load the simulation configuration from obj
    try:
        net_params = sim.loadNetParams(sim_data_path, setLoaded=False)
    except Exception as e:
        print(f"Error loading netParams from {sim_data_path}: {e}")
        raise
    if net_params is None:
        raise ValueError(f"Network parameters could not be loaded from the provided data path: {sim_data_path}.")
    
    return net_params

def extract_cfg_from_data(sim_data_path: str) -> specs.SimConfig:
    """
    Extract the simulation configuration from a JSON or PKL simulation data file.
    
    Args:
        sim_data_path (str): Path to the simulation data file, which must end with '_data.json' or '_data.pkl'.
        
    Returns:
        specs.SimConfig: The extracted simulation configuration.
        
    Raises:
        ValueError: If the provided path does not end with '_data.json' or '_data.pkl'.
    """
    if not (sim_data_path.endswith('_data.json') or sim_data_path.endswith('_data.pkl')):
        #raise ValueError(f"Expected a file ending with '_data.json', got '{sim_data_path}'.")
        raise ValueError(f"Expected a file ending with '_data.json' or '_data.pkl', got '{sim_data_path}'.")
    
    # Derive the configuration file path from the data path
    try:
        sim.clearAll()  # Clear any previous simulation data
        warnings.warn("Clearing all previous simulation data before loading new configuration.", UserWarning)
    except: pass
    
    # Load the simulation configuration from obj
    sim.load(sim_data_path)
    #data = sim.load._loadFile(sim_data_path)  # Use the internal method to load the simulation data file, this doesnt work
    #data = sim.loadSimData(sim_data_path)  # Use the internal method to load the simulation data file
    #sim.loadSimCfg(sim_data_path, data=data, setLoaded=False)  # Load the simulation configuration
    
    
    # Check if sim.cfg is available and convert it to a SimConfig object
    if hasattr(sim, 'cfg') and sim.cfg is not None:
        sim_cfg = specs.SimConfig(sim.cfg.todict())  # Convert sim.cfg to a SimConfig object
    elif hasattr(sim, 'cfg') and sim.cfg is None:
        raise ValueError(f"Simulation configuration is None for the provided data path: {sim_data_path}.")
    else:
        raise ValueError(f"Simulation configuration could not be loaded from the provided data path: {sim_data_path}.")
    sim.clearAll()  # Clear the simulation to free up memory
    
    return sim_cfg

def get_sim_json_paths(target_dir: str, recursive: bool = True) -> list:
    """
    Get all simulation data paths ending with '_data.json' in the target directory.
    
    Parameters:
        target_dir (str): The directory to search for simulation data files.
        recursive (bool): Whether to search recursively in subdirectories.
        
    Returns:
        list: A list of paths to simulation data files.
    """
    if recursive:
        pattern = os.path.join(target_dir, '**', '*_data.json')
    else:
        pattern = os.path.join(target_dir, '*_data.json')
    
    return glob.glob(pattern, recursive=recursive)

def get_sim_pkl_paths(target_dir: str, recursive: bool = True) -> list:
    """
    Get all simulation data paths ending with '_data.pkl' in the target directory.
    
    Parameters:
        target_dir (str): The directory to search for simulation data files.
        recursive (bool): Whether to search recursively in subdirectories.
        
    Returns:
        list: A list of paths to simulation data files.
    """
    if recursive:
        pattern = os.path.join(target_dir, '**', '*_data.pkl')
    else:
        pattern = os.path.join(target_dir, '*_data.pkl')
    
    return glob.glob(pattern, recursive=recursive)

def mutate_sim_and_net_params(netParams, simConfig, param, set_value=None, mult_value=None, add_value=None, evol_params=None, verbose=False):
    """
    Mutate a parameter in both netParams and simConfig by setting, multiplying, or adding a value based on sim_cfg param names.
    
    Parameters:
        netParams (specs.NetParams): The network parameters object.
        simConfig (specs.SimConfig): The simulation configuration object.
        param (str): The parameter to mutate.
        set_value (float, optional): Value to set the parameter to.
        mult_value (float, optional): Value to multiply the parameter by.
        add_value (float, optional): Value to add to the parameter.
        evol_params (dict, optional): Evolutionary parameters for bounds checking.
        
    Returns:
        None
    """
    
    print(f"Mutating parameter '{param}' in both netParams and simConfig...")
    before_value = get_attr_sim_cfg(simConfig, param)
    mutate_net_params(netParams, simConfig, param, set_value, mult_value, add_value, evol_params, verbose)
    mutate_sim_cfg(simConfig, param, set_value, mult_value, add_value)
    after_value = get_attr_sim_cfg(simConfig, param)
    print(f"\tMutated parameter '{param}': {before_value} -> {after_value}")
    
    return netParams, simConfig

def mutate_net_params(netParams, simConfig, param, set_value=None, mult_value=None, add_value=None, evol_params=None, verbose=False):
    """
    Mutate a parameter in the netParams object by setting, multiplying, or adding a value.
    
    Parameters:
        netParams (specs.NetParams): The network parameters object.
        param (str): The parameter to mutate.
        set_value (float, optional): Value to set the parameter to.
        mult_value (float, optional): Value to multiply the parameter by.
        add_value (float, optional): Value to add to the parameter.
        
    Returns:
        None
    """
    
    #assert multiple strategies are not used at the same time
    if (set_value is not None) + (mult_value is not None) + (add_value is not None) > 1:
        raise ValueError("Only one of set_value, mult_value, or add_value can be specified at a time.")    
    
    # if not hasattr(netParams, param):
    #     raise AttributeError(f"NetParams does not have attribute '{param}'.")
    
    if not hasattr(simConfig, param):
        raise AttributeError(f"simConfig does not have attribute '{param}'.")
    
    if evol_params is not None:
        if len(evol_params[param]) == 2:
            lower_bound, upper_bound = evol_params[param]
        else:
            warnings.warn(f"Parameter '{param}' in evol_params does not have a valid range. Using default bounds.")
            lower_bound, upper_bound = -np.inf, np.inf
    
    #map cfg to netParams
    current_cfg_value = get_attr_sim_cfg(simConfig, param)
    if current_cfg_value is None:
        raise ValueError(f"Parameter '{param}' is not set in netParams. Cannot multiply by {mult_value}.")
    
    #debug
    # if 'propVelocity' in param:
    #     print(f"Current cfg value for '{param}': {current_cfg_value}")
    # if 'E_diam_stdev' in param:
    #     print(f"Current cfg value for '{param}': {current_cfg_value}")
    if 'probLengthConst' in param:
        print(f"Current cfg value for '{param}': {current_cfg_value}")
    
    mapped_paths = map_cfg_to_netparams({param: current_cfg_value}, netParams)
    
    # get mapped values for checking
    values = []
    for path in mapped_paths[param]:
        if not isinstance(path, str):
            raise ValueError(f"Mapped path for parameter '{param}' is not a string: {path}.")
        value = getNestedParam(netParams, path)
        if isinstance(value, str):
            # if the value is a string, we assume it is a hoc string and we need to check if it contains the current cfg value
            assert str(current_cfg_value) in value, f"Current value '{value}' at path '{path}' does not match cfg value '{current_cfg_value}'."
            #value = simConfig[param]  # use the cfg value as the current value for multiplication
            value = current_cfg_value # use the cfg value as the current value for multiplication
            assert isinstance(value, (int, float)), f"Current value '{value}' at path '{path}' is not a number. Cannot multiply by {mult_value}."
        if value is None:
            raise ValueError(f"Parameter '{param}' has no value at path '{path}'. Cannot multiply by {mult_value}.")
        values.append(value)
        # get the idx of values sorted from least to greatest
        #values_idx = np.argsort(values)
        
    # flag if all values are equal or not
    all_equal = all(value == values[0] for value in values)
    
    mean_labels = ['mean', 'gnabar', 'gkbar']
    if not all_equal:
        if set_value is not None:
            # #if 'mean' in param:
            #     new_mu = set_value
            #     new_values = _mutate_mean_keep_stdev(values, current_cfg_value, new_mu, lower_bound=lower_bound, upper_bound=upper_bound)
            if 'std' in param:
                new_sigma = set_value
                new_values = _mutate_stdev_keep_mean(values, current_cfg_value, new_sigma, lower_bound=lower_bound, upper_bound=upper_bound)
            elif 'mean' in param or any(label in param for label in mean_labels):
                new_mu = set_value
                new_values = _mutate_mean_keep_stdev(values, current_cfg_value, new_mu, lower_bound=lower_bound, upper_bound=upper_bound)
            else:
                print(f'Unhandled distribution type for parameter {param}.')
                raise ValueError(f"Parameter '{param}' has different values across mapped paths and does not contain 'mean' or 'stdev': {values}.")
            value_idx = 0 # index for assigning the new values used later
        else:
            raise ImplementationError(f"Haven't implemented a way to handle different values across mapped paths for parameter '{param}' without set_value. Please provide a set_value or ensure all mapped paths have the same value.")

    # loop through all mapped paths for the parameter, updating each one
    for path in mapped_paths[param]:
        
        # get and validate the current value
        current_value = getNestedParam(netParams, path)
        if current_value is None:
            raise ValueError(f"Parameter '{param}' has no value at path '{path}'. Cannot multiply by {mult_value}.")
        if isinstance(current_value, str): # handle hoc strings
            assert str(current_cfg_value) in current_value, f"Current value '{current_value}' at path '{path}' does not match cfg value '{cfg_value}'."
            current_str = current_value
            #current_value = simConfig[param]  # use the cfg value as the current value for multiplication
            current_value = current_cfg_value # use the cfg value as the current value for multiplication
            assert isinstance(current_value, (int, float)), f"Current value '{current_str}' at path '{path}' is not a number. Cannot multiply by {mult_value}."
            current_value_type = str
        elif isinstance(current_value, (int, float, np.number)):
            current_value_type = type(current_value)
        else:
            raise TypeError(f"Unsupported type for parameter '{param}' at path '{path}': {type(current_value)}. Expected int, float, or str.")
        
        
        if set_value is not None and all_equal:
            new_value = set_value
        elif set_value is not None and not all_equal:
            #warnings.warn(f"Setting '{param}' to {set_value} at path '{path}' but not all mapped paths have the same value. This may lead to inconsistent behavior.", UserWarning)
            # #new_value = set_value
            # if 'mean' in param:
            #     #new_value = set_value
            #     _new_values
            #pass
            new_value = new_values[value_idx]
            value_idx += 1  # increment the index for the next value
        elif mult_value is not None:
            new_value = current_value * mult_value
        elif add_value is not None:
            new_value = current_value + add_value

        # check bounds if specified
        if all_equal: # doesnt really apply to individual values with mean and stdev I guess..
            if lower_bound is not None and new_value < lower_bound:
                warnings.warn(f"Setting '{param}' to {new_value} at path '{path}' is below the lower bound {lower_bound}.", UserWarning)
            if upper_bound is not None and new_value > upper_bound:
                warnings.warn(f"Setting '{param}' to {new_value} at path '{path}' is above the upper bound {upper_bound}.", UserWarning)
            
        # set the new value in netParams
        if current_value_type is str:
            updated_func = current_str.replace(str(current_cfg_value), str(new_value))
            success = setNestedParam(netParams, path, updated_func)
        else:
            success = setNestedParam(netParams, path, new_value)
            
        if success and verbose:
            print(f"\tSet '{param}' to {new_value} at path '{path}'.")

def mutate_sim_cfg(sim_cfg, param, set_value=None, mult_value=None, add_value=None):
    """
    Mutate a parameter in the SimConfig object by setting, multiplying, or adding a value.
    
    Parameters:
        sim_cfg (specs.SimConfig): The simulation configuration object.
        param (str): The parameter to mutate.
        set_value (float, optional): Value to set the parameter to.
        mult_value (float, optional): Value to multiply the parameter by.
        add_value (float, optional): Value to add to the parameter.
        
    Returns:
        None
    """
    
    #assert multiple strategies are not used at the same time
    if (set_value is not None) + (mult_value is not None) + (add_value is not None) > 1:
        raise ValueError("Only one of set_value, mult_value, or add_value can be specified at a time.")    
    
    if not hasattr(sim_cfg, param):
        raise AttributeError(f"SimConfig does not have attribute '{param}'.")

    if set_value is not None:
        #setattr(sim_cfg, param, set_value)
        set_attr_sim_cfg(sim_cfg, param, set_value)
    elif mult_value is not None:
        #current_value = getattr(sim_cfg, param)
        current_value = get_attr_sim_cfg(sim_cfg, param)
        set_attr_sim_cfg(sim_cfg, param, current_value * mult_value)
        #setattr(sim_cfg, param, current_value * mult_value)
    elif add_value is not None:
        #current_value = getattr(sim_cfg, param)
        current_value = get_attr_sim_cfg(sim_cfg, param)
        set_attr_sim_cfg(sim_cfg, param, current_value + add_value)
        #setattr(sim_cfg, param, current_value + add_value)

def _validate_path_element_count(paths):
    """
    Validate that all paths have the same number of elements.
    
    Parameters:
        paths (list): List of paths to validate.
        
    Raises:
        ValueError: If paths do not have the same number of elements.
    """
    if not paths:
        return  # No paths to validate
    
    # filter out None paths
    path_lengths = [len(path.split('.')) for path in paths]
    
    # check that they're all the same length
    if len(set(path_lengths)) > 1:
        warnings.warn(f"Paths for parameter '{param}' have inconsistent lengths: {path_lengths}. This may indicate a mapping issue.")

def _validate_type_consistency(simConfig, param, paths, netParams):
    """
    Validate that the type of the parameter in simConfig matches the type
    of the parameter in netParams at the specified paths, allowing any
    real-number types to match each other.
    """
    values_match = []
    target_val = simConfig[param]
    
    for path in paths:
        actual_val = getNestedParam(netParams, path)
        
        # exact‐type match
        if isinstance(actual_val, type(target_val)):
            values_match.append(True)
        
        # any two real‐number types (ints, floats, numpy scalars, Decimals, etc.)
        elif isinstance(actual_val, numbers.Number) and isinstance(target_val, numbers.Number):
            values_match.append(True)
        
        else:
            values_match.append(False)
    
    if not any(values_match):
        # report the last-seen actual_val’s type just to help you debug
        warnings.warn(
            f"No valid paths for '{param}': "
            f"simConfig has {type(target_val).__name__}, "
            f"netParams values were {[type(getNestedParam(netParams, p)).__name__ for p in paths]}."
        )
    
    return values_match

def getNestedParam(netParams, mapped_path):
    try:
        if '.' in mapped_path: 
            mapped_path = mapped_path.split('.')
        if isinstance(mapped_path, list ) or isinstance(mapped_path, tuple):
            container = netParams
            for ip in range(len(mapped_path) - 1):
                if hasattr(container, mapped_path[ip]):
                    container = getattr(container, mapped_path[ip])
                else:
                    container = container[mapped_path[ip]]
            return container[mapped_path[-1]]
    except Exception as e:
        print(f"Error getting nested parameter '{mapped_path}': {e}")
        traceback.print_exc()
        return None

def setNestedParam(netParams, mapped_path, value):
    '''
    Wrapper around netParams.setNestedParam to set a nested parameter value.
    '''
    try:
        netParams.setNestedParam(mapped_path, value)
        return True
    except Exception as e:
        print(f"Error setting nested parameter '{mapped_path}' to value '{value}': {e}")
        traceback.print_exc()
        return False

def _find_value_in_netparams(value, netParams, current_path=""):
    """
    Recursively search for a numeric or string value in netParams and return matching paths.
    """
    stack = [(netParams, current_path)]
    matches = []
    while stack:
        #print(f"Stack size: {len(stack)}")  # Debugging output
        obj, path = stack.pop()
        try:
            #print(f"Processing object at path: {path}")  # Debugging output
            if isinstance(obj, dict):
                #print(f"Object is a dict at path: {path}")  # Debugging output
                for key, val in obj.items():
                    #print(f"Checking key: {key}, value: {val}")
                    new_path = f"{path}.{key}" if path else key
                    if isinstance(val, (int, float)) and val == value:
                        matches.append(new_path)
                    elif isinstance(val, str) and str(value) in val:
                        matches.append(new_path)
                    elif isinstance(val, (dict, list)):
                        stack.append((val, new_path))
                    #print(val, type(val), new_path)  # Debugging output
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    new_path = f"{path}[{idx}]"
                    if isinstance(item, (int, float)) and item == value:
                        matches.append(new_path)
                    elif isinstance(item, str) and str(value) in item:
                        matches.append(new_path)
                    elif isinstance(item, (dict, list)):
                        stack.append((item, new_path))
                    #print(item, type(item), new_path)  # Debugging output
            else:
                print(f"Unhandled type at path {path}: {type(obj)}")
                raise ValueError(f"Unhandled type: {type(obj)} at path {path}")
        except Exception:
            print(f"Error processing object at path: {path}")
            traceback.print_exc()
            continue
    return matches

def _find_name_in_netparams(name, netParams, current_path=""):
    """
    Recursively search for a key or list element equal to `name` in netParams and return matching paths.
    """
    
    # search for name or elements in netParams
    stack = [(netParams, current_path)]
    matches = []
    while stack:
        obj, path = stack.pop()
        if isinstance(obj, dict):
            for key, val in obj.items():
                new_path = f"{path}.{key}" if path else key
                if key == name:
                    matches.append(new_path)
                elif isinstance(val, (dict, list)):
                    stack.append((val, new_path))
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                new_path = f"{path}[{idx}]"
                if item == name:
                    matches.append(new_path)
                elif isinstance(item, (dict, list)):
                    stack.append((item, new_path))
    return matches

def _find_elements_in_netparams(elements, netParams, current_path=""):
    """
    Recursively search for a list of elements in netParams and return matching paths.
    
    Parameters:
        elements (list): List of elements to search for.
        netParams (object): Network parameters container.
        current_path (str): Current path in the recursive search.
        
    Returns:
        list: A list of paths to the matching elements.
    """
    stack = [(netParams, current_path)]
    matches = []
    
    while stack:
        obj, path = stack.pop()
        
        if isinstance(obj, dict):
            for key, val in obj.items():
                new_path = f"{path}.{key}" if path else key
                if all(element in new_path for element in elements):
                    matches.append(new_path)
                elif isinstance(val, (dict, list)):
                    stack.append((val, new_path))
                    
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                new_path = f"{path}[{idx}]"
                if all(element in new_path for element in elements):
                    matches.append(new_path)
                elif isinstance(item, (dict, list)):
                    stack.append((item, new_path))
                    
    return matches

def _map_cfg_to_netparams_by_name(simConfig, netParams):
    """
    Map each parameter name in simConfig to its matching paths in netParams by name.

    Parameters:
        simConfig (dict): Configuration dictionary.
        netParams (object): Network parameters container.

    Returns:
        dict: Mapping from config parameter to list of netParams paths or None.
    """
    mapping = {}
    for param in simConfig:
        paths = _find_name_in_netparams(param, netParams.todict())
        
        # validate type consistency
        valid_idx = _validate_type_consistency(simConfig, param, paths, netParams)
        valid_paths = [path for path, is_valid in zip(paths, valid_idx) if is_valid]
        
        # all mapped paths will generally have the same number of elements, this can be a last sanity check, not affecting return value
        _validate_path_element_count(valid_paths)
        # assign valid paths to the mapping
        mapping[param] = valid_paths if valid_paths else None
        
        
        #mapping[param] = paths if paths else None
    return mapping

def _map_cfg_to_netparams_by_value(simConfig, netParams):
    """
    Map each parameter value in simConfig to its matching paths in netParams by value.

    Parameters:
        simConfig (dict): Configuration dictionary.
        netParams (object): Network parameters container.

    Returns:
        dict: Mapping from config parameter to list of netParams paths or None.
        
    Suggeested Usage:
        set simConfig to {param_name: param_value, ...}
        e.g. simConfig = {'gnabar_E': 0.12, 'gkbar_I': 0.04, ...}
        of course, one can pass the whole simConfig object, but the different params may require different strategies.
        This function will map each parameter to its corresponding paths in netParams based on the value.
    """
    mapping = {}
    for param, value in simConfig.items():
        paths = _find_value_in_netparams(value, netParams.todict())

        # validate type consistency
        #valid_idx = _validate_type_consistency(simConfig, param, paths, netParams)
        #valid_paths = [path for path, is_valid in zip(paths, valid_idx) if is_valid]
        # all mapped paths will generally have the same number of elements, this can be a last sanity check, not affecting return value
        
        # validate by name elements... for certain cases...
        if 'tau' in param:
            valid_paths = []
            elements = param.split('_') if '_' in param else [param]
            for element in elements:
                # lets say at least one element should be in the path...
                for path in paths:
                    if element in path:
                        valid_paths.append(path)
            # if no paths are found, we can assume that the parameter is not present in netParams
            if len(valid_paths) == 0:
                raise ValueError(f"Parameter '{param}' with value '{value}' not found in netParams. Paths: {paths}")
            
            valid_paths = list(set(valid_paths))  # remove duplicates
        else:
            # if no paths are found, we can assume that the parameter is not present in netParams
            if len(paths) == 0:
                raise ValueError(f"Parameter '{param}' with value '{value}' not found in netParams. Paths: {paths}")
            
            valid_paths = paths
            
        
        # since we're finding paths by value, we can assume that the paths are valid and types are consistent
        #valid_paths = paths
        
        _validate_path_element_count(valid_paths)
        # assign valid paths to the mapping
        mapping[param] = valid_paths if valid_paths else None
            
        # if no paths are found, assign None
        #mapping[param] = paths if paths else None
    return mapping

def _map_cfg_to_netparams_by_element(simConfig, netParams):
    """
    Map each parameter in simConfig to its matching paths in netParams by elements.
    
    Parameters:
        simConfig (dict): Configuration dictionary.
        netParams (object): Network parameters container.
        
    Returns:
        dict: Mapping from config parameter to list of netParams paths or None.
    """
    elements_to_remove = ['mean', 'stdev', 'std']  # Elements to remove from the parameter names for matching
    
    mapping = {}
    for param in simConfig:
        # split param into elements if it contains underscores
        if '_' in param:
            elements = param.split('_')
        else:
            #elements = [param]
            raise ValueError(f"Parameter '{param}' does not contain underscores to split into elements. How did you get here?")
        
        # remove specific elements that are not needed for matching
        elements = [element for element in elements if element not in elements_to_remove]
        
        paths = _find_elements_in_netparams(elements, netParams.todict())
        
        #validate type consistency
        valid_idx = _validate_type_consistency(simConfig, param, paths, netParams)
        valid_paths = [path for path, is_valid in zip(paths, valid_idx) if is_valid]
        mapping[param] = valid_paths if valid_paths else None
        
        # all mapped paths will generally have the same number of elements, this can be a last sanity check, not affecting return value
        _validate_path_element_count(valid_paths)
            
    return mapping

def _determine_mapping_strategy(cfg_param):
    """
    Determine the mapping strategy based on the configuration parameter name.
    
    Parameters:
        cfg_param (str): The configuration parameter name.
    
    Returns:
        str: The mapping strategy ('by_name' or 'by_value').
    """
    # Typical case:
    strategy = 'by_value'
    
    # SPECIAL CASES: gnabar, gkbar, L, diam, Ra
    handle_by_name = ['gnabar', 'gkbar', 'L', 'diam', 'Ra']
    if any(name in cfg_param for name in handle_by_name):
        if '_' in cfg_param:
            elements = cfg_param.split('_')
            for element in elements:
                if element in handle_by_name:
                    strategy = 'by_name'
                    break
        elif any(name == cfg_param for name in handle_by_name):
            strategy = 'by_name'
            
    if strategy == 'by_name':
        # parse name into elements if it contains underscores
        if '_' in cfg_param: elements = cfg_param.split('_')
        else: elements = None
        
        if elements is not None:
            strategy = 'by_elements'
    
    return strategy

def map_cfg_to_netparams(simConfig, netParams):
    """
    Map attributes in simConfig to their corresponding locations in netParams based on values.
    
    Parameters:
        simConfig (dict): The configuration dictionary (cfg).
        netParams (object): The network parameters object.
    
    Returns:
        dict: A mapping from simConfig parameters to their paths in netParams.
    """
    # Determine the strategy based on the first parameter
    strategy = _determine_mapping_strategy(list(simConfig.keys())[0])
    
    if strategy == 'by_name':
        return _map_cfg_to_netparams_by_name(simConfig, netParams)
    elif strategy == 'by_value':
        return _map_cfg_to_netparams_by_value(simConfig, netParams)
    elif strategy == 'by_elements':
        return _map_cfg_to_netparams_by_element(simConfig, netParams)
    else:
        raise ValueError(f"Invalid mapping strategy: {strategy}")

def get_attr_sim_cfg(sim_cfg: specs.SimConfig, attr: str, default=None):
    """
    Get an attribute from a SimConfig object, returning a default value if the attribute does not exist.

    Args:
        sim_cfg (specs.SimConfig): The simulation configuration object.
        attr (str): The attribute to retrieve.
        default: The default value to return if the attribute does not exist.

    Returns:
        The value of the specified attribute or the default value if the attribute is not found.
    """
    if hasattr(sim_cfg, attr):
        return getattr(sim_cfg, attr)
    else:
        # print(f"Warning: SimConfig does not have attribute '{attr}'. Returning default value: {default}")
        warnings.warn(f"SimConfig does not have attribute '{attr}'. Returning default value: {default}", UserWarning)
    #return getattr(sim_cfg, attr, default)

def set_attr_sim_cfg(sim_cfg: specs.SimConfig, attr: str, value):
    """
    Set an attribute on a SimConfig object.

    Args:
        sim_cfg (specs.SimConfig): The simulation configuration object.
        attr (str): The attribute to set.
        value: The value to set the attribute to.

    Returns:
        None
    """
    if hasattr(sim_cfg, attr):
        setattr(sim_cfg, attr, value)
    else:
        raise AttributeError(f"SimConfig does not have attribute '{attr}'.")

def derive_sim_cfg_path_from_sim_data_path(sim_data_path: str) -> Path:
    """
    Derive the path to the simulation configuration file from the simulation data path.

    Args:
        sim_data_path (str): Path to the simulation data file (must end with '_data.pkl').

    Returns:
    
    """
    sim_data_path = Path(sim_data_path)
    
    if not sim_data_path.name.endswith('_data.pkl'):
        raise ValueError(f"Expected a file ending with '_data.pkl', got '{sim_data_path.name}'.")

    # Replace '_data.pkl' with '_cfg.json' to get the configuration file path
    cfg_path = sim_data_path.with_name(sim_data_path.name.replace('_data.pkl', '_cfg.json'))
    
    return cfg_path

def load_sim_config_from_json(cfg_path: Path, **kwargs): # -> SimConfig:
    """
    Load a SimConfig object from a JSON configuration file using netpyne's native loader.

    Args:
        cfg_path (Path): Path to the JSON configuration file.
        **kwargs: Additional keyword arguments to pass to sim.loadSimCfg.

    Returns:
        SimConfig: Loaded simulation configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if isinstance(cfg_path, str):
        cfg_path = Path(cfg_path)
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file {cfg_path} not found.")

    SimConfig = sim.loadSimCfg(str(cfg_path), setLoaded=False)
    
    if SimConfig is None:
        print("Warning: Loaded configuration is None. Trying to load as standard JSON and recreate SimConfig.")
        with open(cfg_path, 'r') as f:
            cfg_data = json.load(f)
        assert cfg_data is not None, "Configuration data is None after loading JSON."
        assert isinstance(cfg_data, dict), "Configuration data must be a dictionary for this fallback."
        SimConfig = specs.SimConfig(cfg_data)
    return SimConfig

def load_net_params_from_json(net_params_path: Path, **kwargs): # -> specs.NetParams:
    """Load network parameters from a JSON file using netpyne's native loader.
    Args:
        net_params_path (Path): Path to the JSON network parameters file.
        **kwargs: Additional keyword arguments to pass to sim.loadNetParams.
    Returns:
        specs.NetParams: Loaded network parameters.
    Raises:
        FileNotFoundError: If the network parameters file does not exist.
    """
    if isinstance(net_params_path, str):
        net_params_path = Path(net_params_path)
    
    if not net_params_path.exists():
        raise FileNotFoundError(f"Network parameters file {net_params_path} not found.")
    
    netParams = sim.loadNetParams(str(net_params_path), setLoaded=False)
    
    if netParams is None:
        print("Warning: Loaded network parameters are None. Trying to load as standard JSON and recreate NetParams.")
        with open(net_params_path, 'r') as f:
            net_params_data = json.load(f)
        assert net_params_data is not None, "Network parameters data is None after loading JSON."
        assert isinstance(net_params_data, dict), "Network parameters data must be a dictionary for this fallback."
        netParams = specs.NetParams(net_params_data)
    
    return netParams

def load_sim_config_from_data_json(sim_data_path: str, **kwargs): # -> SimConfig:
    """
    Load a SimConfig object from a simulation data path ending with '_data.json'.
    This function first attempts to load the configuration from a '_cfg.json' file next to the data.
    If that yields None, it falls back to loading the simulation directly and building a SimConfig from sim.cfg.
    Args:
        sim_data_path (str): Path to the simulation data file (must end with '_data.json').
        **kwargs: Additional keyword arguments to pass through to the loader or SimConfig constructor.
    Returns:
        SimConfig: The loaded or constructed simulation configuration.
    Raises:
        ValueError: If sim_data_path does not end with '_data.json'.
        FileNotFoundError: If the corresponding '_cfg.json' file is missing.
    """
    data_path = Path(sim_data_path)
    
    if not data_path.name.endswith('_data.json'):
        raise ValueError(f"Expected a file ending with '_data.json', got '{data_path.name}'.")

    # Derive the configuration file path
    #cfg_path = derive_sim_cfg_path_from_sim_data_path(data_path)
    cfg_path = data_path.with_name(data_path.name.replace('_data.json', '_cfg.json'))
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file {cfg_path} not found. Cannot load simulation configuration.")
    
    # Attempt to load the configuration from the JSON file
    try:
        sim_config = load_sim_config_from_json(cfg_path, **kwargs)
    except FileNotFoundError:
        print(f"Configuration file {cfg_path} not found. Falling back to loading simulation data.")
        sim_config = get_cfg_obj_from_sim_pkl(data_path, **kwargs)

    return sim_config

def get_cfg_obj_from_sim_pkl(sim_data_path: str): # -> SimConfig:
    """
    Given a simulation data path ending with '_data.pkl', load and return the corresponding SimConfig.

    This function first attempts to load the configuration from a '_cfg.json' file next to the data.
    If that yields None, it falls back to loading the simulation directly and building a SimConfig from sim.cfg.

    Args:
        sim_data_path (str): Path to the simulation data file (must end with '_data.pkl').
        **kwargs: Additional keyword arguments to pass through to the loader or SimConfig constructor.

    Returns:
        SimConfig: The loaded or constructed simulation configuration.

    Raises:
        ValueError: If sim_data_path does not end with '_data.pkl'.
        FileNotFoundError: If the corresponding '_cfg.json' file is missing.
    """
    data_path = Path(sim_data_path)
    
    if not data_path.name.endswith('_data.pkl'):
        raise ValueError(f"Expected a file ending with '_data.pkl', got '{data_path.name}'.")

    # Fallback: load simulation data and construct SimConfig
    try:
        sim.clearAll()
        # print("Warning: This method required the netpyne sim module to be cleared before loading.")
        # print(" If you had previously loaded a simulation, it will be cleared now.")
        # print these messages as more formal warnings
        warnings.warn("This method requires the netpyne sim module to be cleared before loading. "
                      "If you had previously loaded a simulation, it will be cleared now.", UserWarning)
    except Exception:
        pass

    sim.load(str(data_path))
    sim_config = specs.SimConfig(sim.cfg.todict())
    sim.clearAll()

    return sim_config

def get_netParam_obj_from_sim_pkl_path(sim_data_path: str, **kwargs): # -> netParams:
    """
    Given a simulation data path ending with '_data.pkl', load and return the corresponding network object.

    Args:
        sim_data_path (str): Path to the simulation data file (must end with '_data.pkl').
        **kwargs: Additional keyword arguments to pass through to the loader or network object constructor.

    Returns:
        Network: The loaded or constructed network object.
    """
    data_path = Path(sim_data_path)
    
    if not data_path.name.endswith('_data.pkl'):
        raise ValueError(f"Expected a file ending with '_data.pkl', got '{data_path.name}'.")

    netParams = sim.loadNetParams(str(data_path), setLoaded=False)
    if netParams is None:
        raise ValueError(f"Failed to load netParams from {data_path}. Ensure the file is a valid simulation data file.")
    
    return netParams