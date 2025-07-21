''' This script is used to save space by removing unnecessary files and extracting netParams and simConfig from existing data files.
    -- Ensuring all the necessry data are available to rerun simulations, but raw data files are not taking up space.
    -- This is useful for minimizing space usage on the cluster, especially when running large simulations or batches of simulations.
    -- It is also useful for ensuring that all the necessary data are available to rerun simulations
    
    Inputs:
    - list of directories to search for data files.
    
    Outputs:
    - netParams files and/or simConfig files saved next to all _data.pkl and/or _data.json files found recursively in a target directory if they do not already exist.
    - all _data.pkl and/or _data.json files found recursively in a target directory deleted.    
'''

# Imports
from RBS_network_models.utils.netpyne_helpers import *

# these only run with debug = True, otherwise argparse is used to get the target directories.
target_dirs = [
    #"/global/homes/a/adammwea/ben-shalom_nas/simulated_data",
    "/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_FxHET/batch_runs",
    ]
        
if __name__ == "__main__":

    debug = True # set to false to use argparse args instead of hardcoded values.
    if not debug:
        import argparse
        parser = argparse.ArgumentParser(description="Save space by removing unnecessary files and extracting netParams and simConfig from existing data files.")
        parser.add_argument("--target_dirs", type=str, nargs="+", required=True, help="List of directories to search for data files.")
        parser.add_argument("--recursive", action="store_true", help="Search directories recursively.")
        
        args = parser.parse_args()
        
        #main(target_dirs=args.target_dirs, recursive=args.recursive)
        space_saver_protocol(**kwargs)
        print("Space saving script completed.")
        print("All necessary netParams and simConfig files should now be saved next to the data files, and the data files should be deleted.")
        
    else:
        # Hardcoded values for testing
        
        kwargs = {
            "target_dirs": target_dirs,
            "recursive": True,
            
            # default is false, just comment out to not overwrite existing files.
            #"overwrite_sim_cfg": True,  # set to True to overwrite existing simConfig files
            #"overwrite_net_params": True,  # set to True to overwrite existing netParams files
        }
        
        #main(**kwargs)
        space_saver_protocol(**kwargs)
        print("Space saving script completed.")
        print("All necessary netParams and simConfig files should now be saved next to the data files, and the data files should be deleted.")    
