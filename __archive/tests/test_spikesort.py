
import os
import glob
import spikeinterface.sorters as ss
#from RBS_network_models import network_analysis
from MEA_Analysis.MEAProcessingLibrary import mea_processing_library as mea
# =============================================================================
script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(script_path))
raw_data_path = (
    os.path.abspath( 
    #this should be an .h5 file
    #'../data/raw/CDKL5-E6D_T2_C1_05212024/240611/M08034/Network/000096/data.raw.h5',
    '../data/raw/CDKL5-E6D_T2_C1_05212024/240611/M06844/Network/000076/data.raw.h5'
    ))
output_dir = (
    os.path.abspath(
    '../data/CDKL5/DIV21/sorted'
    ))
# =============================================================================
def run_sorter(
    use_docker=True,
    try_load=True,
    ):
    
    #init
    print(f'raw_data_path: {raw_data_path}')
    print(f'output_dir: {output_dir}')
    
    # load recording 
    _, recordings, _, _ = mea.load_recordings(raw_data_path) # expects a string path to a .h5 file, gets dict of recordings - 1 for each well
    recording_details = mea.extract_recording_details(raw_data_path)[0]     # NOTE: this works for a list of dirs or a single dir - but treats single dir 
                                                                            # as a list of a single dir
    h5_file_path = recording_details['h5_file_path']
    runID = recording_details['runID']
    scanType = recording_details['scanType']
    chipID = recording_details['chipID']
    date = recording_details['date']
    projectName = recording_details['projectName'] 

    # spike sort recordings
    kilosort2_params = ss.Kilosort2Sorter.default_params()
    for wellid, recording in recordings.items():
        # print
        print('====================================================================================================')
        print(f'Processing well: {wellid}')
        
        for segment in recording:  #   NOTE: generally, for this project we're working with network recordings, so there's only one segment per recording
                                            #       still, in the even we use activity scans or axon tracking scans - there will be multiple segments that need to be 
                                            #       handled after spike sorting
            # define output folder
            output_folder = os.path.join(output_dir, projectName, date, chipID, scanType, runID, wellid) #same filepath structure as the raw data + wellid
            
            # try to load the kilosort2 results if they already exist
            if try_load:
                # check if the output folder already exists
                if os.path.exists(output_folder):
                    try:
                        mea.load_kilosort2_results(output_folder) # FIXME: this function may not exist right now. Need to check.
                        print(f'Kilosort2 results already exist for {wellid}. Skipping...')
                        continue
                    except:
                        print(f'Kilosort2 results failed to load for {wellid}. Running...')
                else:
                    print(f'Kilosort2 results do not exist for {wellid}. Running...')
            
            # run kilosort2
            try:
                if use_docker:
                    mea.run_kilosort2_docker_image(
                        segment, # each segment should be a maxwellrecordingextractor object
                        output_folder=output_folder,
                        docker_image="spikeinterface/kilosort2-compiled-base:latest", #FIXME: this is the wrong image... pretty sure
                        verbose=True, 
                        logger=None, 
                        params=kilosort2_params,
                        
                    )
                else:
                    mea.kilosort2_wrapper(
                        segment, # each segment should be a maxwellrecordingextractor object
                        output_folder=output_folder,
                        sorting_params=kilosort2_params,
                        verbose=True,
                    )
            except Exception as e:
                print(f'Kilosort2 failed for {wellid} with error: {e}')
                continue
                
        # print big separator so its easy to see where each well starts and ends
        print('====================================================================================================')
            
    # attempt to find the log file and rename it for the current well
    def capture_logs(output_dir, projectName, date, chipID, scanType, runID, wellid):
        log_dir = os.path.join(output_dir, 'logs')
        log_files = glob.glob(os.path.join(log_dir, 'test_spikesort.log'))
        if len(log_files) > 0:
            for log_file in log_files:
                from datetime import datetime
                now = datetime.now()
                #make folder named for yyyymmdd
                log_date = now.strftime('%Y%m%d')
                log_dir = os.path.join(log_dir, log_date)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                #rename log file 
                #new_log_file = os.path.join(log_dir, f'{projectName}_{date}_{chipID}_{scanType}_{runID}_{wellid}_spikesort.log')
                new_log_file = os.path.join(log_dir, f'{projectName}_{date}_{chipID}_{scanType}_{runID}_spikesort.log')
                os.rename(log_file, new_log_file)
    try: capture_logs(output_dir, projectName, date, chipID, scanType, runID, wellid)
    except: pass
    
    # print done
    print('done')                          
run_sorter(
    use_docker=False,   # NOTE: Default is True. Comment out this line to use docker.
                        #       If running on NERSC, you'll need to run without docker and with shifter.
                        #       see below for shifter command to run on NERSC
    #try_load = False,   # NOTE: Default is True. Comment out this line to try loading the sorted data.
    )
# =============================================================================
# bash command to run kilosort2 on NERSC
# TODO: Update shifter to include installed editable packages. RBS_network_models and MEA_Analysis.

#  to run spikesorting as needed in interactive node with gpu:
'''
salloc -A m2043_g -q interactive -C gpu -t 04:00:00 --nodes=4 --gpus=4 --image=adammwea/axonkilo_docker:v7
salloc -A m2043_g -q interactive -C gpu -t 04:00:00 --nodes=1 --gpus=1 --image=adammwea/axonkilo_docker:v7
'''
# after salloc, run the following command: # NOTE: replace path to script as needed.
'''
shifter --image=adammwea/axonkilo_docker:v7 /bin/bash
pip install -e /pscratch/sd/a/adammwea/workspace/RBS_network_models
pip install -e /pscratch/sd/a/adammwea/workspace/MEA_Analysis
python /pscratch/sd/a/adammwea/workspace/RBS_network_models/tests/test_spikesort.py \
    > /pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/sorted/logs/test_spikesort.log 2>&1
'''
# or
'''
shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_network_models/tests/test_spikesort.py
'''