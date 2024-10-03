import os
import datetime as dt
from collections import defaultdict
import re

def get_list_of_h5_files(h5_parent_dirs, allowed_scan_types=None, **kwargs):
    if allowed_scan_types is None:
        allowed_scan_types = kwargs.get('sorting_params', {}).get('allowed_scan_types', [''])[0]
    
    h5_files = []
    for h5_parent_dir in h5_parent_dirs:
        if h5_parent_dir.endswith('.h5') and allowed_scan_types in h5_parent_dir:
            h5_files.append(h5_parent_dir)
            continue
        for root, dirs, files in os.walk(h5_parent_dir):
            for file in files:
                if file.endswith('.h5') and allowed_scan_types in root:
                    h5_files.append(os.path.join(root, file))
    return h5_files

def process_csv_to_dict(df, h5_parent_dirs, allowed_scan_types=None):
    # Initialize the dictionary to hold the structured data
    data_dict = defaultdict(dict)
    
    # Get list of h5 files from the directories
    h5_files = get_list_of_h5_files(h5_parent_dirs, allowed_scan_types)
    
    unmatched_rows = []  # List to track rows that couldn't be matched

    # Loop over each row in the DataFrame
    for _, row in df.iterrows():
        # Convert the date to YYMMDD format
        date_str = dt.datetime.strptime(row['Date'], '%m/%d/%Y').strftime('%y%m%d')
        chip_id = row['ID']
        RBS_scan_type = row['Assay']
        source = row['Neuron Source'].split(', ')
        
        # Filter relevant h5 files for this chip_id and date
        relevant_files = [f for f in h5_files if chip_id in f and date_str in f]
        
        if not relevant_files:
            # If no relevant files found, add row to unmatched list and continue
            unmatched_rows.append(row)
            continue

        matched = False  # Flag to check if we successfully matched a file
        
        for h5_file in relevant_files:
            # Extract the scan type from the path, it should be the directory name before the chip_id
            maxwell_scan_type = h5_file.split('/')[-3]
            
            if maxwell_scan_type in allowed_scan_types:
                try:
                    assert chip_id in h5_file, f"{chip_id} not in {h5_file}"  # Assert that the chip_id is in the h5_file path
                    assert date_str in h5_file, f"{date_str} not in {h5_file}" # Assert that the date_str is in the h5_file path
                except AssertionError as e:  # If the assertions fail, print the error and continue to the next file
                    print(e)
                    continue

                # Extract the run ID from the path (6-digit number in the Network folder)
                run_id_match = re.search(r'/(\d{6})/data\.raw\.h5', h5_file)
                
                # if not run_id_match:
                #     continue  # If we can't find the run_id, skip this file

                run_id = run_id_match.group(1)
                
                # Create a unique chip identifier based on the run ID (e.g., M08018_000120)
                chip_id_with_run = f"{chip_id}_{run_id}"
                
                # Insert the structured data into the dictionary
                if date_str not in data_dict:
                    data_dict[date_str] = {}
                
                data_dict[date_str][chip_id_with_run] = {
                    "path": h5_file,
                    "scan_type": maxwell_scan_type,
                    "RBS_scan_type": RBS_scan_type,
                    "source": source
                }
                matched = True
                break  # If a match is found, no need to check further

        if not matched:
            unmatched_rows.append(row)

    # If there are any unmatched rows, print them out
    if unmatched_rows:
        print("Unmatched rows:")
        for row in unmatched_rows:
            print(row)
    
    return dict(data_dict)