import os
import shutil
import csv

sources = [
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_axon_reconstruction_output',
]

target = '/pscratch/sd/a/adammwea/workspace/zOutputs/axon_reconstruction_outputs'

included_file_types = [
    '.csv',
    # '.json',
    # '.png',
    # '.pdf',
]

dirs_to_ignore = [
    '_archive',
]

if not os.path.exists(target):
    os.makedirs(target)    
from workspace.zOutputs.mea_processing_library import extract_recording_details
# Copy files
for source in sources:
    for root, dirs, files in os.walk(source):
        for file in files:
            if any([file.endswith(file_type) for file_type in included_file_types]):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target, os.path.relpath(source_file, source))
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                switch = True
                if switch:
                    if os.path.exists(target_file):
                        print(f'{target_file} already exists')
                        continue
                shutil.copyfile(source_file, target_file)
                print(f'Copied {source_file} to {target_file}')

# Process copied CSV files and generate reference files
for root, dirs, files in os.walk(target):
    if any([dir_to_ignore in root for dir_to_ignore in dirs_to_ignore]):
        continue
    reference_data = []
    for file in files:
        if file.endswith('.csv'):
            print(f'Processing {file}')
            #project, date, chipid, scantype, runid= extract_details_from_filename(file)
            file_path = os.path.join(root, file)
            #details = extract_recording_details_from_csv_file_path(file_path)
            #project, date, chipid, scantype, runid, wellid = extract_recording_details_from_csv_file_path(file_path)[0]            
            def extract_recording_details_from_csv_file_path(h5_dirs):


                # If h5_dirs is a string, convert it to a list with a single element
                if isinstance(h5_dirs, str):
                    h5_dirs = [h5_dirs]

                #logger.info(f"Extracting recording details from h5 directories:")
                print(f"Extracting recording details from h5 directories:")
                records = []
                for h5_dir in h5_dirs:
                    #try: assert '.h5' in h5_dir, "The input is not a list of h5 directories."
                    #except: 
                    try: 
                        #h5_subdirs = extract_raw_h5_filepaths(h5_dir)
                        h5_subdirs = [h5_dir]
                        assert len(h5_subdirs) > 0, "No .h5 files found in the directory."
                        assert len(h5_subdirs) == 1, "Ambiguous file selection. Multiple .h5 files found in the directory."
                        h5_dir = h5_subdirs[0]
                    except: 
                        #logger.error("Some error occurred during the extraction of .h5 file paths."); 
                        
                        print("Some error occurred during the extraction of .h5 file paths.") 
                        continue

                    parent_dir =  os.path.dirname(h5_dir)
                    output_dir = os.path.basename(parent_dir)
                    
                    grandparent_dir = os.path.dirname(parent_dir)
                    wellid = os.path.basename(grandparent_dir)
                    
                    h5_dir = grandparent_dir #reshift things so the function works like it was before
                    
                    #start over.       
                    parent_dir = os.path.dirname(h5_dir)
                    runID = os.path.basename(parent_dir)

                    grandparent_dir = os.path.dirname(parent_dir)
                    scan_type = os.path.basename(grandparent_dir)

                    great_grandparent_dir = os.path.dirname(grandparent_dir)
                    chipID = os.path.basename(great_grandparent_dir)

                    ggg_dir = os.path.dirname(great_grandparent_dir)
                    date = os.path.basename(ggg_dir)
                    
                    gggg_dir = os.path.dirname(ggg_dir)
                    project_name = os.path.basename(gggg_dir)

                    record = {'h5_file_path': h5_dir, 
                                'projectName': project_name,
                                'date': date,
                                'chipID': chipID,
                                'scanType': scan_type, 
                                'runID': runID,
                                'wellID': wellid 
                            }
                    records.append(record)
                    
                    print(f'Details extracted: ')
                    print(f'Project Name: {project_name}')
                    print(f'Date: {date}')
                    print(f'Chip ID: {chipID}')
                    print(f'Scan Type: {scan_type}')
                    print(f'Run ID: {runID}')
                    print(f'Well ID: {wellid}')

                return records
            details = extract_recording_details_from_csv_file_path(file_path)
            project = details[0]['projectName']
            date = details[0]['date']
            chipID = details[0]['chipID']
            scan_type = details[0]['scanType']
            runid = details[0]['runID']
            wellid = details[0]['wellID']
            
            #add row to reference_data csv file
            with open(os.path.join(root, file), 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    reference_data.append({
                        'date': date,
                        'chipid': chipID,
                        'scantype': scan_type,
                        'runid': runid,
                        'wellid': wellid,
                        #'unit_id': row.get('unit_id'),
                        #'num_branches': row.get('num_branches'),
                        #'axon_length': row.get('axon_length'),
                        #'average_branches_per_unit': row.get('num_branches'),  # Example stat
                        # Add other high-level stats as needed
                    })          
                
# save reference_data to a csv file to the project directory
if reference_data:
    project_dir = os.path.join(target, project)
    assert os.path.exists(project_dir), f"Project directory {project_dir} does not exist."
    reference_file = os.path.join(project_dir, 'reference.csv')
    with open(reference_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=reference_data[0].keys())
        
        writer.writeheader()
        writer.writerows(reference_data)
    print(f'Generated reference file: {reference_file}')
    
            

            #date = details[0]['date']
            #some quick checks to ensure the date is in the correct format
            #assert date is actually a date in the format YYMMDD
    #         import re
    #         try:
    #             assert re.match(r'^\d{6}$', date), "Date is not in the format YYMMDD"
    #             #assert that MM isnt greater than 12 and DD isnt greater than 31 to avoid confusion
    #             assert int(date[2:4]) <= 12, "Month is greater than 12"
    #             assert int(date[2:4]) > 0, "Month is less than 1"
    #             assert int(date[4:6]) <= 31, "Day is greater than 31"
    #             assert int(date[4:6]) > 0, "Day is less than 1"
    #         except:
    #             print(f"Date {date} is not in the format YYMMDD")
    #             continue
    #         # chipID = details[0]['chipID']
    #         # scan_type = details[0]['scanType']
    #         # runid = details[0]['runID']
    #         # project_name = details[0]['projectName']
    #         chipid = details[0]['chipID']
    #         scantype = details[0]['scanType']
    #         if date and chipid and scantype and runid and wellid:
    #             #get template type from .csv file name. options will include: vt, dvdt, or milos in the file name
    #             if '_vt' in file:
    #                 template_type = 'vt'
    #             elif '_dvdt' in file:
    #                 template_type = 'dvdt'
    #             elif '_milos' in file:
    #                 template_type = 'milos'
    #             else:
    #                 template_type = 'unknown'
    #             with open(os.path.join(root, file), 'r') as f:
    #                 reader = csv.DictReader(f)
    #                 for row in reader:
    #                     reference_data.append({
    #                         #'date': date,
    #                         'date': details[0]['date'],
    #                         #'chipid': chipid,
    #                         'chipid': details[0]['chipID'],
    #                         #'scantype': scantype,
    #                         'scantype': details[0]['scanType'],
    #                         #'runid': runid,
    #                         'runid': details[0]['runID'],
    #                         #'wellid': wellid,
    #                         'template_type': template_type,
    #                         'unit_id': row.get('unit_id'),
    #                         'num_branches': row.get('num_branches'),
    #                         'axon_length': row.get('axon_length'),
    #                         'average_branches_per_unit': row.get('num_branches'),  # Example stat
    #                         # Add other high-level stats as needed
    #                     })
    
    # if reference_data:
    #     reference_file = os.path.join(root, 'reference.csv')
    #     with open(reference_file, 'w', newline='') as f:
    #         writer = csv.DictWriter(f, fieldnames=reference_data[0].keys())
    #         writer.writeheader()
    #         writer.writerows(reference_data)
    #     print(f'Generated reference file: {reference_file}')