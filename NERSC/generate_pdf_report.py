import os
import PyPDF2
HOF = True
HOF = False

plots_dir = '/home/adamm/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/plots'
#check if exists
if not os.path.exists(plots_dir): plots_dir = '/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/plots'
if not os.path.exists(plots_dir): raise FileNotFoundError


if HOF:
    #plots_dir = '/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/plots/240503_HOF'
    #plots_dir = '/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/plots/240506_HOF'
    plots_dir = '/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/plots/240517_Run1_best_case'
    pdf_writer = PyPDF2.PdfWriter()

#job_dir = os.path.abspath(job_dir)
job_dirs = [f.path for f in os.scandir(plots_dir) if f.is_dir()]

for job_dir in job_dirs:
    
    #print(f'Creating PDF report for {job_dir}')
    
    # Create a PDF writer object
    if HOF == False: pdf_writer = PyPDF2.PdfWriter()
    
    gen_dirs = [f.path for f in os.scandir(job_dir) if f.is_dir() and 'gen' in f.name]

    #sort gendirs numerically such that, gen_9 comes before gen_10
    gen_dirs = sorted(gen_dirs, key=lambda x: int(x.split('_')[-1]))

    for gen_dir in gen_dirs:

        #print(gen_dir)
        
        #print(f'Adding {gen_dir} to the PDF')
        cand_dirs = [f.path for f in os.scandir(gen_dir) if 'cand' in f.name and '.pdf' in f.name]
        #print(f'cand_dirs: {cand_dirs}')
        cand_dirs = sorted(cand_dirs, key=lambda x: int(x.split('_')[-1].replace('.pdf','')))
        #import sys
        #sys.exit()
        #print(f'cand_dirs: {cand_dirs}')

        for cand_dir in cand_dirs:

            #print(cand_dir)        
            # # Walk through the directory and its subdirectories
            # for dirpath, dirs, files in os.walk(gen_dir):
            #     for filename in files:
            # If the file is a PDF, add it to the PDF writer
            filename = os.path.basename(cand_dir)
            dirpath = os.path.dirname(cand_dir)
            file_path = os.path.join(dirpath, filename)
            # print(f'Adding {file_path} to the PDF')

            if filename.endswith('.pdf'):
                #print(f'Adding {filename} to the PDF')
                try:
                    pdf_file_path = os.path.join(dirpath, filename)
                    pdf_file_obj = open(pdf_file_path, 'rb')
                    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
                    for page in range(len(pdf_reader.pages)):
                        pdf_writer.add_page(pdf_reader.pages[page])
                    #print(f'Added {filename} to the PDF')
                except Exception as e:
                    print(f'Error: {e}')
                    pass
    
    #save pdf will all HOFs if HOF true
    if not HOF:
        # Write the output to a new PDF file
        reports_dir = plots_dir.replace('plots', 'reports')
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        job_name = os.path.basename(job_dir)
        job_file_name = f'{job_name}_report.pdf'
        report_dir = os.path.join(reports_dir, job_file_name)
        with open(report_dir, 'wb') as out:
            pdf_writer.write(out)

#save pdf will all HOFs if HOF true
if HOF:
    reports_dir = job_dirs[0]
    #go to grandparent directory
    reports_dir = os.path.dirname(os.path.dirname(reports_dir))
    # Write the output to a new PDF file
    reports_dir = reports_dir.replace('plots', 'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    job_name = os.path.basename(os.path.dirname(job_dirs[0]))
    job_file_name = f'{job_name}_report.pdf'
    report_dir = os.path.join(reports_dir, job_file_name)
    with open(report_dir, 'wb') as out:
        pdf_writer.write(out)
