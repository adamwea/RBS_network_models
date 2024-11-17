import os
import PyPDF2
#HOF = True
import argparse
import datetime

parser = argparse.ArgumentParser(description='Generate PDF reports for all candidates in a given directory')
parser.add_argument('--HOF', action='store_true', help='Generate PDF report for Hall of Fame candidates')
parser.add_argument('--HOF_dir', type=str, help='Directory containing the job directories')
args = parser.parse_args()

if args.HOF: HOF = True
else: HOF = False

plots_dir = '/home/adamm/adamm/Documents/GithubRepositories/Network_Simulations/NERSC/plots'
#check if exists
if not os.path.exists(plots_dir): plots_dir = '/pscratch/sd/a/adammwea/Network_Simulations/NERSC/plots'
if not os.path.exists(plots_dir): raise FileNotFoundError
print(f'plots_dir: {plots_dir}')


if HOF:
    if args.HOF_dir: 
        plots_dir = args.HOF_dir
        if plots_dir[0] != '/': plots_dir = f'./{plots_dir}'
        plots_dir = os.path.abspath(plots_dir)
    else:
        #get datestring
        yymmdd_str = datetime.datetime.now().strftime('%y%m%d') #get datestring
        plots_dir = './NERSC/plots'
        plots_dir = os.path.abspath(plots_dir)
        plots_dir = os.path.join(plots_dir, yymmdd_str + '_HOF')
        #check if exists
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        #plots_dir = '/pscratch/sd/a/adammwea/2DNetworkSimulations/NERSC/plots/240529_HOF'
    pdf_writer = PyPDF2.PdfWriter()

#job_dir = os.path.abspath(job_dir)
job_dirs = [f.path for f in os.scandir(plots_dir) if f.is_dir()]

for job_dir in job_dirs:
    
    #print(f'Creating PDF report for {job_dir}')
    
    # Create a PDF writer object
    print(f'job_dir: {job_dir}')
    if HOF == False: pdf_writer = PyPDF2.PdfWriter()
    #print(f'pdf_writer: {pdf_writer}')
    
    gen_dirs = [f.path for f in os.scandir(job_dir) if f.is_dir() and 'gen' in f.name]
    #print(f'gen_dirs: {gen_dirs}')

    #sort gendirs numerically such that, gen_9 comes before gen_10
    gen_dirs = sorted(gen_dirs, key=lambda x: int(x.split('_')[-1]))

    for gen_dir in gen_dirs:

        #print(gen_dir)
        
        #print(f'Adding {gen_dir} to the PDF')
        cand_dirs = [f.path for f in os.scandir(gen_dir) if 'cand' in f.name and '.pdf' in f.name and 'repaired' not in f.name]
        #print(f'cand_dirs: {cand_dirs}')
        #if 'repairs' in gen_dir: continue
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
            if '.archive' in dirpath: continue
            file_path = os.path.join(dirpath, filename)
            # print(f'Adding {file_path} to the PDF')

            # if filename.endswith('.pdf'):
            #     #print(f'Adding {filename} to the PDF')
            #     try:
            #         pdf_file_path = os.path.join(dirpath, filename)
            #         assert os.path.exists(pdf_file_path), f'{pdf_file_path} does not exist'
            #         #open the pdf in vscode
            #         #os.system(f'code {pdf_file_path}'
            #         pdf_file_obj = open(pdf_file_path, 'rb')
            #         pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            #         for page in range(len(pdf_reader.pages)):
            #             pdf_writer.add_page(pdf_reader.pages[page])
            #         #print(f'Added {filename} to the PDF')
            #     except Exception as e:
            #         print(f'Error: {e}')
            #         pass

            #from pdfrw import PdfReader, PdfWriter
            from time import sleep
            from PyPDF2 import PdfReader

            import os
            import subprocess

            def repair_pdf(input_file, output_file):
                gs_command = f"gs -o {output_file} -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress {input_file}"
                subprocess.run(gs_command, shell=True, check=True)

            if filename.endswith('.pdf'):
                try:
                    pdf_file_path = os.path.join(dirpath, filename)
                    assert os.path.exists(pdf_file_path), f'{pdf_file_path} does not exist'
                    #pdf_reader = PdfReader(pdf_file_path)
                    # Usage
                    input_file = pdf_file_path
                    output_file = pdf_file_path.replace('.pdf', '_repaired.pdf')
                    #os.system(f'evince {input_file}')
                    repair_pdf(input_file, output_file)
                    #evince pdf_file_path
                    #os.system(f'evince {output_file}')
                    with open(output_file, 'rb') as f:
                        pdf_reader = PdfReader(f)
                        pdf_writer.add_page(pdf_reader.pages[0])
                    # for i, page in enumerate(pdf_reader.pages):
                    #     try:
                    #         pdf_writer.add_page(page)
                    #     except Exception as e:
                    #         print(f"Error on page {i}: {e}")
                    
                except Exception as e:
                    print(f'Error: {e}')
                    #sleep(50) #sleep for 50 seconds
    
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
