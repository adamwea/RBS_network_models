import os
import PyPDF2

job_plots_dir = '/home/adamm/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/plots/240426_Run12_26AprSAFE_1x100'

# Create a PDF writer object
pdf_writer = PyPDF2.PdfWriter()

#job_dir = os.path.abspath(job_dir)
gen_dirs = [f.path for f in os.scandir(job_dir) if f.is_dir() and 'gen' in f.name]

#sort gendirs numerically such that, gen_9 comes before gen_10
gen_dirs = sorted(gen_dirs, key=lambda x: int(x.split('_')[-1]))

for gen_dir in gen_dirs:
    # Walk through the directory and its subdirectories
    for dirpath, dirs, files in os.walk(gen_dir):
        for filename in files:
            # If the file is a PDF, add it to the PDF writer
            if filename.endswith('.pdf'):
                try:
                    pdf_file_path = os.path.join(dirpath, filename)
                    pdf_file_obj = open(pdf_file_path, 'rb')
                    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
                    for page in range(len(pdf_reader.pages)):
                        pdf_writer.add_page(pdf_reader.pages[page])
                except Exception as e:
                    print(f'Error: {e}')
                    pass

# Write the output to a new PDF file
with open('combined.pdf', 'wb') as out:
    pdf_writer.write(out)
