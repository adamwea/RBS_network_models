import os
import PyPDF2

job_plots_dir = '/home/adamm/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/plots/240426_Run12_26AprSAFE_1x100'

# Create a PDF writer object
pdf_writer = PyPDF2.PdfWriter()

# Walk through the directory and its subdirectories
for dirpath, dirs, files in os.walk(job_plots_dir):
    for filename in files:
        # If the file is a PDF, add it to the PDF writer
        if filename.endswith('.pdf'):
            pdf_file_path = os.path.join(dirpath, filename)
            pdf_file_obj = open(pdf_file_path, 'rb')
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            for page in range(len(pdf_reader.pages)):
                pdf_writer.add_page(pdf_reader.pages[page])

# Write the output to a new PDF file
with open('combined.pdf', 'wb') as out:
    pdf_writer.write(out)
