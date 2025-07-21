#!/usr/bin/env python3
import os
import glob
import io
import json

from PyPDF2 import PdfReader, PdfWriter, PageObject, Transformation
from reportlab.pdfgen import canvas
import subprocess
import fitz  # PyMuPDF
import re # for regex

# List all batch directories you want to process
batch_paths = [
    '/global/homes/a/adammwea/pscratch/z_simulated_data/'
    'CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-21',
    # add more batch dirs here if needed
]

from PyPDF2 import PdfReader, PageObject
import io
from reportlab.pdfgen import canvas

def annotate_and_collect(pdf_path, candidate, fit, bottom_margin=40):
    """
    Read pdf_path, create a taller page with bottom_margin,
    shift original content up, and draw candidate+fit in the margin.
    Returns a list of new PageObjects.
    """
    reader = PdfReader(pdf_path)
    annotated_pages = []

    for page in reader.pages:
        # original size
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)
        new_h = h + bottom_margin

        # 1) Create a blank page that’s taller
        new_page = PageObject.create_blank_page(width=w, height=new_h)

        # 2) shift the original page up by bottom_margin
        #    (this mutates 'page' in place—safe since we won't reuse it)
        page.add_transformation(Transformation().translate(0, bottom_margin))
        
        # 3) merge the shifted page onto our new_page, expanding if needed
        new_page.merge_page(page, expand=True)

        # 3) Draw the footer text in the bottom_margin
        packet = io.BytesIO()
        c = canvas.Canvas(packet, pagesize=(w, new_h))
        text = f"{candidate}    fit: {fit}"
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, 20, text)  # 20 points from the very bottom
        c.save()
        packet.seek(0)
        watermark = PdfReader(packet).pages[0]

        # 5) stamp the text watermark onto new_page
        new_page.merge_page(watermark)

        annotated_pages.append(new_page)

    return annotated_pages

def flatten_pdf(input_path, output_path, dpi=150):
    """
    Raster‑flatten each page of input_path into a new PDF at ~dpi,
    saving the result to output_path.
    """
    src = fitz.open(input_path)
    dst = fitz.open()  # new empty PDF

    for page in src:
        # render page to a pixmap
        zoom = dpi / 72  # PyMuPDF uses 72 dpi as its base
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # size of the original page in points
        rect = page.rect
        # create a new blank page with the same dimensions
        new_page = dst.new_page(width=rect.width, height=rect.height)

        # insert the rasterized image across the entire page
        img_bytes = pix.tobytes("png")
        new_page.insert_image(rect, stream=img_bytes)
        print(f"Flattened page {page.number + 1} of {src.page_count}")

    # save the flattened document
    dst.save(output_path)
    src.close()
    dst.close()

def merge_pdfs_in_dir(batch_dir,
                      pattern='**/*_3p.pdf',
                      output_name='batch_report.pdf'):
    """
    Find all PDFs matching pattern under batch_dir, annotate each page
    with its candidate path & fit, then write one combined PDF.
    """
    # Find all .pdf files (exclude our eventual output)
    pdf_paths = [
        p for p in glob.glob(os.path.join(batch_dir, pattern), recursive=True)
        if os.path.basename(p) != output_name
    ]
    if not pdf_paths:
        print(f"No PDFs found in {batch_dir} with pattern {pattern}")
        return
    
    # initial sort alphanumerically
    pdf_paths.sort()

    # make sure filepaths are specifically sorted by gen and cand numbers. such that 10 doesnt follow 1. 0 should be before 1. etc. 
    # pdf_test = pdf_paths[0].split('_')
    # sorted_pdf_paths = sorted(pdf_paths, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3])))
    sorted_pdfs = {}
    for pdf in pdf_paths:
        # get number after gen_ using regex
        match = re.search(r'gen_(\d+)', pdf)
        if match: gen_num = int(match.group(1))
        
        if gen_num not in sorted_pdfs:
            sorted_pdfs[gen_num] = {}
        
        # get number after cand_ using regex
        match = re.search(r'cand_(\d+)', pdf)
        if match: cand_num = int(match.group(1))
        
        if cand_num not in sorted_pdfs[gen_num]:
            sorted_pdfs[gen_num][cand_num] = [pdf]
        else:
            sorted_pdfs[gen_num][cand_num].append(pdf)       
    
    # flatten the dict into a list following sequence of gen and cand numbers
    sorted_pdf_paths = []
    for gen_num in sorted(sorted_pdfs.keys()):
        for cand_num in sorted(sorted_pdfs[gen_num].keys()):
            # add the pdfs to the list
            sorted_pdf_paths += sorted_pdfs[gen_num][cand_num]
    
    # Build list of (pdf, candidate_dir, fit)
    jobs = []
    for pdf in sorted_pdf_paths:
        parent_dir = os.path.dirname(pdf)
        # assume fitness json is named <parent>_fitness.json
        fitness_json = os.path.join(parent_dir + '_fitness.json')
        if not os.path.exists(fitness_json):
            print(f"Warning: {fitness_json} not found; skipping {pdf}")
            continue
        
        # load json
        with open(fitness_json) as f:
            data = json.load(f)
        fit = data.get('fit', 'N/A')
        
        # check if fit is smaller than 1000
        if fit>=1000:
            print(f"Warning: {fitness_json} has fit >= 1000; skipping {pdf}")
            continue
        
        # append to jobs list      
        jobs.append((pdf, parent_dir, fit))

    if not jobs:
        print(f"No valid (PDF + JSON) pairs in {batch_dir}")
        return

    # Annotate and merge
    writer = PdfWriter()
    for pdf, candidate, fit in jobs:
        print(f"Annotating {pdf}\n  -> candidate={candidate}, fit={fit}")
        pages = annotate_and_collect(pdf, candidate, fit)
        for pg in pages:
            writer.add_page(pg)

    # Write out combined PDF
    out_path = os.path.join(batch_dir, output_name)
    with open(out_path, 'wb') as out_f:
        writer.write(out_f)

    print(f"\n✅ Merged report written to: {out_path}")
    
    # now flatten it - for easier viewing
    base, ext = os.path.splitext(output_name)
    flat_name = f"{base}_flat{ext}"
    flat_path = os.path.join(batch_dir, flat_name)

    # … inside merge_pdfs_in_dir, after writing batch_report.pdf …
    in_pdf  = os.path.join(batch_dir, output_name)
    flat_pdf = os.path.join(batch_dir, output_name.replace('.pdf','_flat.pdf'))
    flatten_pdf(in_pdf, flat_pdf, dpi=150)
    print(f"Flattened PDF written to: {flat_pdf}")

if __name__ == '__main__':
    for batch in batch_paths:
        merge_pdfs_in_dir(batch)