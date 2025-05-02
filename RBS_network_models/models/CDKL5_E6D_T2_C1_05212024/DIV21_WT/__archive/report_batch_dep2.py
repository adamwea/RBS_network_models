import os
import glob
import io
import json
import re
import argparse
from multiprocessing import Pool, cpu_count
from PyPDF2 import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
import fitz  # PyMuPDF

# ----------------------------- CLI Setup -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Annotate, merge, and flatten PDFs in batch directories.")
    parser.add_argument('--parallel', action='store_true', help="Enable multiprocessing.")
    parser.add_argument('--workers', type=int, default=cpu_count(), help="Number of parallel workers to use.")
    parser.add_argument('--batches', nargs='+', required=True, help="List of batch directories to process.")
    return parser.parse_args()

# --------------------------- File Helpers ----------------------------
def find_pdf_files(batch_dir, pattern='**/*_3p.pdf', exclude_name='batch_report.pdf'):
    return sorted([
        p for p in glob.glob(os.path.join(batch_dir, pattern), recursive=True)
        if os.path.basename(p) != exclude_name
    ])

def group_and_sort_pdfs(pdf_paths):
    sorted_pdfs = {}
    for pdf in pdf_paths:
        match_gen = re.search(r'gen_(\d+)', pdf)
        match_cand = re.search(r'cand_(\d+)', pdf)
        if match_gen and match_cand:
            gen = int(match_gen.group(1))
            cand = int(match_cand.group(1))
            sorted_pdfs.setdefault(gen, {}).setdefault(cand, []).append(pdf)
    sorted_pdf_paths = []
    for gen in sorted(sorted_pdfs):
        for cand in sorted(sorted_pdfs[gen]):
            sorted_pdf_paths.extend(sorted_pdfs[gen][cand])
    return sorted_pdf_paths

# --------------------------- Annotation ------------------------------
def load_fit_info(pdf):
    parent_dir = os.path.dirname(pdf)
    fitness_json = parent_dir + '_fitness.json'
    if not os.path.exists(fitness_json):
        print(f"⚠️  Missing: {fitness_json}")
        return None
    try:
        with open(fitness_json) as f:
            fit = json.load(f).get('fit', None)
            if fit is not None and fit < 1000:
                return (pdf, parent_dir, fit)
            else:
                print(f"⚠️  Skipping {pdf} (fit={fit})")
    except Exception as e:
        print(f"⚠️  Error reading {fitness_json}: {e}")
    return None

def annotate_pdf(args):
    pdf_path, candidate, fit = args
    reader = PdfReader(pdf_path)
    annotated_pages = []
    bottom_margin = 20
    scale_factor = 0.96  # shrink page to 96% height to make room

    for page in reader.pages:
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        new_h = h + bottom_margin

        # Prepare new blank page with extra space
        writer = PdfWriter()
        writer.add_blank_page(width=w, height=new_h)
        new_page = writer.pages[0]

        # Scale and center original page content
        transform = (
            Transformation()
            .scale(1, scale_factor)
            .translate(0, bottom_margin + (h * (1 - scale_factor) / 2))
        )
        page.add_transformation(transform)
        new_page.merge_page(page)  # no expand=True

        # Draw footer text in bottom margin
        packet = io.BytesIO()
        c = canvas.Canvas(packet, pagesize=(w, new_h))
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, 20, f"{candidate}    fit: {fit}")
        c.save()
        packet.seek(0)
        watermark = PdfReader(packet).pages[0]

        new_page.merge_page(watermark)
        annotated_pages.append(new_page)

    print(f"✅ Annotated: {pdf_path}")
    return annotated_pages

# ---------------------------- Flattening -----------------------------
def flatten_pdf_task(task):
    input_path, output_path, dpi = task
    try:
        src = fitz.open(input_path)
        dst = fitz.open()

        for page in src:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            rect = page.rect
            new_page = dst.new_page(width=rect.width, height=rect.height)
            new_page.insert_image(rect, stream=pix.tobytes("png"))

        dst.save(output_path)
        src.close()
        dst.close()
        print(f"✅ Flattened: {output_path}")
    except Exception as e:
        print(f"❌ Error flattening {input_path}: {e}")

# -------------------------- Processing Loop --------------------------
def process_batch(batch_dir, parallel=False, workers=4):
    print(f"\n📁 Processing: {batch_dir}")
    pdf_paths = find_pdf_files(batch_dir)
    sorted_pdfs = group_and_sort_pdfs(pdf_paths)

    print(f"Processing {len(sorted_pdfs)} PDFs using {workers} workers...")
    with Pool(workers) if parallel else DummyContext() as pool:
        job_args = list(filter(None, pool.map(load_fit_info, sorted_pdfs)))

        if not job_args:
            print(f"⚠️  No valid PDF + JSON pairs in {batch_dir}")
            return

        pages_list = pool.map(annotate_pdf, job_args)

    writer = PdfWriter()
    for pages in pages_list:
        for pg in pages:
            writer.add_page(pg)

    output_pdf = os.path.join(batch_dir, "batch_report.pdf")
    with open(output_pdf, 'wb') as f:
        writer.write(f)
    print(f"✅ Merged: {output_pdf}")

    print("Flattening PDF...")
    flat_output = output_pdf.replace(".pdf", "_flat.pdf")
    flatten_pdf_task((output_pdf, flat_output, 1))

# --------------------- Dummy Context for Serial Fallback ---------------------
class DummyContext:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def map(self, func, iterable): return list(map(func, iterable))

# --------------------------- Main Entry Point ---------------------------
if __name__ == '__main__':
    args = parse_args()
    for batch_dir in args.batches:
        process_batch(batch_dir, parallel=args.parallel, workers=args.workers)
