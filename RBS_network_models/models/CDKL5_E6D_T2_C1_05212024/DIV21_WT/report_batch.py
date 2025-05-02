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
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.evol_params import params

# ----------------------------- CLI Setup -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Annotate, flatten, and merge PDFs in batch directories.")
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

def group_and_sort_pdfs(pdf_paths, exclude_gen=0):
    sorted_pdfs = {}
    for pdf in pdf_paths:
        match_gen = re.search(r'gen_(\d+)', pdf)
        match_cand = re.search(r'cand_(\d+)', pdf)
        if match_gen and match_cand:
            gen = int(match_gen.group(1))
            cand = int(match_cand.group(1))
            
            # exclude gen 0 - avoid reporting seeded candidates
            # if gen == exclude_gen:
            #     print(f"⚠️  Excluding gen {gen} from {pdf}")
            #     continue
            
            sorted_pdfs.setdefault(gen, {}).setdefault(cand, []).append(pdf)
    sorted_pdf_paths = []
    for gen in sorted(sorted_pdfs):
        for cand in sorted(sorted_pdfs[gen]):
            sorted_pdf_paths.extend(sorted_pdfs[gen][cand])
    return sorted_pdf_paths

def filter_top_n_candidates(job_args, top_n=10):
    """
    Given job_args [(pdf_path, candidate_path, fit_value), ...],
    keep only the top N candidates based on best (lowest) fitness,
    but restore original generation/candidate ordering afterward.
    """
    if not job_args:
        return []

    # Sort by fitness value (ascending)
    job_args_sorted_by_fit = sorted(job_args, key=lambda x: x[2])

    # Take top N
    top_jobs = job_args_sorted_by_fit[:top_n]

    # Extract gen and cand numbers from the filename to re-sort
    def get_gen_cand_key(job):
        pdf_path = job[0]
        match_gen = re.search(r'gen_(\d+)', pdf_path)
        match_cand = re.search(r'cand_(\d+)', pdf_path)
        if match_gen and match_cand:
            gen = int(match_gen.group(1))
            cand = int(match_cand.group(1))
            return (gen, cand)
        else:
            return (9999, 9999)  # put invalid entries last

    # Now re-sort the top_jobs by (gen, cand)
    #top_jobs_sorted = sorted(top_jobs, key=get_gen_cand_key)

    print(f"✅ Selected top {top_n} candidates based on fitness and sorted by gen/cand order.")
    #return top_jobs_sorted
    return top_jobs

def flag_param_violation(job_args=None):
    """
    Given job_args [(pdf_path, candidate_path, fit_value), ...],
    check for parameter violations and flag them.
    """
    if not job_args:
        return []

    print("⚠️  Checking for parameter violations...")
    #print(job_args)
    
    flagged_jobs = []
    for pdf_path, candidate_path, fit_value, cfg in job_args:
        # Check for parameter violations
        param_violations = {}
        for param_name, param_value in params.items():
            param_hi = param_value[1]
            param_lo = param_value[0]
            if param_name not in cfg:
                #param_violations[param_name] = True  # missing parameter is a violation
                pass
            else:
                if cfg[param_name] < param_lo or cfg[param_name] > param_hi:
                    param_violations[param_name] = True
                else:
                    param_violations[param_name] = False

                    
        updated_job_args = (pdf_path, candidate_path, fit_value, cfg, param_violations)
        flagged_jobs.append(updated_job_args)
            
    return flagged_jobs
        
# --------------------------- Annotation and Flattening ------------------------------
def load_job_info(pdf):
    parent_dir = os.path.dirname(pdf)
    fitness_json = parent_dir + '_fitness.json'
    cfg_json = parent_dir + '_cfg.json'
    if not os.path.exists(fitness_json):
        print(f"⚠️  Missing: {fitness_json}")
        return None
    try:
        with open(fitness_json) as f:
            fit = json.load(f).get('fit', None)
        with open(cfg_json) as f:
            cfg = json.load(f)
            cfg = cfg['simConfig']
        if fit is not None and fit < 1000:
            return (pdf, parent_dir, fit, cfg)
        else:
            print(f"⚠️  Skipping {pdf} (fit={fit})")
    except Exception as e:
        print(f"⚠️  Error reading {fitness_json}: {e}")
    return None

def annotate_and_flatten_dep(args):
    pdf_path, candidate, fit = args
    try:
        reader = PdfReader(pdf_path)
        bottom_margin = 20
        scale_factor = 0.96

        writer = PdfWriter()
        for page in reader.pages:
            w, h = float(page.mediabox.width), float(page.mediabox.height)
            new_h = h + bottom_margin

            writer.add_blank_page(width=w, height=new_h)
            new_page = writer.pages[-1]

            try:
                transform = (
                    Transformation()
                    .scale(1, scale_factor)
                    .translate(0, bottom_margin + (h * (1 - scale_factor) / 2))
                )
                page.add_transformation(transform)
                new_page.merge_page(page)
            except Exception as e:
                print(f"⚠️ Skipped page due to transformation error: {e}")
                continue

            packet = io.BytesIO()
            c = canvas.Canvas(packet, pagesize=(w, new_h))
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, 20, f"{candidate}    fit: {fit}")
            c.save()
            packet.seek(0)
            watermark = PdfReader(packet).pages[0]

            try:
                new_page.merge_page(watermark)
            except Exception as e:
                print(f"⚠️ Skipped page due to watermarking error: {e}")

        annotated_path = pdf_path.replace('.pdf', '_annotated.pdf')
        with open(annotated_path, 'wb') as f:
            writer.write(f)
        print(f"✅ Annotated: {annotated_path}")

        # Now flatten it immediately
        flat_path = annotated_path.replace('_annotated.pdf', '_annotated_flat.pdf')
        flatten_pdf_task_safe((annotated_path, flat_path, 75))

        return flat_path

    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        return None

def annotate_and_flatten(args):
    #pdf_path, candidate, fit = args
    pdf_path, candidate, fit, cfg, param_violations = args
    try:
        doc = fitz.open(pdf_path)
        new_doc = fitz.open()

        for page in doc:
            # Rasterize the original page
            mat = fitz.Matrix(1, 1)  # 72 dpi native, or scale higher if needed
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Create new page
            rect = page.rect
            new_page = new_doc.new_page(width=rect.width, height=rect.height + 40)  # 40 points bottom margin

            # Insert original page as image, shifted up by 40
            img_rect = fitz.Rect(0, 40, rect.width, rect.height + 40)
            new_page.insert_image(img_rect, stream=pix.tobytes("png"))

            # Draw footer text
            footer = f"{candidate}    fit: {fit}"
            new_page.insert_text(
                point=(40, 20),  # 20 points from bottom
                text=footer,
                fontsize=12,
                fontname="helv",
                color=(0, 0, 0)
            )
            
            # if any parameter violations, just add a red asterisk
            if any(param_violations.values()):
                new_page.insert_text(
                    point=(rect.width - 40, 50),  # 20 points from bottom
                    text="*",
                    fontsize=48,
                    fontname="helv",
                    color=(1, 0, 0)  # Red color
                )
            
            # add smalle text with param violations
            if any(param_violations.values()):
                violation_text = "Param violations: " + ", ".join([f"{k}: {v}" for k, v in param_violations.items() if v])
                new_page.insert_text(
                    point=(40, 50),  # 50 points from bottom
                    text=violation_text,
                    fontsize=8,
                    fontname="helv",
                    color=(1, 0, 0)  # Red color
                )

        annotated_flat_path = pdf_path.replace(".pdf", "_annotated_flat.pdf")
        new_doc.save(annotated_flat_path)
        doc.close()
        new_doc.close()

        print(f"✅ Annotated + Flattened: {annotated_flat_path}")
        return annotated_flat_path

    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        return None

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

def flatten_pdf_task_safe(task):
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
        job_args = list(filter(None, pool.map(load_job_info, sorted_pdfs)))
        #top_n = 150  # <-- You can make this a CLI argument too if you want
        top_n = 256
        job_args = filter_top_n_candidates(job_args, top_n=top_n)
        job_args = flag_param_violation(job_args)
        
        
        if not job_args:
            print(f"⚠️  No valid PDF + JSON pairs in {batch_dir}")
            return

        flattened_pdfs = pool.map(annotate_and_flatten, job_args)

    # Merge flattened PDFs
    print("Merging flattened PDFs...")
    
    # filter out None values
    flattened_pdfs = [pdf for pdf in flattened_pdfs if pdf is not None and os.path.exists(pdf)]

    writer = PdfWriter()
    for flat_pdf in flattened_pdfs:
        reader = PdfReader(flat_pdf)
        for page in reader.pages:
            writer.add_page(page)

    final_output = os.path.join(batch_dir, "batch_report_flat.pdf")
    with open(final_output, 'wb') as f:
        writer.write(f)
    print(f"✅ Final flattened report written to: {final_output}")

# --------------------- Dummy Context for Serial Fallback ---------------------
class DummyContext:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def map(self, func, iterable): return list(map(func, iterable))

# --------------------------- Main Entry Point ---------------------------
if __name__ == '__main__':
    #debug = True
    debug = False
    if not debug:
        # regualr use:
        args = parse_args()    
        for batch_dir in args.batches:
            process_batch(batch_dir, parallel=args.parallel, workers=args.workers)
    else:
        # debug use:
        batch_dir = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26'
        process_batch(batch_dir, parallel=False, workers=1)
            
    