import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
#from PyPDF2 import PdfMerger
from PyPDF2 import PdfWriter
from PyPDF2 import Transformation
from PyPDF2._page import PageObject  # for type hinting
from PyPDF2 import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import numpy as np
from datetime import datetime

def find_fitness_jsons(batch_paths):
    fitness_jsons = []
    #today = datetime.now().date()

    for batch_path in batch_paths:
        fitness_jsons += glob.glob(f'{batch_path}/**/*_fitness.json', recursive=True)

    json_info = []
    for fpath in fitness_jsons:
        # Check if the file was modified today
        # if datetime.fromtimestamp(os.path.getmtime(fpath)).date() != today:
        #     continue

        match = re.search(r'gen_(\d+)_cand_(\d+)', fpath)
        if match:
            gen = int(match.group(1))
            cand = int(match.group(2))
            json_info.append((gen, cand, fpath))
        else:
            print(f'❌ No match for: {fpath}')
    return sorted(json_info, key=lambda x: (x[0], x[1]))

def load_fit_file(entry):
    gen, cand, fpath = entry
    try:
        with open(fpath, 'r') as f:
            #fitness = json.load(f).get('fit', np.nan)
            fitness = json.load(f)
    except Exception as e:
        print(f'⚠️ Failed to load {fpath}: {e}')
        fitness = np.nan
    return (gen, cand, fitness)

def load_bursting_fit(entry):
    gen, cand, fpath = entry
    try:
        with open(fpath, 'r') as f:
            #fitness = json.load(f).get('fit', np.nan)
            overall_fitness = json.load(f).get('fit', np.nan)
        with open(fpath, 'r') as f:
            #burst_fitness = json.load(f)['bursting_data']['burst_metrics']
            burst_fitness = json.load(f)['mega_bursting_data']['burst_metrics']
        print(f'Loaded {fpath}')
        # try:
        #     burst_rate_fit = fitness.get('burst_rate', {}).get('fit', np.nan)
        #     ibi_fit = fitness.get('ibi', {}).get('fit', np.nan)
        #     burst_amp_fit = fitness.get('burst_amp', {}).get('fit', np.nan)
        #     burst_duration_fit = fitness.get('burst_duration', {}).get('fit', np.nan)
        #     num_units_per_burst_fit = fitness.get('num_units_per_burst', {}).get('fit', np.nan)
        #     in_burst_fr_fit = fitness.get('in_burst_fr', {}).get('fit', np.nan)
        # except Exception as e:
        #     print(f'⚠️ Failed to load bursting metrics from {fpath}: {e}')
        #     burst_rate_fit = np.nan
        #     ibi_fit = np.nan
        #     burst_amp_fit = np.nan
        #     burst_duration_fit = np.nan
        #     num_units_per_burst_fit = np.nan
        #     in_burst_fr_fit = np.nan
        
    except Exception as e:
        print(f'⚠️ Failed to load {fpath}: {e}')
        burst_fitness = np.nan
        overall_fitness = np.nan
    return (gen, cand, burst_fitness, fpath, overall_fitness)

def filter_top_n_candidates(burst_fits, top_n=256):
    # filter out the top
    fits = []
    for entry in burst_fits:
        #try: fit = entry[2].get('fit', np.nan)
        try: fit = entry[4]
        except: fit = np.nan
        fits.append(fit)
    
    sorted_indices = np.argsort(fits)
    sorted_scores = [fits[i] for i in sorted_indices]
    sorted_fits = [burst_fits[i] for i in sorted_indices]
    
    return sorted_fits[:top_n]

def get_rate_sorted(burst_fits):
    # use arg sort logic, get the indices of the sorted array
    burst_rate_fits = []
    for entry in burst_fits:
        try: burst_rate_fit = entry[2].get('burst_rate', {}).get('fit', np.nan)
        except: burst_rate_fit = np.nan
        burst_rate_fits.append(burst_rate_fit)
        
    sorted_indices = np.argsort(burst_rate_fits)
    sorted_burst_fits = [burst_fits[i] for i in sorted_indices]
    
    return sorted_burst_fits

def get_ibi_sorted(burst_fits):
    # use arg sort logic, get the indices of the sorted array
    ibi_fits = []
    for entry in burst_fits:
        try: ibi_fit = entry[2].get('ibi', {}).get('fit', np.nan)
        except: ibi_fit = np.nan
        ibi_fits.append(ibi_fit)
        
    sorted_indices = np.argsort(ibi_fits)
    sorted_burst_fits = [burst_fits[i] for i in sorted_indices]
    
    return sorted_burst_fits

def get_burst_amp_sorted(burst_fits):
    # use arg sort logic, get the indices of the sorted array
    burst_amp_fits = []
    for entry in burst_fits:
        try: burst_amp_fit = entry[2].get('burst_amp', {}).get('fit', np.nan)
        except: burst_amp_fit = np.nan
        burst_amp_fits.append(burst_amp_fit)
        
    sorted_indices = np.argsort(burst_amp_fits)
    sorted_burst_fits = [burst_fits[i] for i in sorted_indices]
    
    return sorted_burst_fits

def get_burst_duration_sorted(burst_fits):
    # use arg sort logic, get the indices of the sorted array
    burst_duration_fits = []
    for entry in burst_fits:
        try: burst_duration_fit = entry[2].get('burst_duration', {}).get('fit', np.nan)
        except: burst_duration_fit = np.nan
        burst_duration_fits.append(burst_duration_fit)
        
    sorted_indices = np.argsort(burst_duration_fits)
    sorted_burst_fits = [burst_fits[i] for i in sorted_indices]
    
    return sorted_burst_fits

def get_num_units_per_burst_sorted(burst_fits):
    # use arg sort logic, get the indices of the sorted array
    num_units_per_burst_fits = []
    for entry in burst_fits:
        try: num_units_per_burst_fit = entry[2].get('num_units_per_burst', {}).get('fit', np.nan)
        except: num_units_per_burst_fit = np.nan
        num_units_per_burst_fits.append(num_units_per_burst_fit)
        
    sorted_indices = np.argsort(num_units_per_burst_fits)
    sorted_burst_fits = [burst_fits[i] for i in sorted_indices]
    
    return sorted_burst_fits

def get_in_burst_fr_sorted(burst_fits):
    # use arg sort logic, get the indices of the sorted array
    in_burst_fr_fits = []
    for entry in burst_fits:
        try: in_burst_fr_fit = entry[2].get('in_burst_fr', {}).get('fit', np.nan)
        except: in_burst_fr_fit = np.nan
        in_burst_fr_fits.append(in_burst_fr_fit)
        
    sorted_indices = np.argsort(in_burst_fr_fits)
    sorted_burst_fits = [burst_fits[i] for i in sorted_indices]
    
    return sorted_burst_fits

def scale_pdf_page(page: PageObject, scale_factor: float = 0.8) -> PageObject:
    """
    Scale down the contents of a PDF page to make room for annotations.
    
    Args:
        page (PageObject): A single page from a PDF.
        scale_factor (float): How much to scale down (0 < scale <= 1).
    
    Returns:
        PageObject: The modified page.
    """
    original_width = float(page.mediabox.width)
    original_height = float(page.mediabox.height)

    # Calculate translation to center the scaled content
    translate_x = (original_width * (1 - scale_factor)) / 2
    translate_y = (original_height * (1 - scale_factor)) / 2

    # Apply the transformation: scale then translate
    transformation = (
        Transformation()
        .scale(scale_factor, scale_factor)
        .translate(tx=translate_x, ty=translate_y)
    )
    page.add_transformation(transformation)

    return page

def scale_pdf_page_top_left(page: PageObject, scale_factor: float = 0.8) -> PageObject:
    """
    Scale down the contents of a PDF page toward the top-left corner to make space for annotations.
    
    Args:
        page (PageObject): A single page from a PDF.
        scale_factor (float): How much to scale down (0 < scale <= 1).
    
    Returns:
        PageObject: The modified page.
    """
    original_width = float(page.mediabox.width)
    original_height = float(page.mediabox.height)

    # Calculate translation to anchor top-left
    translate_x = 0  # Keep left aligned
    translate_y = original_height * (1 - scale_factor)  # Move content downward

    # Apply the transformation: scale then translate
    transformation = (
        Transformation()
        .scale(scale_factor, scale_factor)
        .translate(tx=translate_x, ty=translate_y)
    )
    page.add_transformation(transformation)

    return page

def extract_fit_text_lines(d, md, indent=0, fit_width=50):
    """
    Recursively extracts lines of 'key: fit_value' from nested dicts,
    with aligned mvalue columns.
    """
    lines = []
    for key, value in d.items():
        if 'by_unit' in key or 'unit_metrics' in key:
            continue
        if key in ['std', 'cov', 'median']:
            continue

        try:
            mvalue = md.get(key, None)
        except Exception:
            mvalue = None

        if isinstance(value, dict):
            if 'fit' in value:
                label = "    " * indent + f"{key}: {value['fit']}"
                label = label.ljust(fit_width)  # pad to fixed width
                if isinstance(mvalue, dict):
                    lines.append(label)
                elif isinstance(mvalue, (float, int, np.int64)):
                    lines.append(f"{label} ({mvalue})")
                else:
                    lines.append(label)
            # Recurse deeper
            lines.extend(extract_fit_text_lines(value, mvalue, indent + 1, fit_width=fit_width))
    return lines

def add_fitness_annotation_dep(page: PageObject, title: str, fitness_json_path: str, metrics_npy_path: str, scale_factor: float = 0.8) -> PageObject:
    """
    Adds fit info as an annotation layer to the bottom of a PDF page,
    positioning the text just to the right of the scaled content area.
    
    Args:
        page (PageObject): The target PDF page (already scaled).
        fitness_json_path (str): Path to the JSON file with fitness data.
        scale_factor (float): The same scale used to shrink the page content.
    
    Returns:
        PageObject: Annotated page.
    """
    # Load the fitness data
    with open(fitness_json_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return page
    
    # Load the metrics data
    try:
        metrics_data = np.load(metrics_npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"⚠️ Failed to load metrics data from {metrics_npy_path}: {e}")
        metrics_data = {}

    #lines = extract_fit_text_lines(data, metrics_data)
    lines = extract_fit_rows(data, metrics_data)

    # Get page dimensions
    page_width = float(page.mediabox.width)
    page_height = float(page.mediabox.height)

    # Calculate dynamic left margin based on scaled content size
    content_right_edge = page_width * scale_factor
    annotation_padding = 20  # space between content and text
    left_margin = content_right_edge + annotation_padding

    # Create in-memory overlay PDF
    packet = BytesIO()
    c = canvas.Canvas(packet, pagesize=(page_width, page_height))

    # Initialize font size
    font_size = 10
    c.setFont("Helvetica", font_size)
    
    while True:
        # Calculate starting position and check if all lines fit
        start_y = page_height - 10
        
        # Draw title
        c.setFont("Helvetica-Bold", font_size + 2)
        y_after_title = start_y - 5
        
        fits = True
        for i, line in enumerate(lines):
            y = y_after_title - i * (font_size + 2)  # Line height includes padding
            if y < 50:  # Bottom margin
                fits = False
                break
        
        if fits:
            break  # Exit loop if all lines fit
        elif font_size <= 1:
            raise ValueError("⚠️ Error: Unable to fit all lines on the page even with minimum font size.")
        else:
            # Reduce font size and try again
            font_size -= 1
            c.setFont("Helvetica", font_size)
    
    # Draw the lines with the final font size
    for i, line in enumerate(lines):
        y = start_y - i * (font_size + 2)
        
        # Draw the title
        if i == 0:
            c.setFont("Helvetica-Bold", font_size + 2)
            c.drawString(left_margin, y, title)
            start_y = y - 5  # Adjust start_y for the next line
        else:
            c.setFont("Helvetica", font_size)
        
            c.drawString(left_margin, y, line)
            
            # draw horizontal line under each line of text
            c.line(left_margin, y - 2, page_width - 50, y - 2)
        
    # #start_y = 50
    # # start at the top of the page
    # start_y = page_height - 50
    # line_height = 12
    # for i, line in enumerate(lines):
    #     #y = start_y + i * line_height
    #     y = start_y - i * line_height
    #     c.drawString(left_margin, y, line)

    c.save()

    # Overlay annotation onto the page
    packet.seek(0)
    overlay_pdf = PdfReader(packet)
    overlay_page = overlay_pdf.pages[0]

    page.merge_page(overlay_page)
    return page

def add_fitness_annotation(
    page: PageObject,
    title: str,
    fitness_json_path: str,
    metrics_npy_path: str,
    targets_npy_path: str,
    scale_factor: float = 0.8
    ) -> PageObject:
    """
    Adds fitness info as a two-column annotation (fit, value),
    scaled to fit the page and aligned next to the scaled content.
    """
    # Load fitness JSON
    with open(fitness_json_path, 'r') as f:
        data = json.load(f)
    try:
        metrics_data = np.load(metrics_npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"⚠️ Failed to load metrics data from {metrics_npy_path}: {e}")
        metrics_data = {}
    try:
        targets_data = np.load(targets_npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"⚠️ Failed to load targets data from {targets_npy_path}: {e}")
        targets_data = {}

    if not isinstance(data, dict):
        return page

    rows = extract_fit_rows(data, metrics_data, targets_data)

    # Page size
    page_width = float(page.mediabox.width)
    page_height = float(page.mediabox.height)

    # Layout positions
    content_right_edge = page_width * scale_factor
    padding = 20
    left_col_x = content_right_edge + padding
    middle_col_x = left_col_x + 220
    right_col_x = middle_col_x + 220
    top_y = page_height - 30
    min_y = 50

    # Try shrinking font size until all rows fit
    font_name = "Courier"
    font_size = 10
    while font_size > 1:
        line_height = font_size + 2
        total_lines = len(rows) + 2  # title + headers + data
        required_height = total_lines * line_height

        if top_y - required_height > min_y:
            break
        font_size -= 1

    if font_size <= 1:
        raise ValueError("⚠️ Annotation too long to fit even at smallest font size.")

    # Create overlay
    packet = BytesIO()
    c = canvas.Canvas(packet, pagesize=(page_width, page_height))

    # Draw title
    c.setFont("Helvetica-Bold", font_size + 2)
    c.drawString(left_col_x, top_y, title)

    # Draw column headers
    header_y = top_y - line_height
    c.setFont(font_name, font_size)
    c.drawString(left_col_x, header_y, "fits")
    c.drawString(middle_col_x, header_y, "values")
    c.drawString(right_col_x, header_y, "targets")
    c.line(left_col_x, header_y - 2, right_col_x + 80, header_y - 2)

    # Draw rows
    for i, (left, middle, right) in enumerate(rows):
        y = header_y - (i + 1) * line_height
        if y < min_y:
            break  # Stop if bottom margin hit
        c.drawString(left_col_x, y, left)
        if middle:
            c.drawString(middle_col_x, y, middle)
        if right:
            c.drawString(right_col_x, y, right)

    c.save()

    # Merge overlay
    packet.seek(0)
    overlay_pdf = PdfReader(packet)
    overlay_page = overlay_pdf.pages[0]
    page.merge_page(overlay_page)

    return page

def extract_fit_rows(d, md, td, indent=0):
    """
    Recursively extracts rows of (left_text, right_text) for annotation printing.
    left_text = "    " * indent + "key: fit"
    right_text = str(mvalue), if available
    """
    rows = []
    for key, value in d.items():
        if 'by_unit' in key or 'unit_metrics' in key:
            continue
        if key in ['std', 'cov', 'median']:
            continue

        try:
            mvalue = md.get(key, None)
        except Exception:
            mvalue = None
            
        try:
            tvalue = td.get(key, None)
        except Exception:
            tvalue = None

        if isinstance(value, dict):
            if 'fit' in value:
                left = "    " * indent + f"{key}: {value['fit']}"
                if isinstance(mvalue, dict) and isinstance(tvalue, dict):
                    middle = ""
                    right = "" 
                elif isinstance(mvalue, (float, int, np.int64)) and isinstance(tvalue, (float, int, np.int64)):
                    middle = str(mvalue)
                    right = str(tvalue)
                else:
                    middle = ""
                    right = ""
                rows.append((left, middle, right))
            # Recurse
            rows.extend(extract_fit_rows(value, mvalue, tvalue, indent + 1))
    return rows

def dimensional_analysis(batch_dirs, parallel=False, workers=4):
    json_paths = find_fitness_jsons(batch_dirs)

    print(f"Processing {len(json_paths)} fitness JSON files...")
    with Pool(workers) if parallel else DummyContext() as pool:
        # job_args = list(filter(None, pool.map(load_job_info, sorted_pdfs)))
        # #top_n = 150  # <-- You can make this a CLI argument too if you want
        # top_n = 256
        # job_args = filter_top_n_candidates(job_args, top_n=top_n)
        # job_args = flag_param_violation(job_args)
        #fitness_entries = pool.map(load_fit_file, json_paths)
        burst_fits = pool.map(load_bursting_fit, json_paths)
    
    # filter out the top 256 candidates
    top_n = 128
    top_n_burst_fits = filter_top_n_candidates(burst_fits, top_n=top_n)
        
    #sort by burst_rate_fit
    rate_sorted = get_rate_sorted(top_n_burst_fits)
    ibi_sorted = get_ibi_sorted(top_n_burst_fits)
    burst_amp_sorted = get_burst_amp_sorted(top_n_burst_fits)
    burst_duration_sorted = get_burst_duration_sorted(top_n_burst_fits)
    num_units_per_burst_sorted = get_num_units_per_burst_sorted(top_n_burst_fits)
    in_burst_fr_sorted = get_in_burst_fr_sorted(top_n_burst_fits)
    
    sorted_lists = {
        'burst_rate': rate_sorted,
        'ibi': ibi_sorted,
        'burst_amp': burst_amp_sorted,
        'burst_duration': burst_duration_sorted,
        'num_units_per_burst': num_units_per_burst_sorted,
        'in_burst_fr': in_burst_fr_sorted
    }
    
    # Create directory to save the merged PDFs
    # merged_output_dir = os.path.join(batch_dir, 'dimensional_analysis')
    # os.makedirs(merged_output_dir, exist_ok=True)

    #pdfs_to_merge = []
    pdfs_to_merge = {}

    for key, sorted_list in sorted_lists.items():
        best_fit = sorted_list[0]
        best_path = best_fit[3]
        fitness_json_path = best_path
        metrics_npy_path = best_path.replace('_fitness.json', '_metrics.npy')
        cfg_path = best_path.replace('_fitness.json', '_cfg.json')
        plot_dir = best_path.replace('_fitness.json', '')
        batch_dir = os.path.dirname(os.path.dirname(plot_dir))
        dim_output_dir = os.path.join(batch_dir, 'dimensional_analysis')
        os.makedirs(dim_output_dir, exist_ok=True)
        pdf_path = os.path.join(plot_dir, 'network_summary_3p_annotated_flat.pdf')
        
        # load the cfg file
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        
        # target_npy
        targets_npy_path = cfg['simConfig']['features_path']
        
        if os.path.exists(pdf_path):
            #pdfs_to_merge.append(pdf_path)
            pdfs_to_merge[key] = (pdf_path, fitness_json_path, metrics_npy_path, targets_npy_path)
            print(f"✅ Added {pdf_path}")
        else:
            print(f"❌ Missing PDF: {pdf_path}")

    # Merge collected PDFs
    if pdfs_to_merge:
        #merger = PdfMerger()
        writer = PdfWriter()
        
        for key, paths in pdfs_to_merge.items():
        #for pdf_path in pdfs_to_merge:
            pdf_path, fitness_json_path, metrics_npy_path, targets_npy_path = paths

            reader = PdfReader(pdf_path)
            page = reader.pages[0]

            title = f"Best Fit: {key}"
            page = scale_pdf_page_top_left(page, scale_factor=0.5)
            page = add_fitness_annotation(page, title, fitness_json_path, metrics_npy_path, targets_npy_path, scale_factor=0.5)

            #merger.add_page(page)
            writer.add_page(page)

        
        merged_pdf_path = os.path.join(dim_output_dir, 'bursting_dim_analysis_summary.pdf')
        # merger.write(merged_pdf_path)
        # merger.close()
        
        with open(merged_pdf_path, 'wb') as f:
            writer.write(f)
            
        
        print(f"📄 Merged PDF created at: {merged_pdf_path}")
    else:
        print("⚠️ No PDFs were collected for merging.")
        
    
    # # copy paste the pdf to a new location (deprecated)
    # for key, sorted_list in sorted_lists.items():
        
    #     best_fit = sorted_list[0]
    #     best_fit_gen = best_fit[0]
    #     best_fit_cand = best_fit[1]
    #     best_path = best_fit[3]
    #     plot_dir = best_path.replace('_fitness.json', '')
    #     pdf_path = os.path.join(plot_dir, 'network_summary_3p_annotated_flat.pdf')
    #     batch_dir = os.path.dirname(os.path.dirname(plot_dir))
        # dim_analysis_dir = os.path.join(batch_dir, 'dimensional_analysis')
        # burst_dim_analysis_dir = os.path.join(dim_analysis_dir, 'bursting')
        # list_dir = os.path.join(burst_dim_analysis_dir, key)
        # dest_path = os.path.join(list_dir, f'gen_{best_fit_gen}_cand_{best_fit_cand}_network_summary_3p_annotated_flat.pdf')
        # if os.path.exists(pdf_path):
        #     # copy to folder to collect.
        #     os.makedirs(list_dir, exist_ok=True)
        #     os.system(f'cp {pdf_path} {dest_path}')
        #     print(f'Copied {pdf_path} to {dest_path}')
        # else:
        #     print(f'❌ {pdf_path} does not exist.')
    print(f"Loaded {len(burst_fits)} fitness entries.")
    #print(f"Loaded {len(fitness_entries)} fitness entries.")

# --------------------- Dummy Context for Serial Fallback ---------------------
class DummyContext:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def map(self, func, iterable): return list(map(func, iterable))

if __name__ == "__main__":
    
    # ─── configure your batch paths here ────────────────────────────────────────────
    batch_paths = [
        
        # aw 2025-04-29 11:03:23
        #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26/',
        #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-28/'

        ## aw 2025-05-16 09:31:55
        #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-15/',


        # aw 2025-05-16 12:03:29
        #'/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-16/',

        # aw 2025-05-20 15:08:54
        '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-19_BRandFRratios/',
    ]
    
    # ─── run the dimensional analysis ───────────────────────────────────────────────
    workers = 24
    dimensional_analysis(batch_paths, parallel=True, workers=workers)