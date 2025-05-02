'''
Dont try to run this on login node. Designed for local or interactive session.
'''

import os
import json
import glob
import time
import numpy as np
from datetime import datetime
from multiprocessing import Process, Queue, Value, Lock, cpu_count
from pathlib import Path
from threading import Thread
from netpyne import sim
from rich.console import Console
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, TimeRemainingColumn
from RBS_network_models.fitnessFunc import fitnessFunc_v3
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.conv_params import conv_params, mega_params

# === Constants ===
REFERENCE_DATA_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well005/metrics.npy'
BATCH_PATH = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-04-26'
CHILD_MAX_WORKERS = 64

# === Globals ===
completed_tasks = Value('i', 0)
lock = Lock()
console = Console()

# === Fitness file loader ===
def load_fitness_scores(batch_path):
    fitness_files = glob.glob(f"{batch_path}/**/*_fitness.json", recursive=True)
    scored = []
    
    for path in fitness_files:
        try:
            with open(path, 'r') as f:
                data = json.load(f).get('fit', 1000) # Default to 1000 if not found
            if data < 1000:
                scored.append((os.path.getmtime(path), path))
        except:
            continue

    scored.sort(key=lambda x: x[0])
    return [s[1] for s in scored]

# === Worker process ===
def worker(queue, completed_tasks, lock, reference_data_path):
    while not queue.empty():
        try:
            sim_data_path = queue.get_nowait()
        except:
            break

        try:
            sim.load(sim_data_path)
            simData = sim.allSimData.todict().copy()

            fitnessFuncArgs = {
                'conv_params': conv_params,
                'mega_params': mega_params,
                'plot_sim': False,
                'reference_data_path': reference_data_path,
                'batching': False,
                'sim_data_path': sim_data_path,
                'try_load': False,
                'run_parallel': True,
                'max_workers': CHILD_MAX_WORKERS,
                'burst_sequencing': True,       
            }

            avg_fitness = fitnessFunc_v3(simData, **fitnessFuncArgs)
            sim.clearAll()

        except Exception as e:
            avg_fitness = f"Error: {str(e)}"

        with lock:
            completed_tasks.value += 1

# === Monitor CLI progress ===
def monitor_progress(total_tasks, completed_tasks, lock):
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing", total=total_tasks)
        while not progress.finished:
            time.sleep(10)
            with lock:
                done = completed_tasks.value
            progress.update(task, completed=done)
            if done >= total_tasks:
                break

# === Main entry ===
def run_manual_process_pool():
    
    #fitness_mode = True
    fitness_mode = False
    if fitness_mode:
        fitness_files = load_fitness_scores(BATCH_PATH)
        sim_data_paths = [str(Path(f).with_name(Path(f).stem.replace('_fitness', '_data') + '.pkl')) for f in fitness_files]
        sim_data_paths = [p for p in sim_data_paths if os.path.exists(p)]
        total_tasks = len(sim_data_paths)
        
    pkl_mode = True
    if pkl_mode:
        sim_data_paths = glob.glob(BATCH_PATH + '/**/*_data.pkl', recursive=True)
        sim_data_paths = [p for p in sim_data_paths if os.path.exists(p)]
        corresponding_npy_paths = [str(Path(p).with_name(Path(p).stem.replace('_data', '_metrics') + '.npy')) for p in sim_data_paths]
        
        # If both _data and _fitness have been modified today, remove corresponding sim_data_paths
        today = datetime.now().date()
        for data_path, fitness_path in zip(sim_data_paths, corresponding_npy_paths):
            if os.path.exists(data_path) and os.path.exists(fitness_path):
                data_mtime = datetime.fromtimestamp(os.path.getmtime(data_path)).date()
                fitness_mtime = datetime.fromtimestamp(os.path.getmtime(fitness_path)).date()
                if data_mtime == today and fitness_mtime == today:
                    sim_data_paths.remove(data_path)
                    print(f"Removing {data_path} because both {data_path} and {fitness_path} were modified today.")
                
        total_tasks = len(sim_data_paths)

    if total_tasks == 0:
        console.print("[bold red]No valid simulation files found.")
        return

    queue = Queue()
    for path in sim_data_paths:
        queue.put(path)

    num_sockets = min(8, cpu_count() // 2)
    console.print(f"[green]Launching {num_sockets} parent processes (each with up to 64 child workers).")

    monitor_thread = Thread(target=monitor_progress, args=(total_tasks, completed_tasks, lock))
    monitor_thread.start()

    processes = []
    for _ in range(num_sockets):
        p = Process(target=worker, args=(queue, completed_tasks, lock, REFERENCE_DATA_PATH))
        p.daemon = False  # Key: must be non-daemonic
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    monitor_thread.join()
    return "Completed"

run_manual_process_pool()
