#!/bin/bash
#SBATCH --job-name=pipeline_analysis
#SBATCH -A m2043
#SBATCH -t 24:00:00
#SBATCH -N 10  # Number of nodes requested, adjust as needed
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --exclusive
#SBATCH --output=./NERSC/output/analysis_%j.out
#SBATCH --error=./NERSC/output/analysis_%j.err

# Functions ====================================================================
# Function to generate the list of wells (HDF5 files) to process
generate_plate_files() {
    PLATE_FILES=()
    for h5_dir in "${H5_PARENT_DIRS[@]}"; do
        while IFS= read -r -d '' file; do
            if [[ "$file" == *"/AxonTracking/"* ]]; then # Get all HDF5 files in the specified directories that contain "AxonTracking" in the path
                PLATE_FILES+=("$file")
            fi
        done < <(find "$h5_dir" -type f -name '*.h5' -print0)
    done
}

# Function to extract recording details and generate unique log file names
generate_log_file_names() {
    local plate_file=$1
    local stream_select=$2

    local parent_dir=$(dirname "$plate_file")
    local runID=$(basename "$parent_dir")

    local grandparent_dir=$(dirname "$parent_dir")
    local scan_type=$(basename "$grandparent_dir")

    local great_grandparent_dir=$(dirname "$grandparent_dir")
    local chipID=$(basename "$great_grandparent_dir")

    local ggg_dir=$(dirname "$great_grandparent_dir")
    local date=$(basename "$ggg_dir")
    
    local gggg_dir=$(dirname "$ggg_dir")
    local project_name=$(basename "$gggg_dir")

    local log_file="${LOG_DIR}/${project_name}_${date}_${chipID}_${runID}_stream${stream_select}.log"
    local error_log_file="${LOG_DIR}/${project_name}_${date}_${chipID}_${runID}_stream${stream_select}.err"

    echo "$log_file" "$error_log_file"
}

# Args ========================================================================
# Define the HDF5 directories containing wells (you may want to specify these as arguments instead)
H5_PARENT_DIRS=(
    "/pscratch/sd/a/adammwea/RBS_synology_rsync/B6J_DensityTest_10012024_AR"
)
# Define output and log directory (ensure these exist before running)
export OUTPUT_DIR="/pscratch/sd/a/adammwea/zRBS_axon_reconstruction_output"
LOG_DIR="/pscratch/sd/a/adammwea/zRBS_axon_reconstruction_output/logs"
mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

# Define the maximum number of parallel jobs (adjust based on available resources)
MAX_JOBS=2  # Change as needed, matching the number of nodes or available resources

# Define the full path to the Python script to srun
PYTHON_SCRIPT_PATH="/pscratch/sd/a/adammwea/RBS_axonal_reconstructions/development_scripts/developing_nbs_for_AR_density_projects/run_pipeline_HPC.py"

# Define the shifter image
#SHIFTER_IMAGE="rohanmalige/benshalom:v3"
SHIFTER_IMAGE="adammwea/axonkilo_docker:v7"

# Main Script ==================================================================

# Load necessary modules
echo "Loading parallel module..."
module load parallel
module load conda
conda activate axon_env

# Generate the list of wells (HDF5 files) to process
generate_plate_files
echo "Number of plates to process: ${#PLATE_FILES[@]}"

# Calculate the number of wells
NUM_WELLS=$(( ${#PLATE_FILES[@]} * 6 ))
echo "Number of wells to process: ${NUM_WELLS}"

# # Loop through plate_files and stream_select values and run the Python command in parallel
# for plate_file in "${PLATE_FILES[@]}"; do
#     for stream_select in {0..5}; do
#         export PLATE_FILE=${plate_file}
#         export STREAM_SELECT=${stream_select}
#         read log_file error_log_file < <(generate_log_file_names "$plate_file" "$stream_select")
#         echo "Processing plate file: $plate_file, stream: $stream_select"
#         mkdir -p "$(dirname "$log_file")"  # Ensure the log directory exists
#         # shifter --image=${SHIFTER_IMAGE} bash -c "
#         #     module load parallel
#         #     conda activate axon_env
#         #     seq 0 $NUM_WELLS | srun parallel -j $MAX_JOBS python3 $PYTHON_SCRIPT_PATH --plate_file $plate_file --stream_select $stream_select --output_dir ${OUTPUT_DIR}
#         # "
#         seq 0 $NUM_WELLS | parallel -j $MAX_JOBS \
#             shifter --image=${SHIFTER_IMAGE} /bin/bash -c "
#             conda activate axon_env; 
#             python3 $PYTHON_SCRIPT_PATH --plate_file $PLATE_FILE --stream_select $STREAM_SELECT --output_dir ${OUTPUT_DIR}"
#     done
# done


# Process each plate file
for plate_file in "${PLATE_FILES[@]}"; do
    for stream in $(seq 0 5); do
        log_file_and_error_log_file=$(generate_log_file_names "$plate_file" "$stream")
        log_file=$(echo $log_file_and_error_log_file | cut -d ' ' -f 1)
        error_log_file=$(echo $log_file_and_error_log_file | cut -d ' ' -f 2)
        echo "Processing plate file: $plate_file, stream: $stream"
        srun -n 1 --gres=gpu:1 shifter --image=$SHIFTER_IMAGE python3 $PYTHON_SCRIPT_PATH --plate_file $plate_file --stream_select $stream --output_dir $OUTPUT_DIR > "$log_file" 2> "$error_log_file" &
        if (( $(jobs -r -p | wc -l) >= MAX_JOBS )); then
            wait -n
        fi
    done
done

wait