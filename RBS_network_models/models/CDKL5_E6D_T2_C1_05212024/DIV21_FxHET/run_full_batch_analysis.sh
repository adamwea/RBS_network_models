#!/bin/bash
# aw 2025-05-26 20:19:47
#BATCH_PATH="/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-25_normBRandFRR_optBLandAmps_db"
#BATCH_PATH="/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-26_normBRandFRR_optBLandAmps_db"
#BATCH_PATH="/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-27_normBRandFRR_optBLandAmps_db"
#BATCH_PATH="/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-27_dealbreakers"

# aw 2025-05-27 14:52:02
#BATCH_PATH="/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-27_reduced_dealbreakers"
#BATCH_PATH="/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-27_reduced_dealbreakers_2"
#BATCH_PATH="/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-27_reduced_dealbreakers_3"
#BATCH_PATH="/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-28_reduced_dealbreakers_4"
BATCH_PATH="/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-28_spiking_only"


#----# Run the analysis
# plot evolution
# Run the script — output will be saved in $BATCH_PATH/evolution_reports/
python /global/homes/a/adammwea/dev/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_WT/plot_evolution.py \
  "$BATCH_PATH" \
  --workers 16

python /global/homes/a/adammwea/dev/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_WT/plot_drift.py \
    "$BATCH_PATH" \
    --output_dir "$BATCH_PATH/drift" \
    --workers 16

python /global/homes/a/adammwea/dev/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_WT/plot_batch.py \
  --batch-paths "$BATCH_PATH" \
  --top-n 32 \
  --num-workers 8 \
  --y-lim 0 16 \
  --parallel

python /global/homes/a/adammwea/dev/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_WT/report_batch.py \
  --batches "$BATCH_PATH" \
  --parallel \
  --workers 16