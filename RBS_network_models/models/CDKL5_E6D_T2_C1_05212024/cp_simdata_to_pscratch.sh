# copy all network files from the source directory to the destination directory

# initial/detailed copy
rsync -avhc --progress \
  /global/homes/a/adammwea/ben-shalom_nas/simulated_data/CDKL5-E6D_T2_C1_05212024/ \
  /global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/

# targeted copy for data that is needed for dev
rsync -avh --progress \
  --size-only \
  /global/homes/a/adammwea/ben-shalom_nas/simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/ \
  /global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/

# just sanity check using size  
rsync -avh --size-only --progress \
  /global/homes/a/adammwea/ben-shalom_nas/raw_data/CDKL5-E6D_T2_C1_05212024/ \
  /global/homes/a/adammwea/pscratch/z_raw_data/CDKL5-E6D_T2_C1_05212024/