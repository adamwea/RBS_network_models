# copy all network files from the source directory to the destination directory

# initial/detailed copy
rsync -avhc --progress \
  --include='*/' \
  --include='*Network*/**' \
  --exclude='*analysis*/**' \
  --exclude='*' \
  /global/homes/a/adammwea/ben-shalom_nas/raw_data/CDKL5-E6D_T2_C1_05212024/ \
  /global/homes/a/adammwea/pscratch/z_raw_data/CDKL5-E6D_T2_C1_05212024/

# just sanity check using size  
rsync -avh --size-only --progress \
  --include='*/' \
  --include='*Network*/**' \
  --exclude='*analysis*/**' \
  --exclude='*' \
  /global/homes/a/adammwea/ben-shalom_nas/raw_data/CDKL5-E6D_T2_C1_05212024/ \
  /global/homes/a/adammwea/pscratch/z_raw_data/CDKL5-E6D_T2_C1_05212024/