
# copy all network files from the source directory to the destination directory
# source: /global/homes/a/adammwea/ben-shalom_nas/raw_data/Organoid_RTT_R270X_pA_pD_B1_d91/
# destination: /global/homes/a/adammwea/ben-shalom_nas/pscratch/zinputs/B6J_DensityTest_10012024_AR/
rsync -avhc --progress \
  --include='*/' \
  --include='*Network*/**' \
  --exclude='*analysis*/**' \
  --exclude='*' \
  /global/homes/a/adammwea/ben-shalom_nas/raw_data/SYNGAP1_T1_C1_03212024/ \
  /global/homes/a/adammwea/pscratch/zinputs/SYNGAP1_T1_C1_03212024/