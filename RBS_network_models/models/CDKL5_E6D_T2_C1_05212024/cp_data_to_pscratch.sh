# copy all network files from the source directory to the destination directory
# source: /global/homes/a/adammwea/ben-shalom_nas/raw_data/CDKL5-E6D_T2_C1_05212024/
# destination: /global/homes/a/adammwea/ben-shalom_nas/pscratch/zinputs/CDKL5-E6D_T2_C1_05212024/
#mkdir -p /global/homes/a/adammwea/ben-shalom_nas/pscratch/zinputs/CDKL5-E6D_T2_C1_05212024/
rsync -avhc --progress \
  --include='*/' \
  --include='*Network*/**' \
  --exclude='*analysis*/**' \
  --exclude='*' \
  /global/homes/a/adammwea/ben-shalom_nas/raw_data/CDKL5-E6D_T2_C1_05212024/ \
  /global/homes/a/adammwea/pscratch/zinputs/CDKL5-E6D_T2_C1_05212024/
  # /global/homes/a/adammwea/ben-shalom_nas/raw_data/Organoid_RTT_R270X_pA_pD_B1_d91/ \
  # /global/homes/a/adammwea/pscrach/zinputs/Organoid_RTT_R270X_pA_pD_B1_d91/
  # #/global/homes/a/adammwea/pscratch/zinputs/B6J_DensityTest_10012024_AR/