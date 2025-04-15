# consolidate raw data in source... there are redundant data. all data should be in the deeper directory
# shallow dir: /global/homes/a/adammwea/ben-shalom_nas/raw_data/Organoid_RTT_R270X_pA_pD_B1_d91/
# deep dir: /global/homes/a/adammwea/ben-shalom_nas/raw_data/Organoid_RTT_R270X_pA_pD_B1_d91/Organoid_RTT_R270X_pA_pD_B1_d91/
# rsync -avhc --progress --remove-source-files \
#   --exclude 'Organoid_RTT_R270X_pA_pD_B1_d91' \
#   /global/homes/a/adammwea/ben-shalom_nas/raw_data/Organoid_RTT_R270X_pA_pD_B1_d91/ \
#   /global/homes/a/adammwea/ben-shalom_nas/raw_data/Organoid_RTT_R270X_pA_pD_B1_d91/Organoid_RTT_R270X_pA_pD_B1_d91/ --dry-run

# written as loop for testing
for dir in 250114 250116 250107 241224 241231; do
  rsync -avhc --progress --remove-source-files \
    "/global/homes/a/adammwea/ben-shalom_nas/raw_data/Organoid_RTT_R270X_pA_pD_B1_d91/$dir/" \
    "/global/homes/a/adammwea/ben-shalom_nas/raw_data/Organoid_RTT_R270X_pA_pD_B1_d91/Organoid_RTT_R270X_pA_pD_B1_d91/$dir/" #--dry-run
done

# copy all network files from the source directory to the destination directory
# source: /global/homes/a/adammwea/ben-shalom_nas/raw_data/Organoid_RTT_R270X_pA_pD_B1_d91/
# destination: /global/homes/a/adammwea/ben-shalom_nas/pscratch/zinputs/B6J_DensityTest_10012024_AR/
rsync -avhc --progress \
  --include='*/' \
  --include='*Network*/**' \
  --exclude='*analysis*/**' \
  --exclude='*' \
  /global/homes/a/adammwea/ben-shalom_nas/raw_data/Organoid_RTT_R270X_pA_pD_B1_d91/ \
  /global/homes/a/adammwea/pscratch/zinputs/Organoid_RTT_R270X_pA_pD_B1_d91/
  #/global/homes/a/adammwea/pscratch/zinputs/B6J_DensityTest_10012024_AR/