# ====================== List of Projects to Sync Raw Data From ======================

# aw 2025-07-15 11:23:38 - optogenetics pilot data
# data: /mnt/ben-shalom_nas/raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779
# target: /pscratch/sd/a/adammwea/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779
now=$(date +'%Y-%m-%d %H:%M:%S')
globus transfer "$lab_server_endpoint:/mnt/ben-shalom_nas/raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779" \
"$NERSC_DTN_endpoint:/pscratch/sd/a/adammwea/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779" \
--sync-level checksum --notify failed,inactive,succeeded \
--label "Sync CDKL5-R59X_MaxOnePlus_T1_05202025_PS to NERSC - $now" --verbose

# aw 2025-03-29 19:26:37 - I think B6J raw data might be corrupted on NERSC...resending to verify
#one day of data first to test
# data: /mnt/ben-shalom_nas/raw_data/rbs_maxtwo_desktop/harddisk20tb/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR/241007
# target: /pscratch/sd/a/adammwea/zinputs/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR/241007
now=$(date +'%Y-%m-%d %H:%M:%S')
globus transfer "$lab_server_endpoint:/mnt/ben-shalom_nas/raw_data/rbs_maxtwo_desktop/harddisk20tb/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR/241007" \
"$NERSC_DTN_endpoint:/pscratch/sd/a/adammwea/zinputs/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR/241007" \
--sync-level checksum --notify failed,inactive,succeeded \
--label "Sync B6J_DensityTest_10012024_AR to NERSC - $now" --verbose

#other test data
# data: /mnt/ben-shalom_nas/raw_data/rbs_maxtwo_desktop/harddisk20tb/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR/241004
# target: /pscratch/sd/a/adammwea/zinputs/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR/241004
now=$(date +'%Y-%m-%d %H:%M:%S')
globus transfer "$lab_server_endpoint:/mnt/ben-shalom_nas/raw_data/rbs_maxtwo_desktop/harddisk20tb/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR/241004" \
"$NERSC_DTN_endpoint:/pscratch/sd/a/adammwea/zinputs/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR/241004" \
--sync-level checksum --notify failed,inactive,succeeded \
--label "Sync B6J_DensityTest_10012024_AR to NERSC - $now" --verbose

#full sync
# Sync Raw Data from Local to NERSC - no sync required, just copy
# data: /mnt/ben-shalom_nas/raw_data/rbs_maxtwo_desktop/harddisk20tb/B6J_DensityTest_10012024_AR
# target: /pscratch/sd/a/adammwea/zinputs/B6J_DensityTest_10012024_AR
now=$(date +'%Y-%m-%d %H:%M:%S')
globus transfer "$lab_server_endpoint:/mnt/ben-shalom_nas/raw_data/rbs_maxtwo_desktop/harddisk20tb/B6J_DensityTest_10012024_AR" \
"$NERSC_DTN_endpoint:/pscratch/sd/a/adammwea/zinputs/B6J_DensityTest_10012024_AR" \
--sync-level checksum --notify failed,inactive,succeeded \
--label "Sync B6J_DensityTest_10012024_AR to NERSC - $now" --verbose

# before # aw 2025-03-29 19:26:33
# /volume1/MEA_Backup/raw_data/rbs_maxtwo_desktop/harddisk24tbvol1/Organoid_RTT_R270X_pA_pD_B1_d91
# Sync Raw Data from Local to NERSC - use size mtimes of files instead of checksum for faster syncing - use descriptive label for tracking
now=$(date +'%Y-%m-%d %H:%M:%S')
globus transfer "$lab_server_endpoint:/mnt/ben-shalom_nas/raw_data/rbs_maxtwo_desktop/harddisk24tbvol1/Organoid_RTT_R270X_pA_pD_B1_d91" \
"$NERSC_DTN_endpoint:/pscratch/sd/a/adammwea/workspace/_raw_data/Organoid_RTT_R270X_pA_pD_B1_d91" \
--sync-level mtime --notify failed,inactive,succeeded \
--label "Sync Organoid_RTT_R270X_pA_pD_B1_d91 to NERSC - $now" --verbose

# /volume1/MEA_Backup/raw_data/rbs_maxtwo_desktop/harddisk24tbvol1/MEASlices_02032025_PVSandCA
# Sync Raw Data from Local to NERSC - use size mtimes of files instead of checksum for faster syncing - use descriptive label for tracking
now=$(date +'%Y-%m-%d %H:%M:%S')
globus transfer "$lab_server_endpoint:/mnt/ben-shalom_nas/raw_data/rbs_maxtwo_desktop/harddisk24tbvol1/MEASlices_02032025_PVSandCA" \
"$NERSC_DTN_endpoint:/pscratch/sd/a/adammwea/workspace/_raw_data/MEASlices_02032025_PVSandCA" \
--sync-level mtime --notify failed,inactive,succeeded \
--label "Sync MEASlices_02032025_PVSandCA to NERSC - $now" --verbose

# /volume1/MEA_Backup/raw_data/rbs_maxtwo_desktop/harddisk24tbvol1/MEASlices_02122025_PVSandCA
# Sync Raw Data from Local to NERSC - use size mtimes of files instead of checksum for faster syncing - use descriptive label for tracking
now=$(date +'%Y-%m-%d %H:%M:%S')
globus transfer "$lab_server_endpoint:/mnt/ben-shalom_nas/raw_data/rbs_maxtwo_desktop/harddisk24tbvol1/MEASlices_02122025_PVSandCA" \
"$NERSC_DTN_endpoint:/pscratch/sd/a/adammwea/workspace/_raw_data/MEASlices_02122025_PVSandCA" \
--sync-level mtime --notify failed,inactive,succeeded \
--label "Sync MEASlices_02122025_PVSandCA to NERSC - $now" --verbose

# /volume1/MEA_Backup/raw_data/rbs_maxtwo_desktop/harddisk24tbvol1/MEASlices_02242025_PVSandCA
# Sync Raw Data from Local to NERSC - use size mtimes of files instead of checksum for faster syncing - use descriptive label for tracking
now=$(date +'%Y-%m-%d %H:%M:%S')
globus transfer "$lab_server_endpoint:/mnt/ben-shalom_nas/raw_data/rbs_maxtwo_desktop/harddisk24tbvol1/MEASlices_02242025_PVSandCA" \
"$NERSC_DTN_endpoint:/pscratch/sd/a/adammwea/workspace/_raw_data/MEASlices_02242025_PVSandCA" \
--sync-level mtime --notify failed,inactive,succeeded \
--label "Sync MEASlices_02242025_PVSandCA to NERSC - $now" --verbose

# checksum version
# /volume1/MEA_Backup/raw_data/rbs_maxtwo_desktop/harddisk24tbvol1/MEASlices_02242025_PVSandCA
# Sync Raw Data from Local to NERSC - use size mtimes of files instead of checksum for faster syncing - use descriptive label for tracking
now=$(date +'%Y-%m-%d %H:%M:%S')
globus transfer "$lab_server_endpoint:/mnt/ben-shalom_nas/raw_data/rbs_maxtwo_desktop/harddisk24tbvol1/MEASlices_02242025_PVSandCA" \
"$NERSC_DTN_endpoint:/pscratch/sd/a/adammwea/workspace/_raw_data/MEASlices_02242025_PVSandCA" \
--sync-level checksum --notify failed,inactive,succeeded \
--label "Sync MEASlices_02242025_PVSandCA to NERSC - $now" --verbose

## ====================== List of Timers to Sync Analysis Data From ======================
# /volume1/MEA_Backup/raw_data/rbs_maxtwo_desktop/harddisk24tbvol1/MEASlices_02242025_PVSandCA
# Sync Raw Data from Local to NERSC - use size mtimes of files instead of checksum for faster syncing - use descriptive label for tracking
# Define Variables
now=$(date +'%Y-%m-%d %H:%M:%S')
export TZ="America/Los_Angeles"
today=$(date +'%Y-%m-%d')
start_time=$(date -u -d "$today 17:00:00 PST" +'%Y-%m-%dT%H:%M:%SZ')
globus timer create transfer "$lab_server_endpoint:/mnt/ben-shalom_nas/analysis/adamm/workspace_perlmutter/2022-03-01_2022-03-31" \
"$NERSC_DTN_endpoint:/pscratch/sd/a/adammwea/workspace/_analysis_data/2022-03-01_2022-03-31" \
--start "$start_time" \
--label "Scheduled Sync MEASlices_02242025_PVSandCA to NERSC - $now" \
--sync-level mtime \
--notify failed,inactive,succeeded \
--verbose \
--stop-after-runs=1
