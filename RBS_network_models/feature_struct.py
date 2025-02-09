from RBS_network_models.network_analysis import get_min, get_max

import numpy as np

def get_target_E_FR(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_I_FR(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_E_FR(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_I_FR(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_E_FR(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_I_FR(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_E_CoV_FR(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fr_CoV'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_I_CoV_FR(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fr_CoV'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_E_CoV_FR(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fr_CoV'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_I_CoV_FR(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fr_CoV'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_E_CoV_FR(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fr_CoV'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_I_CoV_FR(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fr_CoV'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_E_ISI(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_I_ISI(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_E_ISI(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_I_ISI(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_E_ISI(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_I_ISI(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_E_CoV_ISI(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['isi_CoV'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_I_CoV_ISI(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['isi_CoV'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_E_CoV_ISI(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['isi_CoV'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_I_CoV_ISI(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['isi_CoV'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_E_CoV_ISI(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['isi_CoV'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_I_CoV_ISI(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['isi_CoV'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_fano_factor_E_target(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fano_factor'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_fano_factor_I_target(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fano_factor'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_fano_factor_E_min(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fano_factor'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_fano_factor_I_min(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fano_factor'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_fano_factor_E_max(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fano_factor'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_fano_factor_I_max(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['spiking_data']['spiking_data_by_unit'][unit]['fano_factor'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_withinBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_within'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_withinBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_within'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_withinBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_within'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_withinBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_within'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_withinBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_within'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_withinBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_within'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_CoVWithinBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_within'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_CoVWithinBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_within'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_CoVWithinBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_within'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_CoVWithinBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_within'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_CoVWithinBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_within'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_CoVWithinBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_within'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_outsideBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_outside'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_outsideBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_outside'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_outsideBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_outside'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_outsideBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_outside'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_outsideBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_outside'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_outsideBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['mean_isi_outside'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def get_target_CoVOutsideBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_outside'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmean(spiking_data_by_E_units)

def get_target_CoVOutsideBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_outside'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmean(spiking_data_by_I_units)

def get_min_CoVOutsideBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_outside'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmin(spiking_data_by_E_units)

def get_min_CoVOutsideBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_outside'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmin(spiking_data_by_I_units)

def get_max_CoVOutsideBurst_ISI_E(network_metrics):
    E_units = network_metrics['classification_data']['excitatory_neurons']
    spiking_data_by_E_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_outside'] for unit in E_units]
    #replace nones with nan
    spiking_data_by_E_units = [x if x is not None else np.nan for x in spiking_data_by_E_units]
    return np.nanmax(spiking_data_by_E_units)

def get_max_CoVOutsideBurst_ISI_I(network_metrics):
    I_units = network_metrics['classification_data']['inhibitory_neurons']
    spiking_data_by_I_units = [network_metrics['bursting_data']['bursting_data_by_unit'][unit]['cov_isi_outside'] for unit in I_units]
    #replace nones with nan
    spiking_data_by_I_units = [x if x is not None else np.nan for x in spiking_data_by_I_units]
    return np.nanmax(spiking_data_by_I_units)

def build_network_metric_targets_dict(network_metrics):
    
    #initialize network_metric_targets
    bursting_data = network_metrics['bursting_data']
    mega_bursting_data = network_metrics['mega_bursting_data']
    duration_seconds = network_metrics['timeVector'][-1]
    
    network_metric_targets = {
        #General Data
        #'source': f'{network_metrics['source']}', # 'simulated' or 'experimental'
        'source': 'experimental',
        'duration_seconds': duration_seconds,
        'number_of_units': network_metrics['classification_data']['no. of units'],
        'inhib_units': network_metrics['classification_data']['inhibitory_neurons'],
        'excit_units': network_metrics['classification_data']['excitatory_neurons'],
        'number_of_inhib_units': len(network_metrics['classification_data']['inhibitory_neurons']),
        'number_of_excit_units': len(network_metrics['classification_data']['excitatory_neurons']),
        'unit_locations': network_metrics['classification_data']['unit_locations'],
        #'timeVector': network_metrics['timeVector'],
        
        # Spiking Data
        'spiking_data': {
            #'spike_times': network_metrics['spiking_data']['spike_times'],
            #'spiking_times_by_unit': network_metrics['spiking_data']['spiking_times_by_unit'],
            #'spiking_data_by_unit': network_metrics['spiking_data']['spiking_data_by_unit'],
            'spiking_summary_data': {
                'MeanFireRate': {
                    'target': network_metrics['spiking_data']['spiking_summary_data']['MeanFireRate'],
                    'min': get_min(network_metrics['spiking_data']['spiking_data_by_unit'], 'FireRate'),
                    'max': get_max(network_metrics['spiking_data']['spiking_data_by_unit'], 'FireRate'),      

                    'target_E': get_target_E_FR(network_metrics),
                    'min_E': get_min_E_FR(network_metrics),
                    'max_E': get_max_E_FR(network_metrics),
                    
                    'target_I': get_target_I_FR(network_metrics),
                    'min_I': get_min_I_FR(network_metrics),
                    'max_I': get_max_I_FR(network_metrics),                 
                                        
                    'weight': 1, # TODO: update these with Nfactors
                },
                'CoVFireRate': {
                    'target': network_metrics['spiking_data']['spiking_summary_data']['CoVFireRate'],
                    'min': get_min(network_metrics['spiking_data']['spiking_data_by_unit'], 'fr_CoV'),
                    'max': get_max(network_metrics['spiking_data']['spiking_data_by_unit'], 'fr_CoV'),
                    
                    'target_E': get_target_E_CoV_FR(network_metrics),
                    'min_E': get_min_E_CoV_FR(network_metrics),
                    'max_E': get_max_E_CoV_FR(network_metrics),
                    
                    'target_I': get_target_I_CoV_FR(network_metrics),
                    'min_I': get_min_I_CoV_FR(network_metrics),
                    'max_I': get_max_I_CoV_FR(network_metrics),
                    
                    'weight': 1, # TODO: update these with Nfactors
                },
                'MeanISI': {
                    'target': network_metrics['spiking_data']['spiking_summary_data']['MeanISI'],
                    'min': get_min(network_metrics['spiking_data']['spiking_data_by_unit'], 'meanISI'),
                    'max': get_max(network_metrics['spiking_data']['spiking_data_by_unit'], 'meanISI'),
                    
                    'target_E': get_target_E_ISI(network_metrics),
                    'min_E': get_min_E_ISI(network_metrics),
                    'max_E': get_max_E_ISI(network_metrics),
                    
                    'target_I': get_target_I_ISI(network_metrics),
                    'min_I': get_min_I_ISI(network_metrics),
                    'max_I': get_max_I_ISI(network_metrics),
                    
                    'weight': 1, # TODO: update these with Nfactors
                },
                'CoV_ISI': {
                    'target': network_metrics['spiking_data']['spiking_summary_data']['CoV_ISI'],
                    'min': get_min(network_metrics['spiking_data']['spiking_data_by_unit'], 'isi_CoV'),
                    'max': get_max(network_metrics['spiking_data']['spiking_data_by_unit'], 'isi_CoV'),
                    
                    'target_E': get_target_E_CoV_ISI(network_metrics),
                    'min_E': get_min_E_CoV_ISI(network_metrics),
                    'max_E': get_max_E_CoV_ISI(network_metrics),
                    
                    'target_I': get_target_I_CoV_ISI(network_metrics),
                    'min_I': get_min_I_CoV_ISI(network_metrics),
                    'max_I': get_max_I_CoV_ISI(network_metrics),
                    
                    #'target_E': get_target_E_CoV_ISI(network_metrics),                    
                    'weight': 1, # TODO: update these with Nfactors
                },
                'fano_factor': {
                    'target_E': get_fano_factor_E_target(network_metrics),
                    'min_E': get_fano_factor_E_min(network_metrics),
                    'max_E': get_fano_factor_E_max(network_metrics),
                    
                    'target_I': get_fano_factor_I_target(network_metrics),
                    'min_I': get_fano_factor_I_min(network_metrics),
                    'max_I': get_fano_factor_I_max(network_metrics),
                    
                    'weight': 1, # TODO: update these with Nfactors
                }
            },
        },
        
        
        #Bursting Data
        'bursting_data': {
            'bursting_summary_data': {
                'MeanBurstRate': {
                    'target': bursting_data['bursting_summary_data'].get('mean_Burst_Rate'),
                    # 'min': get_min_burst(bursting_data['bursting_data_by_unit'], 'bursts', duration_seconds), # NOTE: Calculated as individual unit burst participation rate.
                    # 'max': get_max_burst(bursting_data['bursting_data_by_unit'], 'bursts', duration_seconds), # NOTE: Calculated as individual unit burst participation rate.
                    'min': 0,
                    'max': None,
                    'weight': 1,
                },                
                'MeanWithinBurstISI': {
                    'target': bursting_data['bursting_summary_data'].get('MeanWithinBurstISI'),
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_isi_within'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_isi_within'),
                    # 'target_E': get_target_withinBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_withinBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_withinBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_withinBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_withinBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_withinBurst_ISI_I(network_metrics),
                    # 'min': 0,
                    # 'max': None,
                    'weight': 1,
                },
                'CovWithinBurstISI': {
                    'target': bursting_data['bursting_summary_data'].get('CoVWithinBurstISI'),
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_within'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_within'),
                    # 'target_E': get_target_CoVWithinBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_CoVWithinBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_CoVWithinBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_CoVWithinBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_CoVWithinBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_CoVWithinBurst_ISI_I(network_metrics),
                    
                    'weight': 1,
                },
                'MeanOutsideBurstISI': {
                    'target': bursting_data['bursting_summary_data'].get('MeanOutsideBurstISI'),
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_isi_outside'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_isi_outside'),
                    # 'target_E': get_target_outsideBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_outsideBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_outsideBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_outsideBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_outsideBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_outsideBurst_ISI_I(network_metrics),
                    
                    # 'min': 0,
                    # 'max': None,
                    'weight': 1, 
                },
                'CoVOutsideBurstISI': {
                    'target': bursting_data['bursting_summary_data'].get('CoVOutsideBurstISI'),
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    # 'target_E': get_target_CoVOutsideBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_CoVOutsideBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_CoVOutsideBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_CoVOutsideBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_CoVOutsideBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_CoVOutsideBurst_ISI_I(network_metrics),
                    
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    'weight': 1,
                },
                # 'MeanNetworkISI': {
                #     'target': bursting_data['bursting_summary_data'].get('MeanNetworkISI'),
                #     'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     # 'target_E': get_target_meanNetwork_ISI_E(network_metrics),
                #     # 'target_I': get_target_meanNetwork_ISI_I(network_metrics),
                #     # 'min_E': get_min_meanNetwork_ISI_E(network_metrics),
                #     # 'min_I': get_min_meanNetwork_ISI_I(network_metrics),
                #     # 'max_E': get_max_meanNetwork_ISI_E(network_metrics),
                #     # 'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     # 'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     # 'min': 0,
                #     # 'max': None,
                #     'weight': 1,
                # },
                # 'CoVNetworkISI': {
                #     'target': bursting_data['bursting_summary_data'].get('CoVNetworkISI'),
                #     'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     # 'target_E': get_target_CoVNetwork_ISI_E(network_metrics),
                #     # 'target_I': get_target_CoVNetwork_ISI_I(network_metrics),
                #     # 'min_E': get_min_CoVNetwork_ISI_E(network_metrics),
                #     # 'min_I': get_min_CoVNetwork_ISI_I(network_metrics),
                #     # 'max_E': get_max_CoVNetwork_ISI_E(network_metrics),
                #     # 'max_I': get_max_CoVNetwork_ISI_I(network_metrics),
                    
                #     # 'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     # 'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     'weight': 1,
                # },
                'NumUnits': {
                    # 'target': bursting_data['bursting_summary_data'].get('NumUnits'),
                    # 'min': 1,
                    # 'max': None,
                    # 'weight': 1,
                    'target': network_metrics['classification_data']['no. of units'],
                    'target_I': len(network_metrics['classification_data']['inhibitory_neurons']),
                    'target_E': len(network_metrics['classification_data']['excitatory_neurons']),
                },
                'Number_Bursts': {
                    'target': bursting_data['bursting_summary_data'].get('Number_Bursts'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'mean_IBI': {
                    'target': bursting_data['bursting_summary_data'].get('mean_IBI'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'cov_IBI': {
                    'target': bursting_data['bursting_summary_data'].get('cov_IBI'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'mean_Burst_Peak': {
                    'target': bursting_data['bursting_summary_data'].get('mean_Burst_Peak'),
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_burst_peak'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_burst_peak'),
                    # 'min': None,
                    # 'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'cov_Burst_Peak': {
                    'target': bursting_data['bursting_summary_data'].get('cov_Burst_Peak'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'fano_factor': {
                    'target': bursting_data['bursting_summary_data'].get('fano_factor'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'baseline': {
                    'target': bursting_data['bursting_summary_data'].get('baseline'),
                    'min': None,
                    'max': None,
                    'weight': 1,
                },
            },
        },
        
        #Mega Bursting Data
        'mega_bursting_data': {
            'bursting_summary_data': {
                'MeanBurstRate': {
                    'target': mega_bursting_data['bursting_summary_data'].get('mean_Burst_Rate'),
                    # 'min': get_min_burst(bursting_data['bursting_data_by_unit'], 'bursts', duration_seconds), # NOTE: Calculated as individual unit burst participation rate.
                    # 'max': get_max_burst(bursting_data['bursting_data_by_unit'], 'bursts', duration_seconds), # NOTE: Calculated as individual unit burst participation rate.
                    'min': 0,
                    'max': None,
                    'weight': 1,
                },                
                'MeanWithinBurstISI': {
                    'target': mega_bursting_data['bursting_summary_data'].get('MeanWithinBurstISI'),
                    # 'min': get_min(mega_bursting_data['bursting_data_by_unit'], 'mean_isi_within'),
                    # 'max': get_max(mega_bursting_data['bursting_data_by_unit'], 'mean_isi_within'),
                    # 'target_E': get_target_withinBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_withinBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_withinBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_withinBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_withinBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_withinBurst_ISI_I(network_metrics),
                    # 'min': 0,
                    # 'max': None,
                    'weight': 1,
                },
                'CovWithinBurstISI': {
                    'target': mega_bursting_data['bursting_summary_data'].get('CoVWithinBurstISI'),
                    # 'min': get_min(mega_bursting_data['bursting_data_by_unit'], 'cov_isi_within'),
                    # 'max': get_max(mega_bursting_data['bursting_data_by_unit'], 'cov_isi_within'),
                    # 'target_E': get_target_CoVWithinBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_CoVWithinBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_CoVWithinBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_CoVWithinBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_CoVWithinBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_CoVWithinBurst_ISI_I(network_metrics),
                    
                    'weight': 1,
                },
                'MeanOutsideBurstISI': {
                    'target': mega_bursting_data['bursting_summary_data'].get('MeanOutsideBurstISI'),
                    # 'min': get_min(mega_bursting_data['bursting_data_by_unit'], 'mean_isi_outside'),
                    # 'max': get_max(mega_bursting_data['bursting_data_by_unit'], 'mean_isi_outside'),
                    # 'target_E': get_target_outsideBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_outsideBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_outsideBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_outsideBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_outsideBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_outsideBurst_ISI_I(network_metrics),
                    
                    # 'min': 0,
                    # 'max': None,
                    'weight': 1, 
                },
                'CoVOutsideBurstISI': {
                    'target': mega_bursting_data['bursting_summary_data'].get('CoVOutsideBurstISI'),
                    # 'min': get_min(mega_bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    # 'max': get_max(mega_bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    # 'target_E': get_target_CoVOutsideBurst_ISI_E(network_metrics),
                    # 'target_I': get_target_CoVOutsideBurst_ISI_I(network_metrics),
                    # 'min_E': get_min_CoVOutsideBurst_ISI_E(network_metrics),
                    # 'min_I': get_min_CoVOutsideBurst_ISI_I(network_metrics),
                    # 'max_E': get_max_CoVOutsideBurst_ISI_E(network_metrics),
                    # 'max_I': get_max_CoVOutsideBurst_ISI_I(network_metrics),
                    
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_outside'),
                    'weight': 1,
                },
                # 'MeanNetworkISI': {
                #     'target': bursting_data['bursting_summary_data'].get('MeanNetworkISI'),
                #     'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     # 'target_E': get_target_meanNetwork_ISI_E(network_metrics),
                #     # 'target_I': get_target_meanNetwork_ISI_I(network_metrics),
                #     # 'min_E': get_min_meanNetwork_ISI_E(network_metrics),
                #     # 'min_I': get_min_meanNetwork_ISI_I(network_metrics),
                #     # 'max_E': get_max_meanNetwork_ISI_E(network_metrics),
                #     # 'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     # 'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_isi_all'),
                #     # 'min': 0,
                #     # 'max': None,
                #     'weight': 1,
                # },
                # 'CoVNetworkISI': {
                #     'target': bursting_data['bursting_summary_data'].get('CoVNetworkISI'),
                #     'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     # 'target_E': get_target_CoVNetwork_ISI_E(network_metrics),
                #     # 'target_I': get_target_CoVNetwork_ISI_I(network_metrics),
                #     # 'min_E': get_min_CoVNetwork_ISI_E(network_metrics),
                #     # 'min_I': get_min_CoVNetwork_ISI_I(network_metrics),
                #     # 'max_E': get_max_CoVNetwork_ISI_E(network_metrics),
                #     # 'max_I': get_max_CoVNetwork_ISI_I(network_metrics),
                    
                #     # 'min': get_min(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     # 'max': get_max(bursting_data['bursting_data_by_unit'], 'cov_isi_all'),
                #     'weight': 1,
                # },
                'NumUnits': {
                    # 'target': bursting_data['bursting_summary_data'].get('NumUnits'),
                    # 'min': 1,
                    # 'max': None,
                    # 'weight': 1,
                    'target': network_metrics['classification_data']['no. of units'],
                    'target_I': len(network_metrics['classification_data']['inhibitory_neurons']),
                    'target_E': len(network_metrics['classification_data']['excitatory_neurons']),
                },
                'Number_Bursts': {
                    'target': mega_bursting_data['bursting_summary_data'].get('Number_Bursts'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'mean_IBI': {
                    'target': mega_bursting_data['bursting_summary_data'].get('mean_IBI'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'cov_IBI': {
                    'target': mega_bursting_data['bursting_summary_data'].get('cov_IBI'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'mean_Burst_Peak': {
                    'target': mega_bursting_data['bursting_summary_data'].get('mean_Burst_Peak'),
                    # 'min': get_min(bursting_data['bursting_data_by_unit'], 'mean_burst_peak'),
                    # 'max': get_max(bursting_data['bursting_data_by_unit'], 'mean_burst_peak'),
                    # 'min': None,
                    # 'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'cov_Burst_Peak': {
                    'target': mega_bursting_data['bursting_summary_data'].get('cov_Burst_Peak'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'fano_factor': {
                    'target': mega_bursting_data['bursting_summary_data'].get('fano_factor'),
                    'min': None,
                    'max': None,
                    'weight': 1, #TODO: update these with Nfactors
                },
                'baseline': {
                    'target': mega_bursting_data['bursting_summary_data'].get('baseline'),
                    'min': None,
                    'max': None,
                    'weight': 1,
                },
            },
        },
    }
    return network_metric_targets