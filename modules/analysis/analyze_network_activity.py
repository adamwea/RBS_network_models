#from modules.analysis_functions.network_activity_analysis import measure_network_activity
import numpy as np
from scipy.signal import convolve, find_peaks
from scipy.stats import norm
import matplotlib.pyplot as plt
import submodules.MEA_Analysis.IPNAnalysis.helper_functions as helper

'''deprecated fuctions'''
def convolve_network_activity(rasterData, binSize=None, gaussianSigma=None, thresholdBurst=None, min_peak_distance=None):

    conv_params = init_convolution_params(binSize, gaussianSigma, thresholdBurst, min_peak_distance)
    binSize = conv_params['binSize']
    gaussianSigma = conv_params['gaussianSigma']
    thresholdBurst = conv_params['thresholdBurst']
    min_peak_distance = conv_params['min_peak_distance']
    
    relativeSpikeTimes = np.array(rasterData['spkTimes'])
    relativeSpikeTimes -= relativeSpikeTimes[0]
    
    timeVector = np.arange(0, max(relativeSpikeTimes) + binSize, binSize)
    binnedTimes, _ = np.histogram(relativeSpikeTimes, bins=timeVector)
    binnedTimes = np.append(binnedTimes, 0)

    kernelRange = np.arange(-3*gaussianSigma, 3*gaussianSigma + binSize, binSize)
    kernel = norm.pdf(kernelRange, 0, gaussianSigma)
    kernel *= binSize
    spike_counts_at_time = convolve(binnedTimes, kernel, mode='same') / binSize

    rmsSpikeCounts = np.sqrt(np.mean(spike_counts_at_time**2))
    peaks, properties = find_peaks(spike_counts_at_time, height=thresholdBurst * rmsSpikeCounts, distance=min_peak_distance)

    burstPeakTimes = timeVector[peaks]
    burstPeakValues = properties['peak_heights']

    return burstPeakTimes, burstPeakValues, spike_counts_at_time, timeVector, rmsSpikeCounts

'''Initialization Functions'''
def init_convolution_params(binSize=None, gaussianSigma=None, thresholdBurst=None, min_peak_distance=None):
    try:
        from simulate._config_files.convolution_params import conv_params
        conv_params_temp = conv_params
        conv_params_temp['binSize'] = binSize if binSize is not None else conv_params['binSize']
        conv_params_temp['gaussianSigma'] = gaussianSigma if gaussianSigma is not None else conv_params['gaussianSigma']
        conv_params_temp['thresholdBurst'] = thresholdBurst if thresholdBurst is not None else conv_params['thresholdBurst']
        conv_params_temp['min_peak_distance'] = min_peak_distance if min_peak_distance is not None else conv_params['min_peak_distance']
    except:
        conv_params_temp = {
            'binSize': 0.1 if binSize is None else binSize,
            'gaussianSigma': 0.15 if gaussianSigma is None else gaussianSigma,
            'thresholdBurst': 1.0 if thresholdBurst is None else thresholdBurst,
            'min_peak_distance':1 if min_peak_distance is None else min_peak_distance,
        }
    
    conv_params = conv_params_temp
    return conv_params

def init_burst_analysis_params(isi_threshold=None):
    try:
        from simulate._config_files.burst_analysis_params import burst_analysis_params
        burst_analysis_params_temp = burst_analysis_params
        burst_analysis_params_temp['isi_threshold'] = isi_threshold if isi_threshold is not None else burst_analysis_params['isi_threshold']
    except:
        burst_analysis_params_temp = {
            'isi_threshold': 0.1 if isi_threshold is None else isi_threshold,
        }
    
    burst_analysis_params = burst_analysis_params_temp
    return burst_analysis_params

def init_network_data_dict():
    empty_network_data = {
        #General Data
        'source': None, # 'simulated' or 'experimental'
        'timeVector': None,
        
        #Simulated Data
        'simulated_data': {
            'soma_voltage': None,
            'E_Gids': None,
            'I_Gids': None,
            'MeanFireRate_E': None,
            'CoVFireRate_E': None,
            'MeanFireRate_I': None,
            'CoVFireRate_I': None,
            'MeanISI_E': None,
            'MeanISI_I': None,
            'CoV_ISI_E': None,
            'CoV_ISI_I': None,
            'spiking_data_by_unit': None, 
        },
        
        #Spiking Data
        'spiking_data': {
            'spike_times': None,
            'spiking_summary_data': {
                #'spike_times': None,
                'MeanFireRate': None,
                'CoVFireRate': None,
                'MeanISI': None,
                'CoV_ISI': None,         
            },
            'spiking_times_by_unit': None,
            'spiking_data_by_unit': None,
        },
        
        #Bursting Data
        'bursting_data': {
            'bursting_summary_data': {
                'baseline': None,
                'MeanWithinBurstISI': None,
                'CovWithinBurstISI': None,
                'MeanOutsideBurstISI': None,
                'CoVOutsideBurstISI': None,
                'MeanNetworkISI': None,
                'CoVNetworkISI': None,
                'NumUnits': None,
                'Number_Bursts': None,
                'mean_IBI': None,
                'cov_IBI': None,
                'mean_Burst_Peak': None,
                'cov_Burst_Peak': None,
                'fano_factor': None,
            },
            'bursting_data_by_unit': None,
        }
    }
    global network_data
    network_data = empty_network_data
    return empty_network_data
    
    ## Keep for reference
    
            # return {
        #     # 'burstPeakValues': None,
        #     # 'IBIs': None,
        #     # 'baseline': None,
        #     # 'peakFreq': None,
        #     # 'firingRate': None,
        #     # 'burstPeakTimes': None,
        #     # 'timeVector': None,
        #     # 'threshold': None,
            
        #     'Number_Bursts': None,
        #     'mean_IBI': None,
        #     'cov_IBI': None,
        #     'mean_Burst_Peak': None,
        #     'cov_Burst_Peak': None,
        #     'fano_factor': None,
        #     'MeanWithinBurstISI': None,
        #     'CoVWithinBurstISI': None,
        #     'MeanOutsideBurstISI': None,
        #     'CoVOutsideBurstISI': None,
        #     'MeanNetworkISI': None,
        #     'CoVNetworkISI': None,
        #     'NumUnits': None,
        #     #'fileName': None
        # }

'''Spiking Analysis Functions'''
def get_spiking_stats_by_unit(spike_times_by_unit):
    spiking_data_by_unit = {}
    # if 'E_Gids' in network_data['simulated_data']:
    #     E_Gids = network_data['simulated_data']['E_Gids']
    for unit, spike_times in spike_times_by_unit.items():
        isi = np.diff(spike_times)
        meanISI = np.mean(isi)
        #CoV_ISI = np.std(isi) / meanISI
        CoV_ISI = np.cov(isi) 
        spiking_data_by_unit[unit] = {
            #'pop': 'E' if unit in E_Gids else 'I',
            'meanISI': meanISI,
            'CoV_ISI': CoV_ISI,
            'spike_times': spike_times,
        }
    return spiking_data_by_unit

def get_spiking_summary_data(spiking_data_by_unit):
    meanISI_E = np.mean([unit_stats['meanISI'] for unit_stats in spiking_data_by_unit.values() if unit_stats['meanISI'] is not None])
    CoV_ISI_E = np.mean([unit_stats['CoV_ISI'] for unit_stats in spiking_data_by_unit.values() if unit_stats['CoV_ISI'] is not None])
    return meanISI_E, CoV_ISI_E

'''Bursting Analysis Functions'''
def convolve_signal_get_baseline(spike_times, binSize=None, gaussianSigma=None):  
    #init convolution parameters
    conv_params = init_convolution_params(binSize, gaussianSigma)
    binSize = conv_params['binSize']
    gaussianSigma = conv_params['gaussianSigma']
    
    #convolve signal and get baseline
    timeVector = np.arange(0, max(spike_times) + binSize, binSize)
    binnedTimes, _ = np.histogram(spike_times, bins=timeVector)
    binnedTimes = np.append(binnedTimes, 0)

    kernelRange = np.arange(-3*gaussianSigma, 3*gaussianSigma + binSize, binSize)
    kernel = norm.pdf(kernelRange, 0, gaussianSigma)
    kernel *= binSize
    spike_counts_at_time = convolve(binnedTimes, kernel, mode='same') / binSize
    
    baseline = np.mean(spike_counts_at_time)

    return baseline

''' Use Functions from MEA Analysis Repository to do Bursting Analysis'''
def analyze_convolved_spiking_signal(spike_times, min_peak_distance, binSize, gaussianSigma, thresholdBurst):
    fig, ax = plt.subplots()
    #spike_times = {i: rasterData['spkt'][rasterData['spkid'] == i] for i in np.unique(rasterData['spkid'])} #plot_network_activity expects spike_times as dictionary
    ax, network_data = helper.plot_network_activity(ax, spike_times, min_peak_distance=min_peak_distance, binSize=binSize, gaussianSigma=gaussianSigma, thresholdBurst=thresholdBurst)
    network_data['bursting_data']['bursting_summary_data']['baseline'] = convolve_signal_get_baseline(spike_times)
    return network_data

def analyze_bursting_activity(spike_times, isi_threshold):
    '''Slightly modified version of the code from mea_analysis_pipeline.py'''
    #spike_times = {int(i): rasterData['spkt'][rasterData['spkid'] == i] for i in np.unique(rasterData['spkid'])}
    burst_statistics = helper.detect_bursts_statistics(spike_times, isi_threshold=isi_threshold)
    bursts_by_unit = [unit_stats['bursts'] for unit_stats in burst_statistics.values()]
    
    all_isis_within_bursts = np.concatenate([stats['isis_within_bursts'] for stats in burst_statistics.values() if stats['isis_within_bursts'].size > 0])
    all_isis_outside_bursts = np.concatenate([stats['isis_outside_bursts'] for stats in burst_statistics.values() if stats['isis_outside_bursts'].size > 0])
    all_isis = np.concatenate([stats['isis_all'] for stats in burst_statistics.values() if stats['isis_all'].size > 0])

    mean_isi_within_combined = np.mean(all_isis_within_bursts) if all_isis_within_bursts.size > 0 else np.nan
    cov_isi_within_combined = np.cov(all_isis_within_bursts) if all_isis_within_bursts.size > 0 else np.nan

    mean_isi_outside_combined = np.mean(all_isis_outside_bursts) if all_isis_outside_bursts.size > 0 else np.nan
    cov_isi_outside_combined = np.cov(all_isis_outside_bursts) if all_isis_outside_bursts.size > 0 else np.nan

    mean_isi_all_combined = np.mean(all_isis) if all_isis.size > 0 else np.nan
    cov_isi_all_combined = np.cov(all_isis) if all_isis.size > 0 else np.nan

    bursting_summary_data = {}
    bursting_summary_data['MeanWithinBurstISI'] = mean_isi_within_combined
    bursting_summary_data['CoVWithinBurstISI'] = cov_isi_within_combined
    bursting_summary_data['MeanOutsideBurstISI'] = mean_isi_outside_combined   
    bursting_summary_data['CoVOutsideBurstISI'] = cov_isi_outside_combined
    bursting_summary_data['MeanNetworkISI'] = mean_isi_all_combined
    bursting_summary_data['CoVNetworkISI'] = cov_isi_all_combined
    bursting_summary_data['NumUnits'] = len(spike_times)
    bursting_summary_data['Number_Bursts'] = sum(len(unit_stats['bursts']) for unit_stats in burst_statistics.values())
    bursting_summary_data['mean_IBI'] = np.mean(all_isis) if all_isis.size > 0 else np.nan
    bursting_summary_data['cov_IBI'] = np.cov(all_isis) if all_isis.size > 0 else np.nan
    
    bursting_data = {
        'bursting_summary_data': bursting_summary_data,
        'bursting_data_by_unit': bursts_by_unit,
    }

    return bursting_data

def get_network_data(spike_times):
    
    #Add Spiking Data to Network Data - #TODO - Resume development from here
    spiking_data_by_unit = get_spiking_stats_by_unit(spike_times)
    spiking_data_summary = get_spiking_summary_data(spiking_data_by_unit)
    
    #network_data
    
    
    
    
    
    #Add Bursting Data to Network Data
    #convolved signal burst analysis 
    conv_params = init_convolution_params()
    binSize = conv_params['binSize']
    gaussianSigma = conv_params['gaussianSigma']
    thresholdBurst = conv_params['thresholdBurst']
    min_peak_distance = conv_params['min_peak_distance']
    
    #unit-wise burst analysis
    burst_analysis_params = init_burst_analysis_params()
    isi_threshold = burst_analysis_params['isi_threshold']
    
    convolved_data = analyze_convolved_spiking_signal(spike_times, min_peak_distance, binSize, gaussianSigma, thresholdBurst)
    bursting_data = analyze_bursting_activity(spike_times, isi_threshold)
    
    #add convolved data to the bursting summary data
    bursting_summary_data = bursting_data['bursting_summary_data']
    for key in convolved_data.keys():
        bursting_summary_data[key] = convolved_data[key]
    bursting_data['bursting_summary_data'] = bursting_summary_data
    
    # Verify that any single-element array is converted to a scalar
    for key in bursting_data['bursting_summary_data'].keys():
        value = bursting_data['bursting_summary_data'][key]
        if isinstance(value, np.ndarray) and value.size == 1:
            bursting_data['bursting_summary_data'][key] = value.item()  # .item() fetches the scalar value directly
    
    network_data['bursting_data'] = bursting_data
    # network_data = {
    #     'bursting_data': bursting_data,
    # }

    #return network_data

def _analyze_network_activity(spike_times):
    #try: network_data = get_network_data(spike_times)
    try: get_network_data(spike_times)
    except Exception as e:
        print(f'Error calculating network activity metrics: {e}')
        pass
    
    print('Network Activity Metrics Calculated')
    #return network_data


'''Main Functions to be used - Warpper Functions for Experimental and Simulated Cases'''
def calculate_simulated_network_activity_metrics(spike_times_by_unit):
    #Calculate CoVFiringRate_E, CoVFiringRate_I, CoV_ISI_E, CoV_ISI_I, MeanISI_E, MeanISI_I by taking advantage of E_Gids, I_Gids, and spiking_data_by_unit
    network_data['simulated_data']['spiking_data_by_unit'] = {}
    for unit, spike_times in spike_times_by_unit.items():
        if len(spike_times) < 2:
            meanISI = np.nan
            CoV_ISI = np.nan
        else:
            isi = np.diff(spike_times)
            meanISI = np.mean(isi)
            CoV_ISI = np.cov(isi)
        
        fr = len(spike_times) / network_data['timeVector'][-1]
        
        network_data['simulated_data']['spiking_data_by_unit'][unit] = {
            'FireRate': fr,
            'meanISI': meanISI,
            'CoV_ISI': CoV_ISI
        }
        
        if unit in network_data['simulated_data']['E_Gids']:
            network_data['simulated_data']['spiking_data_by_unit'][unit]['pop'] = 'E'
        elif unit in network_data['simulated_data']['I_Gids']:
            network_data['simulated_data']['spiking_data_by_unit'][unit]['pop'] = 'I'
        else:
            raise ValueError(f'Unit {unit} not found in E_Gids or I_Gids')
    
    #Calculate CoVFireRate_E, CoVFireRate_I
    E_Gids = network_data['simulated_data']['E_Gids']
    I_Gids = network_data['simulated_data']['I_Gids']
    
    E_CoV = np.nanmean([network_data['simulated_data']['spiking_data_by_unit'][unit]['CoV_ISI'] for unit in E_Gids])
    I_CoV = np.nanmean([network_data['simulated_data']['spiking_data_by_unit'][unit]['CoV_ISI'] for unit in I_Gids])
    MeanISI_E = np.nanmean([network_data['simulated_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in E_Gids])
    MeanISI_I = np.nanmean([network_data['simulated_data']['spiking_data_by_unit'][unit]['meanISI'] for unit in I_Gids])
    CoVFiringRate_E = np.cov([network_data['simulated_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in E_Gids])
    CoVFiringRate_I = np.cov([network_data['simulated_data']['spiking_data_by_unit'][unit]['FireRate'] for unit in I_Gids])
    
    #add to network_data
    network_data['simulated_data']['CoV_ISI_E'] = E_CoV
    network_data['simulated_data']['CoV_ISI_I'] = I_CoV
    network_data['simulated_data']['MeanISI_E'] = MeanISI_E
    network_data['simulated_data']['MeanISI_I'] = MeanISI_I
    network_data['simulated_data']['CoVFireRate_E'] = CoVFiringRate_E
    network_data['simulated_data']['CoVFireRate_I'] = CoVFiringRate_I
            

def get_simulated_network_activity_metrics(simData=None, popData=None, **kwargs):
    #this part should be useful for fitness during simulation
    if simData is None:
        try: 
            simData = sim.simData
            print('Using simData from netpyne.sim.simData')
        except:
            print('No simData provided or found in netpyne.sim.simData')
            return None

    #initialize network_data
    #network_data = init_network_data_dict()
    network_data = init_network_data_dict()
    rasterData = simData.copy()
    
    #convert time to seconds - get initially available data
    spike_times = np.array(rasterData['spkt']) / 1000
    timeVector = np.array(rasterData['t']) / 1000
    spike_times_by_unit = {int(i): spike_times[rasterData['spkid'] == i] for i in np.unique(rasterData['spkid'])} #mea_analysis_pipeline.py expects spike_times as dictionary    
    
    #add immediately available data to network_data
    network_data['source'] = 'simulated'
    network_data['timeVector'] = timeVector
    network_data['spiking_data']['spike_times'] = spike_times
    network_data['simulated_data']['soma_voltage'] = rasterData['soma_voltage']
    network_data['spiking_data']['spiking_times_by_unit'] = spike_times_by_unit

    #add uniquely simulated data to network_data
    network_data['simulated_data']['E_Gids'] = popData['E']['cellGids']
    network_data['simulated_data']['I_Gids'] = popData['I']['cellGids']
    network_data['simulated_data']['MeanFireRate_E'] = rasterData['popRates']['E']
    network_data['simulated_data']['MeanFireRate_I'] = rasterData['popRates']['I']
    
    calculate_simulated_network_activity_metrics(spike_times_by_unit)
    
    #derive low-hanging fruit network activity metrics from simulated data
    network_data['spiking_data']['spiking_summary_data']['MeanFireRate'] = (rasterData['popRates']['E'] + rasterData['popRates']['I'])/2
    
    return _analyze_network_activity(spike_times_by_unit)

def get_experimental_network_activity_metrics(experimentalData):
    net_activity_metrics = {}
    implemented = False
    assert implemented, 'Experimental network activity metrics not implemented yet.'
    rasterData = experimentalData.copy()

    return _analyze_network_activity(rasterData)
