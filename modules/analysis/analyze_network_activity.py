#from modules.analysis_functions.network_activity_analysis import measure_network_activity
from modules.analysis.analyze_burst_activity import analyze_burst_activity

import numpy as np

def get_simulated_network_activity_metrics(simData=None):
    #this part should be useful for fitness during simulation
    if simData is None:
        try: 
            simData = sim.simData
            print('Using simData from netpyne.sim.simData')
        except:
            print('No simData provided or found in netpyne.sim.simData')
            return None

    net_activity_metrics = {}
    rasterData = simData.copy()
    rasterData['spkt'] = np.array(rasterData['spkt']) / 1000

    return _calculate_network_activity_metrics(rasterData)

def get_experimental_network_activity_metrics(experimentalData):
    net_activity_metrics = {}
    implemented = False
    assert implemented, 'Experimental network activity metrics not implemented yet.'
    rasterData = experimentalData.copy()

    return _calculate_network_activity_metrics(rasterData)

def _calculate_network_activity_metrics(rasterData):
    conv_params = None
    try:
        from simulate._config_files.convolution_params import conv_params
    except:
        conv_params = {
            'binSize': 0.1,
            'gaussianSigma': 0.15,
            'thresholdBurst': 1.0,
            'min_peak_distance': 1
        }
    assert conv_params is not None, 'Convolution parameters not found.'
    
    try:
        #spike_times = np.array(rasterData['spkt'])
        spike_times = rasterData['spkt']
        #write spike_times as dictionary
        spike_times = {i: rasterData['spkt'][rasterData['spkid'] == i] for i in np.unique(rasterData['spkid'])}
        network_activity_metrics = analyze_burst_activity(spike_times)#, conv_params=conv_params)
    except Exception as e:
        print(f'Error calculating network activity metrics: {e}')
        return {
            # 'burstPeakValues': None,
            # 'IBIs': None,
            # 'baseline': None,
            # 'peakFreq': None,
            # 'firingRate': None,
            # 'burstPeakTimes': None,
            # 'timeVector': None,
            # 'threshold': None,
            
            'Number_Bursts': None,
            'mean_IBI': None,
            'cov_IBI': None,
            'mean_Burst_Peak': None,
            'cov_Burst_Peak': None,
            'fano_factor': None,
            'MeanWithinBurstISI': None,
            'CoVWithinBurstISI': None,
            'MeanOutsideBurstISI': None,
            'CoVOutsideBurstISI': None,
            'MeanNetworkISI': None,
            'CoVNetworkISI': None,
            'NumUnits': None,
            #'fileName': None
        }

    return net_metrics