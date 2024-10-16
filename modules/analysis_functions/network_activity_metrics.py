from simulation_analysis_functions.network_activity_analysis import analyze_network_activity
import numpy as np

def get_simulated_network_activity_metrics(simData=None):
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
    try:
        conv_params = {
            'binSize': 0.1,
            'gaussianSigma': 0.15,
            'thresholdBurst': 1.0,
            'min_peak_distance': 1
        }
        net_metrics = analyze_network_activity(rasterData, conv_params=conv_params)

    except Exception as e:
        print(f'Error calculating network activity metrics: {e}')
        return {
            'burstPeakValues': None,
            'IBIs': None,
            'baseline': None,
            'peakFreq': None,
            'firingRate': None,
            'burstPeakTimes': None,
            'timeVector': None,
            'threshold': None,
        }

    return net_metrics