import numpy as np

def fit_burst_frequency(net_activity_metrics, **kwargs):
    def fitness_function(burst_peak_frequency, target, burst_freq_min=0, burst_freq_max=700, maxFitness=1000, scale_factor=1.0):
        if burst_freq_min < burst_peak_frequency <= burst_freq_max:
            fitness = 1 + (maxFitness - 1) * (1 - np.exp(-abs(burst_peak_frequency - target) / (burst_freq_max - burst_freq_min))) * scale_factor
        else:
            fitness = maxFitness
        return min(fitness, maxFitness)

    try:
        burst_peak_frequency = net_activity_metrics['peakFreq']
        pops = kwargs['pops']
        pops_frequency = pops['burst_frequency_target']
        maxFitness = kwargs['maxFitness']
        scale_factor = pops_frequency['scale_factor']
        max_freq = pops_frequency['max']
        min_freq = pops_frequency['min']
        
        burst_peak_frequency_fitness = fitness_function(burst_peak_frequency, pops_frequency['target'], min_freq, max_freq, maxFitness, scale_factor=scale_factor)

        print('Burst Frequency: %.3f, Fitness: %.3f' % (burst_peak_frequency, burst_peak_frequency_fitness))
        return {'Value': burst_peak_frequency, 'Fit': burst_peak_frequency_fitness}

    except Exception as e:
        print(f'Error calculating burst peak frequency fitness.')
        print(f'Error: {e}')
        maxFitness = kwargs['maxFitness']
        print('Burst Frequency: '+f'{None} fit={maxFitness:.3f}')
        return {'Value': None, 'Fit': maxFitness}
