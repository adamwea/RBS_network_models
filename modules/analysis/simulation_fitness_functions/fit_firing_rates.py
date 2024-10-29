import numpy as np
from scipy import stats

def fit_E_firing_rate(neuron_metrics, **kwargs):
    try:
        assert neuron_metrics is not None, 'neuron_metrics must be specified'
        print('Calculating excitatory firing rate fitness...')
        
        pops = kwargs['pops']
        pops_rate = pops['E_rate_target']
        E_FRs = list(neuron_metrics['E_average_firing_rates'].values())
        maxFitness = kwargs['maxFitness']
        target = pops_rate['target']
        min_FR = pops_rate['min']
        max_FR = pops_rate['max']
        width = pops_rate['width']

        # Fitness calculation for excitatory firing rate
        popFitness = [
            min(np.exp(abs(target - FR) / width), maxFitness)
            if min_FR <= FR <= max_FR else maxFitness
            for FR in E_FRs
        ]
        E_rate_fitness = np.mean(popFitness)
        E_rate_mean = np.nanmean(E_FRs)

        # Additional metrics (stdev, kurtosis, skewness)
        stdev, kurtosis, skewness = calc_distribution_metrics(E_FRs, pops_rate, maxFitness)

        # Prioritize fitness metrics
        if E_rate_fitness == maxFitness:
            stdev['fitness'], kurtosis['fitness'], skewness['fitness'] = maxFitness, maxFitness, maxFitness

        # Combine fitness metrics
        fitness = np.nanmean([E_rate_fitness, kurtosis['fitness'], skewness['fitness']])

        print_summary_metrics(E_FRs, E_rate_mean, E_rate_fitness, stdev, kurtosis, skewness, fitness)
        return {'Value': E_rate_mean, 'Fit': fitness, 'Features': {'E_rate_mean': E_rate_mean, 'E_rate_fitness': E_rate_fitness, **stdev, **kurtosis, **skewness}}

    except Exception as e:
        print(f'Error calculating excitatory firing rate fitness: {e}')
        return {'Value': None, 'Fit': kwargs['maxFitness']}

def fit_I_firing_rate(neuron_metrics, **kwargs):
    try:
        assert neuron_metrics is not None, 'neuron_metrics must be specified'
        print('Calculating inhibitory firing rate fitness...')

        pops = kwargs['pops']
        pops_rate = pops['I_rate_target']
        I_FRs = list(neuron_metrics['I_average_firing_rates'].values())
        maxFitness = kwargs['maxFitness']
        target = pops_rate['target']
        min_FR = pops_rate['min']
        max_FR = pops_rate['max']
        width = pops_rate['width']

        # Fitness calculation for inhibitory firing rate
        popFitness = [
            min(np.exp(abs(target - FR) / width), maxFitness)
            if min_FR <= FR <= max_FR else maxFitness
            for FR in I_FRs
        ]
        I_rate_fitness = np.mean(popFitness)
        I_rate_mean = np.nanmean(I_FRs)

        # Additional metrics (stdev, kurtosis, skewness)
        stdev, kurtosis, skewness = calc_distribution_metrics(I_FRs, pops_rate, maxFitness)

        # Prioritize fitness metrics
        if I_rate_fitness == maxFitness:
            stdev['fitness'], kurtosis['fitness'], skewness['fitness'] = maxFitness, maxFitness, maxFitness

        # Combine fitness metrics
        fitness = np.nanmean([I_rate_fitness, kurtosis['fitness'], skewness['fitness']])

        print_summary_metrics(I_FRs, I_rate_mean, I_rate_fitness, stdev, kurtosis, skewness, fitness)
        return {'Value': I_rate_mean, 'Fit': fitness, 'Features': {'I_rate_mean': I_rate_mean, 'I_rate_fitness': I_rate_fitness, **stdev, **kurtosis, **skewness}}

    except Exception as e:
        print(f'Error calculating inhibitory firing rate fitness: {e}')
        return {'Value': None, 'Fit': kwargs['maxFitness']}

def calc_distribution_metrics(firing_rates, pops_rate, maxFitness):
    # Calculate standard deviation, kurtosis, and skewness fitness
    try:
        stdev = np.nanstd(firing_rates)
        stdev_fitness = min(np.exp(abs(stdev - pops_rate['stdev'])), maxFitness)
    except:
        stdev, stdev_fitness = None, maxFitness

    try:
        kurtosis = stats.kurtosis(np.nan_to_num(firing_rates))
        kurtosis_fitness = min(np.exp(abs(kurtosis - pops_rate['kurtosis'])), maxFitness)
    except:
        kurtosis, kurtosis_fitness = None, maxFitness

    try:
        skewness = stats.skew(np.nan_to_num(firing_rates))
        skew_fitness = min(np.exp(abs(skewness - pops_rate['skew'])), maxFitness)
    except:
        skewness, skew_fitness = None, maxFitness

    return (
        {'Value': stdev, 'fitness': stdev_fitness}, 
        {'Value': kurtosis, 'fitness': kurtosis_fitness}, 
        {'Value': skewness, 'fitness': skew_fitness}
    )

def print_summary_metrics(firing_rates, rate_mean, rate_fitness, stdev, kurtosis, skewness, fitness):
    print(f'Maximum FR: {np.nanmax(firing_rates):.3f}, Minimum FR: {np.nanmin(firing_rates):.3f}')
    print(f'Mean FR: {rate_mean:.3f}, Fitness: {rate_fitness:.3f}')
    print(f'Stdev: {stdev["Value"]:.3f}, Fitness: {stdev["fitness"]:.3f}')
    print(f'Kurtosis: {kurtosis["Value"]:.3f}, Fitness: {kurtosis["fitness"]:.3f}')
    print(f'Skewness: {skewness["Value"]:.3f}, Fitness: {skewness["fitness"]:.3f}')
    print(f'Overall Fitness: {fitness:.3f}')
