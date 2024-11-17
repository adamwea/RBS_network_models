import numpy as np
from scipy import stats

def fit_E_ISI(neuron_metrics, **kwargs):
    try:
        assert neuron_metrics is not None, 'neuron_metrics must be specified'
        print('Calculating excitatory ISI fitness...')
        
        pops = kwargs['pops']
        pops_ISI = pops['E_ISI_target']
        E_ISIs = list(neuron_metrics['E_average_ISIs'].values())
        maxFitness = kwargs['maxFitness']
        target = pops_ISI['target']
        min_ISI = pops_ISI['min']
        max_ISI = pops_ISI['max']
        width = pops_ISI['width']

        # ISI fitness calculation for excitatory neurons
        popFitness = [
            min(np.exp(abs(target - ISI) / width), maxFitness)
            if min_ISI <= ISI <= max_ISI else maxFitness
            for ISI in E_ISIs
        ]
        E_ISI_fitness = np.mean(popFitness)
        E_ISI_mean = np.nanmean(E_ISIs)

        # Additional metrics (stdev, kurtosis, skewness)
        stdev, kurtosis, skewness = calc_distribution_metrics(E_ISIs, pops_ISI, maxFitness)

        # Prioritize fitness metrics
        if E_ISI_fitness == maxFitness:
            stdev['fitness'], kurtosis['fitness'], skewness['fitness'] = maxFitness, maxFitness, maxFitness

        # Combine fitness metrics
        fitness = np.nanmean([E_ISI_fitness, kurtosis['fitness'], skewness['fitness']])

        print_summary_metrics(E_ISIs, E_ISI_mean, E_ISI_fitness, stdev, kurtosis, skewness, fitness)
        return {'Value': E_ISI_mean, 'Fit': fitness, 'Features': {'E_ISI_mean': E_ISI_mean, 'E_ISI_fitness': E_ISI_fitness, **stdev, **kurtosis, **skewness}}

    except Exception as e:
        print(f'Error calculating excitatory ISI fitness: {e}')
        return {'Value': None, 'Fit': kwargs['maxFitness']}

def fit_I_ISI(neuron_metrics, **kwargs):
    try:
        assert neuron_metrics is not None, 'neuron_metrics must be specified'
        print('Calculating inhibitory ISI fitness...')

        pops = kwargs['pops']
        pops_ISI = pops['I_ISI_target']
        I_ISIs = list(neuron_metrics['I_average_ISIs'].values())
        maxFitness = kwargs['maxFitness']
        target = pops_ISI['target']
        min_ISI = pops_ISI['min']
        max_ISI = pops_ISI['max']
        width = pops_ISI['width']

        # ISI fitness calculation for inhibitory neurons
        popFitness = [
            min(np.exp(abs(target - ISI) / width), maxFitness)
            if min_ISI <= ISI <= max_ISI else maxFitness
            for ISI in I_ISIs
        ]
        I_ISI_fitness = np.mean(popFitness)
        I_ISI_mean = np.nanmean(I_ISIs)

        # Additional metrics (stdev, kurtosis, skewness)
        stdev, kurtosis, skewness = calc_distribution_metrics(I_ISIs, pops_ISI, maxFitness)

        # Prioritize fitness metrics
        if I_ISI_fitness == maxFitness:
            stdev['fitness'], kurtosis['fitness'], skewness['fitness'] = maxFitness, maxFitness, maxFitness

        # Combine fitness metrics
        fitness = np.nanmean([I_ISI_fitness, kurtosis['fitness'], skewness['fitness']])

        print_summary_metrics(I_ISIs, I_ISI_mean, I_ISI_fitness, stdev, kurtosis, skewness, fitness)
        return {'Value': I_ISI_mean, 'Fit': fitness, 'Features': {'I_ISI_mean': I_ISI_mean, 'I_ISI_fitness': I_ISI_fitness, **stdev, **kurtosis, **skewness}}

    except Exception as e:
        print(f'Error calculating inhibitory ISI fitness: {e}')
        return {'Value': None, 'Fit': kwargs['maxFitness']}

def calc_distribution_metrics(ISIs, pops_ISI, maxFitness):
    # Calculate standard deviation, kurtosis, and skewness fitness
    try:
        stdev = np.nanstd(ISIs)
        stdev_fitness = min(np.exp(abs(stdev - pops_ISI['stdev'])), maxFitness)
    except:
        stdev, stdev_fitness = None, maxFitness

    try:
        kurtosis = stats.kurtosis(np.nan_to_num(ISIs))
        kurtosis_fitness = min(np.exp(abs(kurtosis - pops_ISI['kurtosis'])), maxFitness)
    except:
        kurtosis, kurtosis_fitness = None, maxFitness

    try:
        skewness = stats.skew(np.nan_to_num(ISIs))
        skew_fitness = min(np.exp(abs(skewness - pops_ISI['skew'])), maxFitness)
    except:
        skewness, skew_fitness = None, maxFitness

    return (
        {'Value': stdev, 'fitness': stdev_fitness}, 
        {'Value': kurtosis, 'fitness': kurtosis_fitness}, 
        {'Value': skewness, 'fitness': skew_fitness}
    )

def print_summary_metrics(ISIs, ISI_mean, ISI_fitness, stdev, kurtosis, skewness, fitness):
    print(f'Maximum ISI: {np.nanmax(ISIs):.3f}, Minimum ISI: {np.nanmin(ISIs):.3f}')
    print(f'Mean ISI: {ISI_mean:.3f}, Fitness: {ISI_fitness:.3f}')
    print(f'Stdev: {stdev["Value"]:.3f}, Fitness: {stdev["fitness"]:.3f}')
    print(f'Kurtosis: {kurtosis["Value"]:.3f}, Fitness: {kurtosis["fitness"]:.3f}')
    print(f'Skewness: {skewness["Value"]:.3f}, Fitness: {skewness["fitness"]:.3f}')
    print(f'Overall Fitness: {fitness:.3f}')
