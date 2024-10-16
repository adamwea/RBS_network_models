import numpy as np
from scipy import stats

def fit_IBI(net_activity_metrics, **kwargs):
    try:
        print('Calculating IBI fitness...')
        IBIs = net_activity_metrics['IBIs']
        assert len(IBIs) > 0, 'Error: No IBIs found.'

        pops_IBI = kwargs['pops']['IBI_target']
        maxFitness = kwargs['maxFitness']
        target = pops_IBI['target']
        width = pops_IBI['width']
        min_IBI = pops_IBI['min']
        max_IBI = pops_IBI['max']

        fitness_values = [min(np.exp(abs(target - value) / width), maxFitness) if min_IBI <= value <= max_IBI else maxFitness for value in IBIs]
        IBI_fitness = np.mean(fitness_values)
        avg_IBI = np.nanmean(IBIs)

        stdev, kurtosis, skewness = calc_distribution_metrics(IBIs, pops_IBI, maxFitness)
        fitness = np.nanmean([IBI_fitness, stdev['fitness'], kurtosis['fitness'], skewness['fitness']])

        print_summary_metrics(IBIs, avg_IBI, IBI_fitness, stdev, kurtosis, skewness, fitness, 'IBI')
        return {'Value': avg_IBI, 'Fit': fitness, 'Features': {**stdev, **kurtosis, **skewness}}

    except Exception as e:
        print(f'Error calculating IBI fitness: {e}')
        return {'Value': None, 'Fit': kwargs['maxFitness']}

def calc_distribution_metrics(values, target_params, maxFitness):
    # Calculate standard deviation, kurtosis, and skewness fitness
    try:
        stdev = np.nanstd(values)
        stdev_fitness = min(np.exp(abs(stdev - target_params['stdev'])), maxFitness)
    except:
        stdev, stdev_fitness = None, maxFitness

    try:
        kurtosis = stats.kurtosis(np.nan_to_num(values))
        kurtosis_fitness = min(np.exp(abs(kurtosis - target_params['kurtosis'])), maxFitness)
    except:
        kurtosis, kurtosis_fitness = None, maxFitness

    try:
        skewness = stats.skew(np.nan_to_num(values))
        skew_fitness = min(np.exp(abs(skewness - target_params['skew'])), maxFitness)
    except:
        skewness, skew_fitness = None, maxFitness

    return (
        {'Value': stdev, 'fitness': stdev_fitness}, 
        {'Value': kurtosis, 'fitness': kurtosis_fitness}, 
        {'Value': skewness, 'fitness': skew_fitness}
    )

def print_summary_metrics(values, mean_val, main_fitness, stdev, kurtosis, skewness, overall_fitness, label):
    print(f'{label} Max: {np.nanmax(values):.3f}, Min: {np.nanmin(values):.3f}')
    print(f'{label} Mean: {mean_val:.3f}, Main Fitness: {main_fitness:.3f}')
    print(f'{label} Stdev: {stdev["Value"]:.3f}, Stdev Fitness: {stdev["fitness"]:.3f}')
    print(f'{label} Kurtosis: {kurtosis["Value"]:.3f}, Kurtosis Fitness: {kurtosis["fitness"]:.3f}')
    print(f'{label} Skewness: {skewness["Value"]:.3f}, Skewness Fitness: {skewness["fitness"]:.3f}')
    print(f'{label} Overall Fitness: {overall_fitness:.3f}')
