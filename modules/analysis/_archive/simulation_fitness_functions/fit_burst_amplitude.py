import numpy as np
from scipy import stats

def fit_big_burst_amplitude(net_activity_metrics, **kwargs):
    try:
        print('Calculating big burst amplitude fitness...')
        pops = kwargs['pops']
        cutoff = pops['big-small_cutoff']
        burstPeakValues = net_activity_metrics['burstPeakValues']
        big_bursts = [value for value in burstPeakValues if value > cutoff]
        assert len(big_bursts) > 0, 'Error: No big bursts found.'

        target = pops['big_burst_target']['target']
        burst_max = pops['big_burst_target']['max']
        burst_min = pops['big_burst_target']['min']
        width = pops['big_burst_target']['width']
        maxFitness = kwargs['maxFitness']

        fitness_values = [min(np.exp(abs(target - value) / width), maxFitness) if burst_min <= value <= burst_max else maxFitness for value in big_bursts]
        Big_BurstVal_fitness = np.mean(fitness_values)
        big_val = np.nanmean(big_bursts)

        stdev, kurtosis, skewness = calc_distribution_metrics(big_bursts, pops['big_burst_target'], maxFitness)
        fitness = np.nanmean([Big_BurstVal_fitness, stdev['fitness'], kurtosis['fitness'], skewness['fitness']])

        print_summary_metrics(big_bursts, big_val, Big_BurstVal_fitness, stdev, kurtosis, skewness, fitness, 'Big Burst')
        return {'Value': big_val, 'Fit': fitness, 'features': {**stdev, **kurtosis, **skewness}}

    except Exception as e:
        print(f'Error calculating big burst amplitude fitness: {e}')
        return {'Value': None, 'Fit': kwargs['maxFitness']}

def fit_small_burst_amplitude(net_activity_metrics, **kwargs):
    try:
        print('Calculating small burst amplitude fitness...')
        pops = kwargs['pops']
        cutoff = pops['big-small_cutoff']
        burstPeakValues = net_activity_metrics['burstPeakValues']
        small_bursts = [value for value in burstPeakValues if value <= cutoff]
        assert len(small_bursts) > 0, 'Error: No small bursts found.'

        target = pops['small_burst_target']['target']
        burst_max = pops['small_burst_target']['max']
        burst_min = pops['small_burst_target']['min']
        width = pops['small_burst_target']['width']
        maxFitness = kwargs['maxFitness']

        fitness_values = [min(np.exp(abs(target - value) / width), maxFitness) if burst_min <= value <= burst_max else maxFitness for value in small_bursts]
        Small_BurstVal_fitness = np.mean(fitness_values)
        small_val = np.nanmean(small_bursts)

        stdev, kurtosis, skewness = calc_distribution_metrics(small_bursts, pops['small_burst_target'], maxFitness)
        fitness = np.nanmean([Small_BurstVal_fitness, stdev['fitness'], kurtosis['fitness'], skewness['fitness']])

        print_summary_metrics(small_bursts, small_val, Small_BurstVal_fitness, stdev, kurtosis, skewness, fitness, 'Small Burst')
        return {'Value': small_val, 'Fit': fitness, 'features': {**stdev, **kurtosis, **skewness}}

    except Exception as e:
        print(f'Error calculating small burst amplitude fitness: {e}')
        return {'Value': None, 'Fit': kwargs['maxFitness']}

def fit_bimodal_burst_amplitude(net_activity_metrics, **kwargs):
    try:
        print('Calculating bimodal burst amplitude fitness...')
        pops = kwargs['pops']
        cutoff = pops['big-small_cutoff']
        burstPeakValues = net_activity_metrics['burstPeakValues']
        big_bursts = [value for value in burstPeakValues if value > cutoff]
        small_bursts = [value for value in burstPeakValues if value <= cutoff]
        assert len(big_bursts) > 0 and len(small_bursts) > 0, 'Error: No big or small bursts found.'

        actual_ratio = len(big_bursts) / len(small_bursts)
        desired_ratio = pops['bimodal_burst_target']['target']
        maxFitness = kwargs['maxFitness']
        ratio_fitness = min(np.exp(abs(desired_ratio - actual_ratio)), maxFitness)

        mean_value = np.mean(np.nan_to_num(burstPeakValues))
        stdev, kurtosis, skewness = calc_distribution_metrics(burstPeakValues, pops['bimodal_burst_target'], maxFitness)
        fitness = np.nanmean([ratio_fitness, stdev['fitness'], kurtosis['fitness'], skewness['fitness']])

        print(f'Bimodal Burst Ratio (big:small): {actual_ratio:.3f}, Fitness: {ratio_fitness:.3f}')
        print_summary_metrics(burstPeakValues, mean_value, ratio_fitness, stdev, kurtosis, skewness, fitness, 'Bimodal Burst')
        return {'Value': actual_ratio, 'Fit': fitness, 'Features': {**stdev, **kurtosis, **skewness}}

    except Exception as e:
        print(f'Error calculating bimodal burst amplitude fitness: {e}')
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
