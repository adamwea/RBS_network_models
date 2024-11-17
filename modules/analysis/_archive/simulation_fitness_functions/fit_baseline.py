import numpy as np

def fit_baseline(net_activity_metrics, **kwargs):
    def fitness_function(baseline, baseline_target, baseline_min=0, baseline_max=700, maxFitness=1000, scale_factor=1.0):
        if baseline_min <= baseline <= baseline_max:
            fitness = 1 + (maxFitness - 1) * (1 - np.exp(-abs(baseline - baseline_target) / (baseline_max - baseline_min))) * scale_factor
        else:
            fitness = maxFitness
        return min(fitness, maxFitness)
    
    try:
        print('Calculating baseline fitness...')
        assert net_activity_metrics['baseline'] is not None, 'Error: baseline is None. Baseline could not be calculated.'
        
        pops = kwargs['pops']
        pops_baseline = pops['baseline_target']
        maxFitness = kwargs['maxFitness']
        baseline = net_activity_metrics['baseline']
        baseline_target = pops_baseline['target']
        baseline_max = pops_baseline['max']
        baseline_min = pops_baseline['min']
        width = pops_baseline['width']
        scale_factor = pops_baseline['scale_factor']

        baselineFitness = fitness_function(baseline, baseline_target, baseline_min, baseline_max, maxFitness, scale_factor=scale_factor)

        print('baseline: %.3f, Fitness: %.3f' % (baseline, baselineFitness)) 
        return {'Value': baseline, 'Fit': baselineFitness}

    except Exception as e:
        print(f'Error calculating baseline fitness.')
        print(f'Error: {e}')
        maxFitness = kwargs['maxFitness']
        return {'Value': None, 'Fit': maxFitness}
