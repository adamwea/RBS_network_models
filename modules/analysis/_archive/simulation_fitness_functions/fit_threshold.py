import numpy as np

def fit_threshold(net_activity_metrics, **kwargs):
    def fitness_function(thresh, target, thresh_min=0, thresh_max=700, maxFitness=1000, scale_factor=1.0):
        if thresh_min <= thresh <= thresh_max:
            fitness = 1 + (maxFitness - 1) * (1 - np.exp(-abs(thresh - target) / (thresh_max - thresh_min))) * scale_factor
        else:
            fitness = maxFitness
        return min(fitness, maxFitness)

    try:
        pops = kwargs['pops']
        maxFitness = kwargs['maxFitness']
        thresh = net_activity_metrics['threshold']
        thresh_target = pops['threshold_target']
        scale_factor = thresh_target['scale_factor']
        max_thresh = thresh_target['max']
        min_thresh = thresh_target['min']

        thresh_fit = fitness_function(thresh, thresh_target['target'], min_thresh, max_thresh, maxFitness, scale_factor=scale_factor)

        print('Thresh: %.3f, Fitness: %.3f' % (thresh, thresh_fit))
        return {'Value': thresh, 'Fit': thresh_fit}

    except Exception as e:
        print(f'Error calculating thresh fitness.')
        print(f'Error: {e}')
        maxFitness = kwargs['maxFitness']
        print('Thresh: '+f'{None} fit={maxFitness:.3f}')
        return {'Value': None, 'Fit': maxFitness}