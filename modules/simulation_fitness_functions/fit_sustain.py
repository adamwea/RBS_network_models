import numpy as np

def fit_sustain(net_activity_metrics, **kwargs):
    try:
        print('Calculating sustain fitness...')
        pops = kwargs['pops']
        maxFitness = kwargs['maxFitness']
        sustained_osci = net_activity_metrics['sustained_oscillation']
        sustained_osci_target = pops['sustained_activity_target']

        # Calculate the fitness as the absolute difference between the sustain duration and the target sustain duration
        sustain_fit = min(np.exp(np.abs(sustained_osci_target['target'] - sustained_osci)), maxFitness)

        print('Percent Duration: %.3f, Fitness: %.3f' % (sustained_osci, sustain_fit))
        return {'Value': sustained_osci, 'Fit': sustain_fit}

    except Exception as e:
        print(f'Error calculating sustain fitness: {e}')
        maxFitness = kwargs['maxFitness']
        return {'Value': None, 'Fit': maxFitness}
