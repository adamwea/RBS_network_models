import numpy as np
from scipy.stats import linregress

def fit_rate_slope(net_activity_metrics, **kwargs):
    try:
        print('Calculating firing rate slope fitness...')
        pops = kwargs['pops']
        rate_slope = pops['slope_target']
        maxFitness = kwargs['maxFitness']

        # Get the firing rate from the network metrics
        firingRate = net_activity_metrics['firingRate']

        # Calculate the trendline of the firing rate
        slope, intercept, r_value, p_value, std_err = linregress(range(len(firingRate)), firingRate)

        # Calculate the fitness as the absolute difference between the slope and the target slope
        slopeFitness = min(np.exp(abs(rate_slope['target'] - slope)), maxFitness)

        print('Slope of firing rate: %.3f, Fitness: %.3f' % (slope, slopeFitness))
        return {'Value': slope, 'Fit': slopeFitness}

    except Exception as e:
        print(f'Error calculating firing rate slope fitness: {e}')
        return {'Value': None, 'Fit': kwargs['maxFitness']}
