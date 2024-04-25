'''
FITNESS FUNCTION TARGETS
'''
pops = {}
#firing rate targets
pops['rate_targets'] = {}
pops['rate_targets']['E'] = {'target': 7.5, 'width': 2.5, 'min': 1}
pops['rate_targets']['I'] = {'target': 30, 'width': 10, 'min': 2}
#burst peak targets
pops['burts_peak_targets'] = {'target': 15, 'width': 5, 'min': 1}
#burst IBI targets
pops['IBI_targets'] = {'target': 3000, 'width': 2000 , 'max': 4000} #ms
#baseline targets
pops['baseline_targets'] = {'target': 1.5, 'width': 1.5 , 'max': 5} #ms
#firing rate variance targets
pops['rate_slope'] = {'target': 0, 'width': 0.5, 'max': 0.5}
#difference between baseline and threshold
pops['baseline_diff'] = {'target': 5, 'width': 2.0, 'min': 2}
#
pops['burst_peak_variance'] = {'target': 5, 'width': 1.0, 'min': 3}
#
pops['burst_peak_frequency'] = {'target':1/3, 'width': 1/3*0.5, 'min': 1/4}    

# fitness function
fitnessFuncArgs = {}
fitnessFuncArgs['pops'] = pops
fitnessFuncArgs['maxFitness'] = 1000