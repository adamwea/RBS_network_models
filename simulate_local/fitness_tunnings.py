'''
FITNESS FUNCTION TARGETS
'''
pops = {}
'''untunned targets'''

## Number of Bursts per Duration

pops['num_bursts_target'] = {
    #'target': 10,
    #'max': 20,
    #'min': 5,
    #'width': 5,
    #'stdev': 1,
    #'skew': 0.5,
}


'''tunned targets'''

## FIRING RATE TARGETS
#firing rate targets
#pops['rate_targets'] = {}
E_target = 0.8773666667
max_rate = 2.033
min_rate = 0.337
# tunned to get fitness = 1.0 for experimental data.
width_mod = 250 # 250 might be a little wide. Probably better to start wide anyway.
pops['E_rate_target'] = {
    'target': E_target, #spikes/second
    'width': (max_rate - min_rate)*width_mod, 
    'max': max_rate*1.05,
    'min': min_rate*0.95,
    'stdev': 0.454,
    'skew': 1.018,
    'kurtosis': 0.095,
    }
I_target = 4.7104651163
max_rate = 12.956666666666667
min_rate = 2.0566666666666666
#tuned to get fitness = 1.0 for experimental data
width_mod = 200 # 250 might be a little wide. Probably better to start wide anyway.
pops['I_rate_target'] = {
    'target': I_target, #spikes/second
    'width': (max_rate - min_rate)*width_mod,
    'max': max_rate*1.05,
    'min': min_rate*0.95,
    'stdev': 2.4583078031387076,
    'skew': 1.4410949114139207,
    'kurtosis': 2.0718602443529788,
    }

# ISI TARGETS
E_target = 1.4235002806463521
max_ISI = 2.95531
min_ISI = 0.48860147783251234
width_mod = 250
pops['E_ISI_target'] = {
    'target': E_target, # seconds
    'width': (max_ISI - min_ISI)*width_mod, 
    'max': max_ISI*1.05,
    'min': min_ISI*0.95,
    'stdev': 0.6487012855947434,
    'skew': 0.4646037342251692,
    'kurtosis': -0.7503793922021273,
}

I_target = 0.26324381530063906
max_ISI = 0.4862136363636364
min_ISI = 0.07717779207411221
width_mod = 200
pops['I_ISI_target'] = {
    'target': I_target, # seconds
    'width': (max_ISI - min_ISI)*width_mod,
    'max': max_ISI*1.05,
    'min': min_ISI*0.95,
    'stdev': 0.11349480200712236,
    'skew': 0.3441884739586352,
    'kurtosis': -1.0144721473791476,
}

## BASELINE TARGETS
#baseline targets
#pops['baseline_targets'] = {'target': 1.5, 'width': 1 , 'max': 3}
width_tuned = 1000.499916708473 # tunned to get fitness = 1.0 for experimental data
baseline_target = 289.929
threshold_target = 718.115
width_mod = 1
pops['baseline_target'] = {
    'target': baseline_target , 
    'width': (threshold_target-0)*width_mod, 
    'max': threshold_target,
    'min': 0 ,
    'scale_factor': 2.0,
    } #spike count

# BURST/SPIKECOUNT TARGETS
big_width_mod = 500
lil_width_mod = 2250
max_big = 1954.7494420346768
min_big = 1253.316936848065
pops['big-small_cutoff'] = 1250
pops['big_burst_target'] = {
    'target': 1616.7837258361646,
    'max': max_big,
    'min': min_big,
    'width': (max_big-min_big)*big_width_mod,
    'stdev': 204.922,  
    'skew': -0.20136526164128335,  
    'kurtosis': -0.9145778975934289, 
}

lil_burst_min = 724.599
pops['small_burst_target'] = {
    'target': 402.633,
    'max': 1204.884+1,
    'min': lil_burst_min-1,
    'width': (1204.884-lil_burst_min)*lil_width_mod,
    'stdev': 124.560197413713,  
    'skew': 1.0072780690750132, 
    'kurtosis': 0.18082015049176414, 
}

num_big = 10  # bursts
num_small = 20  # bursts
pops['bimodal_burst_target'] = {
    'target': num_big/num_small,
    'mean': 1138.973614714823,
    'stdev': 372.14563335805457,
    'min': 756.1386713993633,
    'max': 1954.7494420346768,
    'skew': 0.7885775193441444,
    'kurtosis': -0.8282603799279,
    #'width': 0.5,
}

# BURST IBI TARGETS
width_mod = 450  # tuned to get fitness = 1.0 for experimental data
max_ibi = 23.6
pops['IBI_target'] = {
    'target': 8.790,
    'width': max_ibi*width_mod,
    'max': max_ibi,
    'min': 2.5,
    'stdev': 5.626567690116464,  # Placeholder value
    'skew': 0.8967834423955664,  # Placeholder value
    'kurtosis': -0.17410342980963867,  # Placeholder value
}

## BURST FREQUENCY TARGETS
#pops['burst_peak_frequency'] = {'target':1/3, 'width': 1/3*0.5, 'min': 1/4}
width_mod = 0.75 # tunned to get fitness = 1.0 for experimental data
pops['burst_frequency_target'] = {
    'target': 0.1, 
    #'width': (1-0)*width_mod, 
    'max': 1, # one burst per second 
    'min': 0,
    'scale_factor': 2.5,
    }    

## THRESHOLD TARGETS
#difference between baseline and threshold
#pops['thresh_target'] = {'target': 5, 'width': 1, 'min': 3, 'max': 7}
pops['threshold_target'] = {
    'target': 718.115, 
    #'width': 1, 
    'min': baseline_target-1, 
    'max': lil_burst_min+1,
    'scale_factor': 1,
    }

## OSCILLATION TARGETS
#pops['sustained_osci'] = {'target': 100, 'width': 5, 'min': 75}
pops['sustained_activity_target'] = {
    'target': 90.90303232255916, 
    #'width': 5, 
    #'min': 75
    }

## SLOPE TARGETS
#firing rate variance targets
#pops['rate_slope'] = {'target': 0, 'width': 0.5, 'max': 0.5}
pops['slope_target'] = {
    'target': -0.004, 
    #'width': 0.5, 
    #'max': 0.5
    }

'''Tunned Fitness Functions'''



'''FITNESS FUNCTION ARGUMENTS'''
# fitness function
fitnessFuncArgs = {}
fitnessFuncArgs['pops'] = pops
fitnessFuncArgs['maxFitness'] = 1000