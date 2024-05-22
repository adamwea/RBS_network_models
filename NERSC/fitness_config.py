'''
FITNESS FUNCTION TARGETS
'''
pops = {}
'''Tunned Fitness Functions'''

## BURST/SPIKECOUNT TARGETS
#pops['burts_peak_targets'] = {'target': 15, 'width': 2, 'min': 8}
big_width_mod = 500 #these are set so that experimental data just barely gets fitness = 1.000
lil_width_mod = 2250
lil_burst_min = 724.599
pops['burts_peak_targets'] = {
    'cutoff': 1250, #bigger than 1250 spikes = big burst
    'big_bursts':{
        'target': 1616.784, 
        #'width': 2, 
        'max': 1954.749+1, 
        'min': 1253.317-1,
        'width': (1954.749-1253.317)*big_width_mod,
        'num_target': 31.250, #% of bursts that are big
        'num_min': 0, #min number of big bursts
        #'num_width': 10, #+/- 10% of target
        },
    'lil_bursts':{
        'target': 402.633, 
        #'width': 2, 
        'max': 1204.884+1, 
        'min': lil_burst_min-1,
        'width': (1204.884-lil_burst_min)*lil_width_mod,
        'num_target': 68.750, #% of bursts that are big
        'num_min': 0, #min number of big bursts
        #'num_width': 10, #+/- 10% of target
        }        
    }

## BURST FREQUENCY TARGETS
#pops['burst_peak_frequency'] = {'target':1/3, 'width': 1/3*0.5, 'min': 1/4}
width_mod = 0.75 # tunned to get fitness = 1.0 for experimental data
pops['burst_peak_frequency'] = {
    'target': 0.11636363636363636, 
    #'width': (1-0)*width_mod, 
    'max': 1, # one burst per second 
    'min': 0,
    }    

## BURST IBI TARGETS
#burst IBI targets
#pops['IBI_targets'] = {'target': 3000, 'width': 2000 , 'max': 4000} #ms
width_mod = 450 # tunned to get fitness = 1.0 for experimental data
max_ibi = 23.6+1
pops['IBI_targets'] = {
    'target': 8.790, 
    'width': max_ibi*width_mod, 
    'max': max_ibi,
    #'min': 0, min is technically 0
    } #seconds

## BASELINE TARGETS
#baseline targets
#pops['baseline_targets'] = {'target': 1.5, 'width': 1 , 'max': 3}
width_tuned = 1000.499916708473 # tunned to get fitness = 1.0 for experimental data
target = 294.444
pops['baseline_targets'] = {
    'target': target , 
    #'width': width_tuned , 
    'max': lil_burst_min ,
    'min': 0 ,
    } #spike count

## SLOPE TARGETS
#firing rate variance targets
#pops['rate_slope'] = {'target': 0, 'width': 0.5, 'max': 0.5}
pops['rate_slope'] = {
    'target': 0.002497512709074353, 
    #'width': 0.5, 
    #'max': 0.5
    }

## OSCILLATION TARGETS
#pops['sustained_osci'] = {'target': 100, 'width': 5, 'min': 75}
pops['sustained_osci'] = {
    'target': 90.90303232255916, 
    #'width': 5, 
    #'min': 75
    }

## THRESHOLD TARGETS
#difference between baseline and threshold
#pops['thresh_target'] = {'target': 5, 'width': 1, 'min': 3, 'max': 7}
pops['thresh_target'] = {
    'target': 718.115, 
    #'width': 1, 
    #'min': 3, 
    'max': lil_burst_min
    }

## FIRING RATE TARGETS
#firing rate targets
pops['rate_targets'] = {}
E_target = 0.8773666667
pops['rate_targets']['E'] = {
    'target': E_target, 
    #'width': 2.5, 
    'min': 0
    }
pops['rate_targets']['I'] = {
    'target': 4.7104651163, 
    #'width': 10, 
    'min': E_target*3
    }

'''FITNESS FUNCTION ARGUMENTS'''
# fitness function
fitnessFuncArgs = {}
fitnessFuncArgs['pops'] = pops
fitnessFuncArgs['maxFitness'] = 1000