from fitness_functions import fitnessFunc
from fitness_config import fitnessFuncArgs
import netpyne
from fitness_functions import fit_baseline
from fitness_functions import fit_burst_frequency
from fitness_functions import fit_threshold

# import dummy data for testing
#data_file_path = '/home/adamm/adamm/Documents/GithubRepositories/2DNetworkSimulations/NERSC/output/240527_Run3_this_should_work/gen_19/gen_19_cand_13_data.json'
#netpyne.sim.loadAll(data_file_path)
#simData = netpyne.sim.allSimData
#kwargs = fitnessFuncArgs
#avgScaledFitness = fitnessFunc(simData, plot = False, data_file_path = data_file_path, **kwargs)
#print(avgScaledFitness)

## Test fit_baseline
print('Test fit_baseline_____________')
net_activity_metrics = {}
kwargs = fitnessFuncArgs
width = kwargs['pops']['baseline_target']['width']
kwargs['pops']['baseline_target']['width'] = width/2
kwargs['pops']['baseline_target']['scale_factor'] = 2

i=0
#for dummy_value in range(1000):
for dummy_value in range(1000, 0, -1):
    i +=1
    net_activity_metrics['baseline'] = dummy_value
    fit = fit_baseline(net_activity_metrics, **kwargs)
    #print(kwargs['pops']['baseline_target']['width'])
    if fit['Fit'] != 1000:
        print('Limit Fit')
        print(fit)
        break
print('Zero Fit')
net_activity_metrics['baseline'] = 0
fit_zero = fit_baseline(net_activity_metrics, **kwargs)
print(fit_zero)
print('Target Fit')
net_activity_metrics['baseline'] = kwargs['pops']['baseline_target']['target']
fit_target = fit_baseline(net_activity_metrics, **kwargs)
print(fit_target)

# import sys
# sys.exit()

## Test burst frequency
print('Test fit_burst_frequency_____________')
net_activity_metrics = {}
kwargs = fitnessFuncArgs
#kwargs['pops']['burst_frequency_target']['scale_factor'] = 1.50

i=0
#for dummy_value in range(1000):
for dummy_value in range(1000, 0, -1):
    i +=1
    net_activity_metrics['peakFreq'] = dummy_value
    fit = fit_burst_frequency(net_activity_metrics, **kwargs)
    #print(kwargs['pops']['baseline_target']['width'])
    if fit['Fit'] != 1000:
        print('Limit Fit')
        print(fit)
        break
print('Test Fit') 
net_activity_metrics['peakFreq'] = 0.25
fit_test = fit_burst_frequency(net_activity_metrics, **kwargs)
print(fit_test)
print('Test Fit') 
net_activity_metrics['peakFreq'] = 0.05
fit_test = fit_burst_frequency(net_activity_metrics, **kwargs)
print(fit_test)
print('Zero Fit')
net_activity_metrics['peakFreq'] = 0.0
fit_zero = fit_burst_frequency(net_activity_metrics, **kwargs)
print(fit_zero)
print('Target Fit')
net_activity_metrics['peakFreq'] = kwargs['pops']['burst_frequency_target']['target']
fit_target = fit_burst_frequency(net_activity_metrics, **kwargs)
print(fit_target)

bursts_in_300s = 0.1*300
print('Bursts in 300s')
print(bursts_in_300s)

import sys
sys.exit()

## Test fit_threshold
print('Test fit_threshold_____________')
net_activity_metrics = {}
kwargs = fitnessFuncArgs
#width = kwargs['pops']['threshold_target']['width']
#kwargs['pops']['threshold_target']['width'] = width/2
kwargs['pops']['threshold_target']['scale_factor'] = 175

i=0
#for dummy_value in range(1000):
#reverse for loop to test the upper limit
for dummy_value in range(1000,0,-1):
    i +=1
    net_activity_metrics['threshold'] = dummy_value
    fit = fit_threshold(net_activity_metrics, **kwargs)
    #print(kwargs['pops']['baseline_target']['width'])
    if fit['Fit'] != 1000:
        print('Limit Fit')
        print(fit)
        break
print('Test Fit')
net_activity_metrics['threshold'] = 400
fit_test = fit_threshold(net_activity_metrics, **kwargs)
print(fit_test)

print('min Fit')
fit_zero = 1000
scale_factor = 1
while fit_zero==1000:
    scale_factor = scale_factor+0.01
    kwargs['pops']['threshold_target']['scale_factor'] = scale_factor
    net_activity_metrics['threshold'] = 289.929
    fit_zero = fit_threshold(net_activity_metrics, **kwargs)
    fit_zero = fit_zero['Fit']
print(fit_zero)
print('Target Fit')
net_activity_metrics['threshold'] = kwargs['pops']['threshold_target']['target']
fit_target = fit_threshold(net_activity_metrics, **kwargs)
print(fit_target)


