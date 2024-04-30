from netpyne import specs

## Evol Params
#ProbLengthConst
params = specs.ODict()
params['probLengthConst'] = [50, 2000] # length constant for conn probability (um)    

#Propagation velocity
params['propVelocity'] = 1 #, propVelocity/100, propVelocity/10, propVelocity*10, propVelocity*100]

#Cell Params
params['probIE'] = [0, 1] # min: 0.2, max: 0.6
params['probEE'] = [0, 1] # min: 0.1, max: 0.3
params['probII'] = [0, 1] # min: 0.2, max: 0.6
params['probEI'] = [0, 1] # min: 0.1, max: 0.3

params['weightEI'] = [0, 0.1] # min: 0.001, max: 0.01
params['weightIE'] = [0, 0.1]
params['weightEE'] = [0, 0.1] # min: 0.001, max: 0.01
params['weightII'] = [0, 0.1]

params['gnabar_E'] = [0, 5] 
#params['gkbar_E'] = [0, 0.5]
params['gkbar_E'] = [0, 2]
#params['gnabar_I'] = [0, 1.5]
params['gnabar_I'] = [0, 3] 
#params['gkbar_I'] = [0.005, 2]
params['gkbar_I'] = [0.005, 5] 

#Hold these constant for now
#params['tau1_exc'] = [0.5, 1.5]
params['tau1_exc'] = [0.5, 6]
params['tau2_exc'] = [0, 10.0] 
params['tau1_inh'] = [0.05, 1] 
params['tau2_inh'] = [5.0, 15.0]
#default values for now
# params['tau1_exc'] = 0.8
# params['tau2_exc'] = 6.0
# params['tau1_inh'] = 0.8
# params['tau2_inh'] = 9.0

#Stimulation Params
# params['Erhythmic_stimWeight'] = [0, 0.02] 
# params['Irhythmic_stimWeight'] = [0, 0.02] 
# params['rythmic_stiminterval'] = [0, 5000]  # Interval between spikes (ms)
# params['rythmic_stimnoise'] = [0, 0.6]