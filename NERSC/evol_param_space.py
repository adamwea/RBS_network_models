from netpyne import specs
from USER_INPUTS import USER_seconds
#from __main__ import USER_seconds #import user seconds which should have been declared in the batchRun script

## Evol Params
params = specs.ODict()

#Duration
#TODO - move these to const net params eventually
params['duration_seconds'] = USER_seconds # Duration of the simulation, in ms
#Propagation velocity
params['propVelocity'] = 1 #, propVelocity/100, propVelocity/10, propVelocity*10, propVelocity*100]

#Morphology Params
params['E_diam_mean'] = 18.8 # mean diameter of excitatory cells (um)
params['E_diam_stdev'] = [0, 18.8/3] #
params['E_L_mean'] = 18.8 # mean length of excitatory cells (um)
params['E_L_stdev'] = [0, 18.8/2] # updated 240528
params['E_Ra_mean'] = 128 # mean axial resistance of excitatory cells (ohm/cm)
params['E_Ra_stdev'] = [0, 128/2] # updated 240528
params['I_diam_mean'] = 10.0 # mean diameter of inhibitory cells (um)
params['I_diam_stdev'] = [0, 10.0/3] #
params['I_L_mean'] = 9.0 # mean length of inhibitory cells (um)
params['I_L_stdev'] = [0, 9.0/3] #
params['I_Ra_mean'] = 110 # mean axial resistance of inhibitory cells (ohm/cm)
params['I_Ra_stdev'] = [0, 110/2] # updated 240528

#ProbLengthConst
params['probLengthConst'] = [50, 3000] # length constant for conn probability (um)    

#Cell Params
params['probIE'] = [0, 0.75] # updated 240528, made a bit tighter
params['probIE_std'] = [0, 1/2] #updated 240528
params['probEE'] = [0, 0.75] # updated 240528, made a bit tighter
params['probEE_std'] = [0, 1/2] # min: 0.1, max: 0.3
params['probII'] = [0, 0.75] # updated 240528, made a bit tighter
params['probII_std'] = [0, 1/2] # updated 240528
params['probEI'] = [0.25, 1] #updated 240528, made a bit tighter
params['probEI_std'] = [0, 1/2] #updated 240528

params['weightEI'] = [0, 2] # updated 240528
params['weightEI_std'] = [0, 1/3] # min: 0.001, max: 0.01
params['weightIE'] = [0, 1]
params['weightIE_std'] = [0, 1/3]
params['weightEE'] = [0, 2] # updated 240528
params['weightEE_std'] = [0, 1/2] # updated 240528
params['weightII'] = [0, 2] #updated 240528
params['weightII_std'] = [0, 0.5/2] #updated 240528

params['gnabar_E'] = [0.5, 7] #updated 240528
params['gnabar_E_std'] = [0, 5/3]

#gkbar_E - potassium conductance
#params['gkbar_E'] = [0, 0.5]
#params['gkbar_E'] = [0, 2]
#06may24 - good solutions are heavily trending toward zero. reducing range to 25% of previous range
params['gkbar_E'] = [0, 1]
params['gkbar_E_std'] = [0, 1/2] #updated 240528


#params['gnabar_I'] = [0, 1.5]
params['gnabar_I'] = [0, 5] #updated 240528
params['gnabar_I_std'] = [0, 3/3] 
#params['gkbar_I'] = [0.005, 2]
params['gkbar_I'] = [0, 10] #updated 240528
params['gkbar_I_std'] = [0, 7.5/3] 

#params['tau1_exc'] = [0.5, 1.5]
params['tau1_exc'] = [0, 7.5] #updated 240528, made a bit tighter
params['tau1_exc_std'] = [0, 10/3]
params['tau2_exc'] = [0, 30.0] #updated 240528
params['tau2_exc_std'] = [0, 20.0/2] #updated 240528 
params['tau1_inh'] = [0, 10]
params['tau1_inh_std'] = [0, 10/3] 
params['tau2_inh'] = [0, 20.0]
params['tau2_inh_std'] = [0, 20.0/2] #updated 240528

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