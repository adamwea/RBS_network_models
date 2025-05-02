# intial evolutionary parameter space for CDKL5_DIV21 project

# set up the environment
#from setup_environment import set_pythonpath
#set_pythonpath()
from netpyne import specs

# Evolutionary Parameters
params = specs.ODict()

# Propagation Parameters
params['propVelocity'] = 1  # Propagation velocity (arbitrary scaling examples in comments)

# Morphology Parameters (Excitatory and Inhibitory Cells)
params.update({
    
    #'E_diam_stdev': [0, 18.8 / 3],     # Standard deviation of excitatory diameter
    'E_diam_stdev': [0, 18.8],          # no obvious trend in the data, so I'm significantly increasing possible E_diam_stdev value
                                        # TODO: i hope increasing standard deviation of values like this doesn't break parameter assignment...
    
    #'E_L_stdev': [0, 18.8 / 2],         # Standard deviation of excitatory length
    'E_L_stdev': [0, 18.8],              # modest trend in the positive direction for better sims. I'm increasing the range to 0-18.8. Which is the max value for E_L_mean.
    
    #'E_Ra_stdev': [0, 128 / 2],      # Standard deviation of axial resistance
    'E_Ra_stdev': [0, 128],           # no obvious trend in the data, so I'm significantly increasing possible E_Ra_stdev value
    
    #'I_diam_stdev': [0, 10.0 / 3],   # Standard deviation of inhibitory diameter
    'I_diam_stdev': [0, 10.0],        # no obvious trend in the data, so I'm significantly increasing possible I_diam_stdev value
    
    #'I_L_stdev': [0, 9.0 / 3],         # Standard deviation of inhibitory length
    'I_L_stdev': [0, 9.0],              # no obvious trend in the data, so I'm significantly increasing possible I_L_stdev value
                                        # TODO: learn about the implications of this parameter for the model and neurophysiology
    
    #'I_Ra_stdev': [0, 110 / 2]       # Standard deviation of axial resistance
    'I_Ra_stdev': [0, 110],            # no obvious trend in the data, so I'm significantly increasing possible I_Ra_stdev value
    
    ## constants
    'E_diam_mean': 18.8,              # Mean diameter of excitatory cells (um)
    'E_L_mean': 18.8,                # Mean length of excitatory cells (um)
    'E_Ra_mean': 128,                # Mean axial resistance of excitatory cells (ohm/cm)
    'I_diam_mean': 10.0,             # Mean diameter of inhibitory cells (um)
    'I_L_mean': 9.0,                 # Mean length of inhibitory cells (um)
    'I_Ra_mean': 110,                # Mean axial resistance of inhibitory cells (ohm/cm)
    
    
})

# Connection Probability Length Constant
#params['probLengthConst'] = [50, 3000]  # Length constant for connection probability (um)
params['probLengthConst'] = [1, 5000]  # no obvious trend in the data, so I'm significantly increasing possible probLengthConst value

# Connectivity Parameters
params.update({
    ## probabiilties
    #'probIE': [0, 0.75],            # Inhibitory to Excitatory probability
    'probIE': [0, 1],                # No obvious trend. I'm increasing the range to 0-1. Which is the max value for a probability. obv.
    
    #'probEE': [0, 0.75],            # Excitatory to Excitatory probability
    'probEE': [0, 1],                # clear trend in the positive direction for better sims. I'm increasing the range to 0-1. Which is the max value for a probability. obv.    
    
    #'probII': [0, 0.75],            # Inhibitory to Inhibitory probability
    'probII': [0, 1],                # clear trend in the positive direction for better sims. I'm increasing the range to 0-1. Which is the max value for a probability. obv.
    
    #'probEI': [0.25, 1],            # Excitatory to Inhibitory probability
    'probEI': [0, 1],                # no obvious trend in the data, so I'm significantly increasing possible probEI value
    
    ## weights
    #'weightEI': [0, 2],                # Weight for Excitatory to Inhibitory connections
    'weightEI': [0, 10],                # no obvious trend in the data, so I'm significantly increasing possible weightEI value
                                        # Weight parameters are where I can exert the most arbitrary control over the model...I think.
                                        # TODO: Justify my intuition about the weight parameters.
    
    #'weightIE': [0, 1],             # Weight for Inhibitory to Excitatory connections
    'weightIE': [0, 10],             # no obvious trend in the data, so I'm significantly increasing possible weightIE value
    
    #'weightEE': [0, 2],             # Weight for Excitatory to Excitatory connections
    'weightEE': [0, 10],             # no obvious trend in the data, so I'm significantly increasing possible weightEE value
    
    #'weightII': [0, 2]              # Weight for Inhibitory to Inhibitory connections    
    'weightII': [0, 10]              # no obvious trend in the data, so I'm significantly increasing possible weightII value
})

# Sodium (gnabar) and Potassium (gkbar) Conductances
params.update({
    
    #'gnabar_E': [0.5, 7],           # Sodium conductance for excitatory cells
    'gnabar_E': [0, 15],             # no clear trend in the data, so I'm increasing the range to 0-15
        
    #'gnabar_E_std': [0, 5 / 3], # Standard deviation of sodium conductance for excitatory cells
    'gnabar_E_std': [0, 15],          # no clear trend in the data, so I'm increasing the range to 0-5
    
    'gkbar_E': [0, 1],              # Potassium conductance for excitatory cells
                                    # pretty clear negative trend. holding constant at 0-1 for now.
                                    
    
    #'gkbar_E_std': [0, 1 / 2],         # Standard deviation of potassium conductance for excitatory cells
    'gkbar_E_std': [0, 1],              # actually, looks like a negative trend in the data. not sure if I should tighten the range or not.
                                        # there are values at the positive end of the range still. I have limited data to work with. Just going to widen the range anyway.
    
    # 'gnabar_I': [0, 5],               # Sodium conductance for inhibitory cells
    'gnabar_I': [0, 15],                # clear trend in the positive direction for better sims. I'm increasing the range to 0-10
    
    #'gnabar_I_std': [0, 3 / 3], # Standard deviation of sodium conductance for inhibitory cells
    'gnabar_I_std': [0, 15],          # no clear trend in the data, so I'm increasing the range to 0-3
    
    #'gkbar_I': [0, 10],             # Potassium conductance for inhibitory cells
    'gkbar_I': [0, 15/2],              # clear negative trend. very clear. reducing the range to 0-7.5
    
    #'gkbar_I_std': [0, 7.5 / 3] # Standard deviation of potassium conductance for inhibitory cells
    'gkbar_I_std': [0, 10]          # no clear trend in the data, so I'm increasing the range to 0-10
})

# Synaptic Time Constants
params.update({
    #'tau1_exc': [0, 7.5],              # Rise time of excitatory synaptic conductance
    'tau1_exc': [0, 15],                # no clear trend in the data, so I'm increasing the range to 0-15
    
    #'tau2_exc': [0, 30.0],          # Decay time of excitatory synaptic conductance
    'tau2_exc': [0, 60],            # no clear trend in the data, so I'm increasing the range to 0-60
    
    # 'tau1_inh': [0, 10],            # Rise time of inhibitory synaptic conductance
    'tau1_inh': [0, 20],            # no clear trend in the data, so I'm increasing the range to 0-20
    
    #'tau2_inh': [0, 20.0]           # Decay time of inhibitory synaptic conductance
    'tau2_inh': [0, 40]             # no clear trend in the data, so I'm increasing the range to 0-40
})

# Stimulation Parameters (default values commented out for now)
# params['Erhythmic_stimWeight'] = [0, 0.02]
# params['Irhythmic_stimWeight'] = [0, 0.02]
# params['rythmic_stiminterval'] = [0, 5000]  # Interval between spikes (ms)
# params['rythmic_stimnoise'] = [0, 0.6]
