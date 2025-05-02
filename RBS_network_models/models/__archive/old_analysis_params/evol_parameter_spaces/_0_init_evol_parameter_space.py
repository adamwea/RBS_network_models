# intial evolutionary parameter space for CDKL5_DIV21 project

# set up the environment
from setup_environment import set_pythonpath
set_pythonpath()
from netpyne import specs

# Evolutionary Parameters
params = specs.ODict()

# Propagation Parameters
params['propVelocity'] = 1  # Propagation velocity (arbitrary scaling examples in comments)

# Morphology Parameters (Excitatory and Inhibitory Cells)
params.update({
    'E_diam_mean': 18.8,              # Mean diameter of excitatory cells (um)
    'E_diam_stdev': [0, 18.8 / 3],   # Standard deviation of excitatory diameter
    'E_L_mean': 18.8,                # Mean length of excitatory cells (um)
    'E_L_stdev': [0, 18.8 / 2],      # Standard deviation of excitatory length
    'E_Ra_mean': 128,                # Mean axial resistance of excitatory cells (ohm/cm)
    'E_Ra_stdev': [0, 128 / 2],      # Standard deviation of axial resistance
    'I_diam_mean': 10.0,             # Mean diameter of inhibitory cells (um)
    'I_diam_stdev': [0, 10.0 / 3],   # Standard deviation of inhibitory diameter
    'I_L_mean': 9.0,                 # Mean length of inhibitory cells (um)
    'I_L_stdev': [0, 9.0 / 3],       # Standard deviation of inhibitory length
    'I_Ra_mean': 110,                # Mean axial resistance of inhibitory cells (ohm/cm)
    'I_Ra_stdev': [0, 110 / 2]       # Standard deviation of axial resistance
})

# Connection Probability Length Constant
params['probLengthConst'] = [50, 3000]  # Length constant for connection probability (um)

# Connectivity Parameters
params.update({
    'probIE': [0, 0.75],            # Inhibitory to Excitatory probability
    'probEE': [0, 0.75],            # Excitatory to Excitatory probability
    'probII': [0, 0.75],            # Inhibitory to Inhibitory probability
    'probEI': [0.25, 1],            # Excitatory to Inhibitory probability
    'weightEI': [0, 2],             # Weight for Excitatory to Inhibitory connections
    'weightIE': [0, 1],             # Weight for Inhibitory to Excitatory connections
    'weightEE': [0, 2],             # Weight for Excitatory to Excitatory connections
    'weightII': [0, 2]              # Weight for Inhibitory to Inhibitory connections
})

# Sodium (gnabar) and Potassium (gkbar) Conductances
params.update({
    'gnabar_E': [0.5, 7],           # Sodium conductance for excitatory cells
    'gnabar_E_std': [0, 5 / 3],
    'gkbar_E': [0, 1],              # Potassium conductance for excitatory cells
    'gkbar_E_std': [0, 1 / 2],
    'gnabar_I': [0, 5],             # Sodium conductance for inhibitory cells
    'gnabar_I_std': [0, 3 / 3],
    'gkbar_I': [0, 10],             # Potassium conductance for inhibitory cells
    'gkbar_I_std': [0, 7.5 / 3]
})

# Synaptic Time Constants
params.update({
    'tau1_exc': [0, 7.5],           # Rise time of excitatory synaptic conductance
    'tau2_exc': [0, 30.0],          # Decay time of excitatory synaptic conductance
    'tau1_inh': [0, 10],            # Rise time of inhibitory synaptic conductance
    'tau2_inh': [0, 20.0]           # Decay time of inhibitory synaptic conductance
})

# Stimulation Parameters (default values commented out for now)
# params['Erhythmic_stimWeight'] = [0, 0.02]
# params['Irhythmic_stimWeight'] = [0, 0.02]
# params['rythmic_stiminterval'] = [0, 5000]  # Interval between spikes (ms)
# params['rythmic_stimnoise'] = [0, 0.6]
