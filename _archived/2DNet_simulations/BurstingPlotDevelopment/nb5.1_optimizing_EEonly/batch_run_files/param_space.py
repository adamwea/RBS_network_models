from netpyne import specs

def get_param_space(duration_seconds):
    #from const_netParams import get_const_netParams
    import const_netParams as const_netparams
    #initialize parameters
    params = specs.ODict()
    
    ## Define full parameter space
    #constant
    #Prepare constant netparams parameterrs
    params['duration_seconds'] = duration_seconds
    probLengthConst = const_netparams.netParams.probLengthConst
    params['probLengthConst'] = probLengthConst  # length constant for conn probability (um)

    # General Net Params
    # Propagation velocity
    #propVelocity = 100
    propVelocity = 1
    params['propVelocity'] = propVelocity #, propVelocity/100, propVelocity/10, propVelocity*10, propVelocity*100]

    #Cell Params
    params['probIE'] = [0, 1] # min: 0.2, max: 0.6
    params['probEE'] = [0, 1] # min: 0.1, max: 0.3
    params['probII'] = [0, 1] # min: 0.2, max: 0.6
    params['probEI'] = [0, 1] # min: 0.1, max: 0.3

    params['weightEI'] = [0.0001, 0.1] # min: 0.001, max: 0.01
    #params['weightIE'] = [0.001, 0.1] # min: 0.01, max: 0.1
    params['weightIE'] = [0.0001, 0.1]
    params['weightEE'] = [0.0001, 0.1] # min: 0.001, max: 0.01
    #params['weightII'] = [0.001, 0.1] # min: 0.01, max: 0.1
    params['weightII'] = [0.0001, 0.1]

    params['gnabar_E'] = [0, 5] 
    params['gkbar_E'] = [0, 0.5] 
    params['gnabar_I'] = [0, 1.5] 
    params['gkbar_I'] = [0.005, 2] 

    #Hold these constant for now
    # params['tau1_exc'] = [0.5, 1.5] 
    # params['tau2_exc'] = [3.0, 10.0] 
    # params['tau1_inh'] = [0.05, 0.2] 
    # params['tau2_inh'] = [5.0, 15.0]
    #default values for now
    params['tau1_exc'] = 0.8
    params['tau2_exc'] = 6.0
    params['tau1_inh'] = 0.8
    params['tau2_inh'] = 9.0

    #Stimulation Params
    params['stimWeight'] = [0, 0.002] 
    params['stim_rate'] = [0, 0.15] 
    params['stim_noise'] = [0.2, 0.6]

    # Prepare params for evol batching.
    # If any param is a single value, convert to list with that value twice
    for key, value in params.items():
        if isinstance(value, (int, float)):
            params[key] = [value, value]
        elif isinstance(value, list) and len(value) == 1:
            params[key] = [value[0], value[0]]

    #run_params = params.copy()

    return params

def get_param_space_excite_only(duration_seconds):
    #from const_netParams import get_const_netParams
    import const_netParams_excite as const_netparams
    
    #initialize parameters
    params = specs.ODict()
    
    ## Define full parameter space
    #constant
    #Prepare constant netparams parameterrs
    params['duration_seconds'] = duration_seconds
    probLengthConst = const_netparams.netParams.probLengthConst
    params['probLengthConst'] = probLengthConst  # length constant for conn probability (um)

    # General Net Params
    # Propagation velocity
    #propVelocity = 100
    propVelocity = 1
    params['propVelocity'] = propVelocity #, propVelocity/100, propVelocity/10, propVelocity*10, propVelocity*100]

    #Cell Params
    #params['probIE'] = [0, 1] # min: 0.2, max: 0.6
    params['probEE'] = [0, 1] # min: 0.1, max: 0.3
    #params['probII'] = [0, 1] # min: 0.2, max: 0.6
    #params['probEI'] = [0, 1] # min: 0.1, max: 0.3

    #params['weightEI'] = [0.0001, 0.1] # min: 0.001, max: 0.01
    #params['weightIE'] = [0.001, 0.1] # min: 0.01, max: 0.1
    #params['weightIE'] = [0.0001, 0.1]
    params['weightEE'] = [0.0001, 0.1] # min: 0.001, max: 0.01
    #params['weightII'] = [0.001, 0.1] # min: 0.01, max: 0.1
    #params['weightII'] = [0.0001, 0.1]

    params['gnabar_E'] = [0, 5] 
    params['gkbar_E'] = [0, 0.5] 
    #params['gnabar_I'] = [0, 1.5] 
    #params['gkbar_I'] = [0.005, 2] 

    #Hold these constant for now
    # params['tau1_exc'] = [0.5, 1.5] 
    # params['tau2_exc'] = [3.0, 10.0] 
    # params['tau1_inh'] = [0.05, 0.2] 
    # params['tau2_inh'] = [5.0, 15.0]
    #default values for now
    params['tau1_exc'] = 0.8
    params['tau2_exc'] = 6.0
    #params['tau1_inh'] = 0.8
    #params['tau2_inh'] = 9.0

    #Stimulation Params
    params['stimWeight'] = [0, 0.002] 
    params['stim_rate'] = [0, 0.15] 
    params['stim_noise'] = [0.2, 0.6]

    # Prepare params for evol batching.
    # If any param is a single value, convert to list with that value twice
    for key, value in params.items():
        if isinstance(value, (int, float)):
            params[key] = [value, value]
        elif isinstance(value, list) and len(value) == 1:
            params[key] = [value[0], value[0]]

    #run_params = params.copy()

    return params  