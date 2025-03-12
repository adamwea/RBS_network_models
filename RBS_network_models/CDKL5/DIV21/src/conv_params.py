
# # aw 2025-01-25 12:16:01 - Updated to include mega_params
# conv_params = {
#     'binSize': 0.01,
#     'gaussianSigma': 0.01,
#     #'thresholdBurst': 1.0,
#     'thresholdBurst': None, # no threshold - given the prominence method used in the burst detection, this should be fine...I hope
#     'min_peak_distance': None, # no minimum peak distance - given the prominence method used in the burst detection, this should be fine...I hope
#     'prominence': 2,
# }

# aw 2025-03-10 11:58:27 - now that we have better candidates, we dont need to be as sensitive
conv_params = {
    'binSize': 0.075,
    'gaussianSigma': 0.075,
    #'thresholdBurst': 1.0,
    'thresholdBurst': None, # no threshold - given the prominence method used in the burst detection, this should be fine...I hope
    'min_peak_distance': None, # no minimum peak distance - given the prominence method used in the burst detection, this should be fine...I hope
    'prominence': 2,
}

mega_params = { 
    #'binSize': conv_params['binSize']*15,
    'binSize': 0.01*15,
    #'gaussianSigma': conv_params['gaussianSigma']*15,
    ## aw 2025-02-04 17:33:19 - I want to be a little more sensitive to the peaks in the mega data
    #'gaussianSigma': conv_params['gaussianSigma']*15,
    'gaussianSigma': 0.01*15,
    #'thresholdBurst': 1.0,
    'thresholdBurst': None,
    'min_peak_distance': None, 
    'prominence': 4,
}
