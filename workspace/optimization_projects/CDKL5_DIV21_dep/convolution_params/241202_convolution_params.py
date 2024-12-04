# Standard convolution parameters for network analysis in RBS
# conv_params = {
#     'binSize': 0.1,
#     'gaussianSigma': 0.15,
#     'thresholdBurst': 1.0,
#     'min_peak_distance': 1
# }

# #tweaking binsize to be smaller to get better resolution of network activity for shorter simulations.
# conv_params = {
#     'binSize': 0.01,
#     'gaussianSigma': 0.15,
#     'thresholdBurst': 1.0,
#     'min_peak_distance': 1
# }

#tweaking binsize to be smaller to get better resolution of network activity for shorter simulations.
conv_params = {
    'binSize': 0.01,
    'gaussianSigma': 0.01,
    'thresholdBurst': 1.0,
    'min_peak_distance': None # no minimum peak distance - given the prominence method used in the burst detection, this should be fine...I hope
}

