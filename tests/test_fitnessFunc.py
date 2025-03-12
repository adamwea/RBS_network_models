'''
This test should:
1. Load some data network metrics .npy file.
2. test it's fitness against itself using the current fitnessFunc.
3. Print the result. It should return 1 since it's comparing the data to itself.
'''
# Imports =====================================================================
import os
import numpy as np
from RBS_network_models.fitnessFunc import fitnessFunc_v2

# Main =============================================================================
# experimental data
test_data_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/network_metrics/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/well000/network_metrics.npy'
test_data = np.load(test_data_path, allow_pickle=True).item()

# simulated data
# test_data_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/sensitivity_analyses/2025-03-04_propVelocity_3_data_35s/_propVelocity_3/_propVelocity_3/network_data.npy'
# test_data = np.load(test_data_path, allow_pickle=True).item()

# get the size of the data - gigabytes
data_size = os.path.getsize(test_data_path) / (1024**3)
print("Data size: ", data_size, "GB")

#mock simulation data
#sim_data = test_data.copy()
experimental_data = test_data
sim_data_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/sensitivity_analyses/2025-03-05_propVelocity_3_data_65s/_propVelocity_3/_propVelocity_3/network_data.npy'
sim_data = np.load(sim_data_path, allow_pickle=True).item()

#kwargs
kwargs = {} #no kwargs needed for this test - I think.

#run fitness function
avg_fitness = fitnessFunc_v2(sim_data, experimental_data, **kwargs)

# Print result
print("Average fitness: ", avg_fitness)

# Expected output:
# Average fitness:  1.0
# This is because the test data is being compared to itself.
assert avg_fitness == 1.0, "Test failed. Expected output: 1.0, but got: " + str(avg_fitness)
print("Test passed.")

# End of test_fitnessFunc.py