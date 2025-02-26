print('Running test_dtw.py')
import numpy
from RBS_network_models.feature_struct import dtw_analysis_v3

# load data - reciprocal to this saving script
    ## =================== ##
    ## save test data for easier debugging
    # path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/tests/dtw_test_data/'
    # time_sequence_dict = {i: time_sequence_mat[i] for i in range(len(time_sequence_mat))}
    # cat_sequence_dict = {i: cat_sequence_mat[i] for i in range(len(cat_sequence_mat))}
    # np.save(path + 'time_sequence_dict.npy', time_sequence_dict)
    # np.save(path + 'cat_sequence_dict.npy', cat_sequence_dict)
    # np.save(path + 'sequence_stacks.npy', sequence_stacks)
    # np.save(path + 'bursting_data.npy', bursting_data)

print('Loading test data')    
path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/tests/dtw_test_data/'
time_sequence_dict = numpy.load(path + 'time_sequence_dict.npy', allow_pickle=True).item()
cat_sequence_dict = numpy.load(path + 'cat_sequence_dict.npy', allow_pickle=True).item()
sequence_stacks = numpy.load(path + 'sequence_stacks.npy', allow_pickle=True).item()
bursting_data = numpy.load(path + 'bursting_data.npy', allow_pickle=True).item()

# test dtw
print('Running dtw')
dtw_results = dtw_analysis_v3(
    time_sequence_dict, cat_sequence_dict, sequence_stacks, bursting_data
)

'''
salloc -A m2043 -q interactive -C gpu -t 04:00:00 --nodes=1 --image=adammwea/axonkilo_docker:v7
module load conda
conda activate netsims_env
python /pscratch/sd/a/adammwea/workspace/RBS_network_models/tests/test_dtw.py
'''