#import RBS_network_models.developing.CDKL5.DIV21.src.fitnessFunc as fitnessFunc
from RBS_network_models.developing.CDKL5.DIV21.src.fitnessFunc import fitnessFunc
from RBS_network_models.developing.CDKL5.DIV21.src.fitness_targets import fitnessFuncArgs
#from RBS_network_models.developing.CDKL5.DIV21.src.fitness_targets import fitness_args
from netpyne import sim
from RBS_network_models.developing.CDKL5.DIV21.src.conv_params import conv_params
# =============================================================================
data_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/developing/CDKL5/DIV21/outputs/batch_2025-01-05/gen_0/gen_0_cand_0_data.pkl' # just chose a random path to test
# =============================================================================

'''optimizing mode'''   # NOTE: this is the default mode. This is the mode that is used when running batch processing. 
                        #       Usuall optimization algorithms are used in this mode.

sim.load(data_path)
simData = sim.allSimData
candidate_path = data_path.replace('_data.pkl', '') 

kwargs = {
    'mode': 'optimizing',
    'data_path': data_path, # NOTE: this also wouldn't typically be passed in batch processing. But for simulating fitness function performance, it is necessary.
    'simData': simData, # to simulate optimizing mode, simData needs to be passed to the function
    'skip_existing': False, # TODO: need to implement in batch processing
    'conv_params': conv_params, # Params for convolving the spiking data into network busrting data
    'candidate_path': candidate_path,   #   NOTE need to specify this for testing fitness function performance during batch processing outside of batch processing. 
                                        #   In batch processing, this is automatically set by the searching the call stack.    
    }
kwargs.update(fitnessFuncArgs)

average_fitness = fitnessFunc(**kwargs)
print(average_fitness) # should return a float