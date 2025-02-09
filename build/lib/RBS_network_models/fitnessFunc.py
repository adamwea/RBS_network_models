#from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.fitness_helper import *
#from DIV21.utils.fitness_helper import *
#from .utils.fitness_helper import *
#from RBS_network_models.network_analysis import calculate_network_metrics
import os
import json
import numpy as np

''' Main Func '''
def fitnessFunc(simData=None, mode='optimizing', **kwargs):
    """
    Main logic of the calculate_fitness function.
    Ensures the function does not crash and always returns a fitness value.
    """

    def handle_optimizing_mode(simData, plot_sim, kwargs):
        
        """Handles the logic for 'optimizing' mode."""
        if simData is not None:
            from .utils.extract_simulated_data import get_candidate_and_job_path_from_call_stack, retrieve_sim_data_from_call_stack, extract_data_of_interest_from_sim
            kwargs['source'] = 'simulated'
            candidate_path = kwargs.get('candidate_path', None)
            if candidate_path is None:
                candidate_path, job_path = get_candidate_and_job_path_from_call_stack() #NOTE this part specifcally only works in batch processing, not in single candidate processing
                                                                                        # so, for testing fitness function, this part needs a workaround
            else: 
                candidate_path = kwargs['candidate_path']
            fitness_save_path = f'{candidate_path}_fitness.json'
            
            # Check if fitness results already exist
            skip_existing = kwargs.get('skip_existing', False)
            if skip_existing:
                existing_fitness = handle_existing_fitness(fitness_save_path)
                if existing_fitness is not None:
                    return existing_fitness, kwargs

            #from ....utils.extract_simulated_data import retrieve_sim_data_from_call_stack
            data_file_path = kwargs.get('data_file_path', None)
            if data_file_path is None:
                kwargs = retrieve_sim_data_from_call_stack(simData, **kwargs)
            else:
                # this is for debugging purposes
                data_file_path = data_file_path
                candidate_label = os.path.basename(candidate_path)
                cfg_file_path = f'{candidate_path}_cfg.json'
                extracted_data = extract_data_of_interest_from_sim(data_file_path, simData)
                kwargs.update({
                    'simData': extracted_data['simData'],
                    'cellData': extracted_data['cellData'],
                    'popData': extracted_data['popData'],
                    'simCfg': extracted_data['simCfg'],
                    'netParams': extracted_data['netParams'],
                    'simLabel': candidate_label,
                    'data_file_path': data_file_path,
                    'cfg_file_path': cfg_file_path,
                    'fitness_save_path': fitness_save_path, #output
                })
            error, kwargs = calculate_network_metrics(kwargs)
            if plot_sim:
                # data_file_path = kwargs.get('data_file_path', None)
                # sim_data_path = data_file_path
                # reference_data_path = kwargs.get('reference_data_path', None)
                # conv_params = kwargs.get('conv_params', None)
                # mega_params = kwargs.get('mega_params', None)
                # fitnessFuncArgs = kwargs.copy()
                # from RBS_network_models.sim_analysis import process_simulation
                # process_simulation(
                #     sim_data_path, 
                #     reference_data_path,
                #     DEBUG_MODE=False,
                #     conv_params = conv_params,
                #     mega_params = mega_params,
                #     fitnessFuncArgs = fitnessFuncArgs,
                #     )
                
                submit_plotting_job(candidate_path, kwargs)
            return error, kwargs
        return None, kwargs

    def handle_simulated_data_mode(simData, kwargs):
        """
        
        Handles the logic for 'simulated data' mode.
        
        """
        kwargs['source'] = 'simulated'
        #candidate_path = kwargs['simConfig']['filename']
        candidate_dir = kwargs['simConfig']['saveFolder']
        filename = kwargs['simConfig']['filename']
        fitness_save_path = os.path.join(candidate_dir, f'{filename}_fitness.json')
        kwargs.update({'fitness_save_path': fitness_save_path})
        assert simData is not None, 'Simulated data must be provided in "simulated data" mode.'
        kwargs.update({'simData': simData})
        
        #assert conv_params and mega_params in kwargs, 'Convolution parameters must be provided in "simulated data" mode.'
        assert 'conv_params' in kwargs, 'Convolution parameters must be provided in "simulated data" mode.'
        assert 'mega_params' in kwargs, 'Mega convolution parameters must be provided in "simulated data" mode.'
        
        error, kwargs = calculate_network_metrics(kwargs)
        return error, kwargs

    def handle_experimental_data_mode(
        #source='experimental',
        **kwargs):
        """
        
        Handles the logic for 'experimental data' mode.
        
        Important for initial calibration of the fitness targets against the experimental data prior 
        to the optimization process.
        
        """
        
        #assert source == 'experimental', 'Source must be "experimental" in "experimental data" mode.'
        #kwargs['source'] = 'experimental'
        # implemented = False
        # assert implemented, 'Experimental data source not yet implemented.'        
        #error, kwargs = calculate_network_metrics(kwargs)
        
        # aw 2025-01-26 17:46:29 - do nothing since we already have the network metrics?
        
        assert 'network_metrics' in kwargs, 'Network metrics must be provided in "experimental data" mode.'
        
        #source of the data
        kwargs['source'] = 'experimental'
        
        # define fitness save path
        sorter_dir = kwargs['network_metrics']['sorting_output']        
        fitness_dir = sorter_dir.replace('sorted', 'fitness_metrics').replace('sorter_output', '')        
        fitness_save_path = os.path.join(fitness_dir, 'calibrated_fitness.json')
        kwargs['fitness_save_path'] = fitness_save_path
        
        #break deals?
        kwargs['break_deals'] = False # TODO: need to rethink this anyway.
        
        # Max fitness
        kwargs['maxFitness'] = 1000        
        
        return None, kwargs

    def calculate_and_save_fitness(
        break_deals=False,
        #maxFitness=1000,
        fitness_save_path=None,
        source=None,
        #plot_sim = False,
        **kwargs
        ):
        
        """Calculates fitness and saves the results."""
        try:
            assert source is not None, 'Source must be provided.'
            average_fitness, fitnessResults = get_fitness(source, **kwargs)
            fitnessResults['average_fitness'] = average_fitness
            #fitnessResults['maxFitness'] = kwargs['maxFitness']
            #fitnessResults['maxFitness'] = maxFitness

            #break_deals = kwargs.get('break_deals', False)
            if break_deals:
                deal_broken = check_dealbreakers(fitnessResults, **kwargs)
                assert not deal_broken, 'Dealbreakers found in fitness results.'

            #save_fitness_results(kwargs['fitness_save_path'], fitnessResults)
            assert fitness_save_path is not None, 'Fitness save path must be provided.'
            fitness_save_dir = os.path.dirname(fitness_save_path)
            if not os.path.exists(fitness_save_dir):
                os.makedirs(fitness_save_dir)
            save_fitness_results(fitness_save_path, fitnessResults)
            
            #assert average_fitness is an integer, 'Average fitness must be an integer.'
            try: assert isinstance(average_fitness, (int, float)), 'Average fitness must be an integer or float.'
            except:
                #print in all caps
                print('ERROR: AVERAGE FITNESS IS NOT AN INTEGER OR FLOAT')
                import sys
                sys.exit(1)
                
                average_fitness = 1000
                
                # print(f'Average fitness: {average_fitness}')
                # raise
            
            
            return average_fitness
        except Exception as e:
            maxFitness = kwargs.get('maxFitness', 1000)
            error_trace = str(e)
            fitnessResults = {
                #'average_fitness': kwargs['maxFitness'],
                'average_fitness': maxFitness,
                #'maxFitness': kwargs['maxFitness'],
                'maxFitness': maxFitness,
                'error': 'acceptable' if any(error in error_trace for error in []) else 'new',
                'error_trace': error_trace
            }
            print(f'Error calculating fitness: {e}')
            #save_fitness_results(kwargs['fitness_save_path'], fitnessResults)
            save_fitness_results(fitness_save_path, fitnessResults)
            return 1000

    '''execute the main logic of the calculate_fitness function'''
    try:
        # Select mode and execute corresponding logic
        if mode == 'optimizing':
            plot_sim = kwargs.get('plot_sim', False)
            error, kwargs = handle_optimizing_mode(simData, plot_sim, kwargs)
        elif mode == 'simulated':
            error, kwargs = handle_simulated_data_mode(simData, kwargs)
        elif mode == 'experimental':
            error, kwargs = handle_experimental_data_mode(**kwargs)
        else:
            raise ValueError(f'Unknown mode: {mode}')

        # Handle potential errors from network metrics calculation
        if error is not None:
            #return error
            raise ValueError(f'Error in network metrics: {error}')

        # Calculate fitness and return the result
        return calculate_and_save_fitness(**kwargs)
    except Exception as e:
        print(f'Error calculating fitness: {e}')
        fitnessResults = {
            'average_fitness': kwargs.get('maxFitness', 1000),
            'maxFitness': kwargs.get('maxFitness', 1000),
            'error': 'general',
            'error_trace': str(e)
        }
        save_fitness_results(kwargs.get('fitness_save_path', kwargs.get('fitness_save_path', 'unknown_path.json')), fitnessResults)
        #save_fitness_results(kwargs.get('fitness_save_path', 'unknown_path.json'), fitnessResults)
        return 1000
    
'''The Scoring Functions'''        
def the_scoring_function(val, target_val, weight, maxFitness, min_val=None, max_val=None):
    """
    The function `the_scoring_function` calculates a fitness score for a given metric based on its proximity to a target value.
    The goal is to minimize this score, with lower values indicating better fitness. 
    
    The function employs an exponential penalty for deviations from the target value, where the sensitivity of this penalty 
    is controlled by the `weight` parameter. As the weight increases, the penalty for being far from the target decreases, approaching 
    a constant reward of 1 when the difference is small. 
    
    The function also checks if the input value falls within specified bounds (`min_val` and `max_val`); 
    if not specified, it defaults to allowing all values. If the input value is outside the specified bounds, it returns a 
    maximum fitness score (`maxFitness`), indicating poor fitness. 
    
    Overall, the function rewards values close to the target 
    more heavily while penalizing those further away, with the penalty capped at `maxFitness`.
    """
    if val is None:
        return maxFitness
    
    # Set default min and max values if not provided
    if min_val is None:
        min_val = float('-inf')  # Allow all values below positive infinity
    if max_val is None:
        max_val = float('inf')   # Allow all values above negative infinity

    # Calculate fitness score
    if min_val <= val <= max_val:
        #return min(np.exp(abs(target_val - val) / weight), maxFitness)
        return min(np.exp(abs(target_val - val) * weight), maxFitness)
    else:
        return maxFitness

'''fitness functions for the network activity metrics'''
import os
import subprocess

def submit_plotting_job(candidate_path, kwargs):
    """
    Submits the process_simulation step in handle_optimizing_mode as an MPI job.
    """

    # Paths and parameters
    sim_data_path = kwargs.get('data_file_path', None)
    reference_data_path = kwargs.get('reference_data_path', None)
    conv_params = kwargs.get('conv_params', None)
    mega_params = kwargs.get('mega_params', None)
    fitnessFuncArgs = kwargs.copy()

    # Define job folder and script path
    job_folder = os.path.dirname(candidate_path)
    job_name = os.path.basename(candidate_path) + "_fitness"
    job_script = os.path.join(job_folder, job_name + ".sh")

    # Define the command to run process_simulation
    command = f"""
    python -c "
from RBS_network_models.sim_analysis import process_simulation
process_simulation(
    sim_data_path='{sim_data_path}', 
    reference_data_path='{reference_data_path}',
    DEBUG_MODE=False,
    conv_params={conv_params},
    mega_params={mega_params},
    fitnessFuncArgs={fitnessFuncArgs}
)
    "
    """

    # Create the job script for MPI submission
    job_string = f"""#!/bin/bash
cd {job_folder}
{command}
    """

    # Write the script to file
    with open(job_script, "w") as f:
        f.write(job_string)

    # Ensure the script is executable
    os.chmod(job_script, 0o755)

    # Submit the job using MPI direct
    try:
        with open(f"{job_name}.run", "w") as outf, open(f"{job_name}.err", "w") as errf:
            subprocess.Popen(["/bin/bash", job_script], stdout=outf, stderr=errf, start_new_session=True)
        print(f"Submitted plotting job: {job_name}")
    except Exception as e:
        print(f"Failed to submit job {job_name}: {e}")

def fit_firing_rates(data_source, mega_mode=False, **kwargs):
    
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False    
    
    #def fit_firing_rates(simulated=False, **kwargs):
    try:
        print('Calculating firing rate fitness...')
        MeanFireRate_target = kwargs['targets']['spiking_data']['spiking_summary_data']['MeanFireRate']
        
        # E_I_ratio = 5  # 1:5 ratio of E to I neurons
        # E_fr_target = MeanFireRate_target['target'] * (E_I_ratio / (E_I_ratio + 1))
        # I_fr_target = MeanFireRate_target['target'] / (E_I_ratio + 1)
        
        # assumptions
        # max_E_assumption = MeanFireRate_target['max_E_assumption']
        # max_I_assumption = MeanFireRate_target['max_I_assumption']
        # min_E_assumption = MeanFireRate_target['min_E_assumption']
        # min_I_assumption = MeanFireRate_target['min_I_assumption']
        # max_E_assumption_inBurst = MeanFireRate_target['max_E_assumption_inBurst']
        # max_I_assumption_inBurst = MeanFireRate_target['max_I_assumption_inBurst']
        # min_E_assumption_inBurst = MeanFireRate_target['min_E_assumption_inBurst']
        # min_I_assumption_inBurst = MeanFireRate_target['min_I_assumption_inBurst']
        
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        
        # moduleate targets
        # typical case: 
        # target = MeanFireRate_target['target']
        # min_FR = MeanFireRate_target['min']
        # max_FR = MeanFireRate_target['max']
        # weight = MeanFireRate_target['weight']
        target = MeanFireRate_target.get('target', None)
        min_FR = MeanFireRate_target.get('min', None)
        max_FR = MeanFireRate_target.get('max', None)
        weight = MeanFireRate_target.get('weight', None)
        target_E = MeanFireRate_target.get('target_E', None)
        target_I = MeanFireRate_target.get('target_I', None)
        min_FR_E = MeanFireRate_target.get('min_E', None)
        min_FR_I = MeanFireRate_target.get('min_I', None)
        max_FR_E = MeanFireRate_target.get('max_E', None)
        max_FR_I = MeanFireRate_target.get('max_I', None)
        #excitatory_neurons = kwargs['network_metrics']['classification_data']['excitatory_neurons']
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        
        classified_targets = False
        if target_E is not None and target_I is not None:
            classified_targets = True
        
        # try:
        #     target_E = MeanFireRate_target['target_E']
        
        
        maxFitness = kwargs['maxFitness']
        
        spiking_data_by_unit = kwargs['network_metrics']['spiking_data']['spiking_data_by_unit']
       
        # choose bursting data
        if not mega_mode: 
            # typical case
            bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        else:
            # if scoring with mega_bursting_data 
            mega_bursting_data_by_unit = kwargs['network_metrics']['mega_bursting_data']['bursting_data_by_unit']
            bursting_data_by_unit = mega_bursting_data_by_unit
            
            
        fitness_FRs = []
        val_FRs = []
        type_guesses = []
        classifications = []
        for unit, value in spiking_data_by_unit.items():
            try:
            
                unit_data = {
                    # 'spikes': spiking_data_by_unit[unit],
                    # 'bursts': bursting_data_by_unit[unit],
                    'spikes': spiking_data_by_unit.get(unit, {}),
                    'bursts': bursting_data_by_unit.get(unit, {}),
                }
                #val_FR = unit_data['spikes']['FireRate']
                val_FR = unit_data['spikes'].get('FireRate', None)
                val_FRs.append(val_FR)
                
                #''' if experimental '''
                if simulated is False: 
                    #target = MeanFireRate_target['target'] # experimental target
                    if not classified_targets:
                        fitness = the_scoring_function(val_FR, target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
                    elif classified_targets:
                        if unit in excitatory_neurons:
                            E_fitness = the_scoring_function(val_FR, target_E, weight, maxFitness, min_val=min_FR_E, max_val=max_FR_E)
                            fitness = E_fitness
                        elif unit in inhibitory_neurons:
                            I_fitness = the_scoring_function(val_FR, target_I, weight, maxFitness, min_val=min_FR_I, max_val=max_FR_I)
                            fitness = I_fitness
                        else:
                            # unclassified
                            fitness = the_scoring_function(val_FR, target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
                    # if fitness == maxFitness:
                    #     type_guesses.append('unclassified')
                #''' if simulated '''
                else:
                    E_gids = kwargs['network_metrics']['simulated_data']['E_Gids']
                    I_gids = kwargs['network_metrics']['simulated_data']['I_Gids']
                    if unit in E_gids:
                        # E_fitness = the_scoring_function(val_FR, E_fr_target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
                        # fitness = E_fitness
                        E_fitness = the_scoring_function(val_FR, target_E, weight, maxFitness, min_val=min_FR_E, max_val=max_FR_E)
                        fitness = E_fitness
                    elif unit in I_gids:
                        # I_fitness = the_scoring_function(val_FR, I_fr_target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
                        # fitness = I_fitness
                        I_fitness = the_scoring_function(val_FR, target_I, weight, maxFitness, min_val=min_FR_I, max_val=max_FR_I)
                        fitness = I_fitness
            except Exception as e:
                print(f'Error calculating firing rate fitness for unit {unit}.')
                print(e)
                fitness = 1000
            
            # #HACK
            # if np.isnan(val_FR):
            #     print(f'NaN found in firing rate for unit {unit}.')
            
            fitness_FRs.append(fitness)
        fitness_FR = np.mean(fitness_FRs)
    
        # Set to None if not defined
        E_fitness = E_fitness if 'E_fitness' in locals() else None
        I_fitness = I_fitness if 'I_fitness' in locals() else None
        
        # Create dictionary to store fitness values
        fitness_FR_dict = {
            'fit': fitness_FR,
            'value(s)': val_FRs,
            'fit_E': E_fitness,
            'fit_I': I_fitness,
            # 'target_E': E_fr_target,
            # 'target_I': I_fr_target,
            #
            'target_E': target_E,
            'target_I': target_I,
            'min_E': min_FR_E,
            'min_I': min_FR_I,
            'max_E': max_FR_E,
            'max_I': max_FR_I,
            #
            'target': target,
            'min': min_FR,
            'max': max_FR,
            'weight': weight,        
        }
        return fitness_FR_dict
    except Exception as e:
        print(f"Error in fit_firing_rates: {e}")
        return 1000

def fit_CoV_firing_rate(data_source, mega_mode = False, **kwargs):
    
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
            
    try:
        print('Calculating CoV firing rate fitness...')
        CoVFireRate_target = kwargs['targets']['spiking_data']['spiking_summary_data']['CoVFireRate']
        # E_I_ratio = 1.5 / 0.7  # ratio of E to I neurons
        # E_CoV_target = CoVFireRate_target['target'] * (E_I_ratio / (E_I_ratio + 1))
        # I_CoV_target = CoVFireRate_target['target'] / (E_I_ratio + 1)
        # min_CoV = CoVFireRate_target['min']
        # max_CoV = CoVFireRate_target['max']
        # min_CoV = CoVFireRate_target.get('min', None)
        # max_CoV = CoVFireRate_target.get('max', None)
        weight = CoVFireRate_target['weight']
        maxFitness = kwargs['maxFitness']
        CoVFireRate_target_E = CoVFireRate_target.get('target_E', None)
        CoVFireRate_target_I = CoVFireRate_target.get('target_I', None)
        min_CoV = CoVFireRate_target.get('min', None)
        max_CoV = CoVFireRate_target.get('max', None)
        min_CoV_E = CoVFireRate_target.get('min_E', None)
        min_CoV_I = CoVFireRate_target.get('min_I', None)
        max_CoV_E = CoVFireRate_target.get('max_E', None)
        max_CoV_I = CoVFireRate_target.get('max_I', None)
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)     
        spiking_data_by_unit = kwargs['network_metrics']['spiking_data']['spiking_data_by_unit']
        
        fitness_CoVs = []
        val_CoVs = []
        for unit, value in spiking_data_by_unit.items():
            try: val_CoV = spiking_data_by_unit[unit]['CoV_fr']
            except: val_CoV = spiking_data_by_unit[unit]['fr_CoV']
            val_CoVs.append(val_CoV)
            if simulated is False:
                target = CoVFireRate_target['target']
                try: 
                    #fitness = the_scoring_function(val_CoV, target, weight, maxFitness, min_val=min_CoV, max_val=max_CoV)
                    if unit in excitatory_neurons:
                        fitness = the_scoring_function(val_CoV, CoVFireRate_target_E, weight, maxFitness, min_val=min_CoV_E, max_val=max_CoV_E)
                    elif unit in inhibitory_neurons:
                        fitness = the_scoring_function(val_CoV, CoVFireRate_target_I, weight, maxFitness, min_val=min_CoV_I, max_val=max_CoV_I)
                        
                    # if fitness == maxFitness:
                    #     print(f"CoVFR fitness is maxFitness for unit {unit}.")
                except: 
                    fitness = 1000
            else:
                E_gids = kwargs['network_metrics']['simulated_data']['E_Gids']
                I_gids = kwargs['network_metrics']['simulated_data']['I_Gids']
                if unit in E_gids:
                    #fitness = the_scoring_function(val_CoV, E_CoV_target, weight, maxFitness, min_val=min_CoV, max_val=max_CoV)
                    fitness = the_scoring_function(val_CoV, CoVFireRate_target_E, weight, maxFitness, min_val=min_CoV_E, max_val=max_CoV_E)
                elif unit in I_gids:
                    #fitness = the_scoring_function(val_CoV, I_CoV_target, weight, maxFitness, min_val=min_CoV, max_val=max_CoV)
                    fitness = the_scoring_function(val_CoV, CoVFireRate_target_I, weight, maxFitness, min_val=min_CoV_I, max_val=max_CoV_I)
            
            fitness_CoVs.append(fitness)
        
        fitness_CoV = np.mean(fitness_CoVs)
        fitness_CoV_dict = {
            'fit': fitness_CoV,
            'value(s)': val_CoVs,
            # 'target_E': E_CoV_target,
            # 'target_I': I_CoV_target,
            'target_E': CoVFireRate_target_E,
            'target_I': CoVFireRate_target_I,
            # 'min': min_CoV,
            # 'max': max_CoV,
            'min_E': min_CoV_E,
            'min_I': min_CoV_I,
            'max_E': max_CoV_E,
            'max_I': max_CoV_I,
            'min': min_CoV,
            'max': max_CoV,
            'weight': weight,
        }
        return fitness_CoV_dict
    except Exception as e:
        print(f"Error in fit_CoV_firing_rate: {e}")
        return 1000

def fit_ISI(data_source, **kwargs):
    
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
    
    try:
        print('Calculating ISI fitness...')

        MeanISI_targets = kwargs['targets']['spiking_data']['spiking_summary_data']['MeanISI']
        #E_I_ratio_mean_ISI = 5.0 / 1.0  # Biologically founded E to I ratio for Mean ISI

        # Calculate targets for E and I populations
        #E_meanISI_target = MeanISI_target['target'] * (E_I_ratio_mean_ISI / (E_I_ratio_mean_ISI + 1))
        #I_meanISI_target = MeanISI_target['target'] / (E_I_ratio_mean_ISI + 1)

        # ISI bounds and weights
        min_ISI = MeanISI_targets['min']
        max_ISI = MeanISI_targets['max']
        weight = MeanISI_targets['weight']
        maxFitness = kwargs['maxFitness']
        MeanISI_target = MeanISI_targets.get('target', None)
        
        MeanISI_target_E = MeanISI_targets.get('target_E', None)
        MeanISI_target_I = MeanISI_targets.get('target_I', None)
        min_ISI_E = MeanISI_targets.get('min_E', None)
        min_ISI_I = MeanISI_targets.get('min_I', None)
        max_ISI_E = MeanISI_targets.get('max_E', None)
        max_ISI_I = MeanISI_targets.get('max_I', None)
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        

        spiking_data_by_unit = kwargs['network_metrics']['spiking_data']['spiking_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_ISIs = []
        val_ISIs = []

        for unit, value in spiking_data_by_unit.items():
            try:
                val_ISI = spiking_data_by_unit[unit]['meanISI']
                val_ISIs.append(val_ISI)

                if simulated:
                    # Use population-specific targets for simulated data
                    if unit in E_Gids:
                        #fitness = the_scoring_function(val_ISI, E_meanISI_target, weight, maxFitness, 
                                                    #    min_val=min_ISI, max_val=max_ISI)
                        fitness = the_scoring_function(val_ISI, MeanISI_target_E, weight, maxFitness, 
                                                       min_val=min_ISI_E, max_val=max_ISI_E)
                    elif unit in I_Gids:
                        #fitness = the_scoring_function(val_ISI, I_meanISI_target, weight, maxFitness, 
                                                       #min_val=min_ISI, max_val=max_ISI)
                        fitness = the_scoring_function(val_ISI, MeanISI_target_I, weight, maxFitness,
                                                         min_val=min_ISI_I, max_val=max_ISI_I)
                    else:
                        # Handle unclassified units
                        fitness = the_scoring_function(val_ISI, MeanISI_target, weight, maxFitness, 
                                                       min_val=min_ISI, max_val=max_ISI)
                else:
                    # # Use undifferentiated target for experimental data
                    # fitness = the_scoring_function(val_ISI, MeanISI_target['target'], weight, maxFitness, 
                    #                                min_val=min_ISI, max_val=max_ISI)
                    try:
                        if unit in excitatory_neurons:
                            fitness = the_scoring_function(val_ISI, MeanISI_target_E, weight, maxFitness, 
                                                           min_val=min_ISI_E, max_val=max_ISI_E)
                        elif unit in inhibitory_neurons:
                            fitness = the_scoring_function(val_ISI, MeanISI_target_I, weight, maxFitness, 
                                                           min_val=min_ISI_I, max_val=max_ISI_I)
                        else:
                            # Handle unclassified units
                            fitness = the_scoring_function(val_ISI, MeanISI_target, weight, maxFitness, 
                                                           min_val=min_ISI, max_val=max_ISI)
                    except:
                        # fitness = the_scoring_function(val_ISI, MeanISI_target['target'], weight, maxFitness, 
                        #                                min_val=min_ISI, max_val=max_ISI)
                        fitness = maxFitness
                fitness_ISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_ISI = {
            'fit': np.mean(fitness_ISIs) if fitness_ISIs else None,
            'value(s)': val_ISIs,
            # 'target_E': E_meanISI_target,
            # 'target_I': I_meanISI_target,
            'target': MeanISI_target,
            'min': min_ISI,
            'max': max_ISI,
            
            'target_E': MeanISI_target_E,
            'target_I': MeanISI_target_I,
            'min_E': min_ISI_E,
            'min_I': min_ISI_I,
            'max_E': max_ISI_E,
            'max_I': max_ISI_I,
            
            'weight': weight,
        }

        return fitness_ISI

    except Exception as e:
        print(f"Error in fit_ISI: {e}")
        return 1000

#def fit_CoV_ISI(simulated=False, **kwargs):
def fit_CoV_ISI(data_source, **kwargs):
    
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
        
    try:
        print('Calculating CoV ISI fitness...')

        CoV_ISI_targets = kwargs['targets']['spiking_data']['spiking_summary_data']['CoV_ISI']
        # E_I_ratio_CoV_ISI = 1.5 / 0.7  # Reflects variability differences

        # # Calculate targets for E and I populations
        # E_CoV_ISI_target = CoV_ISI_target['target'] * (E_I_ratio_CoV_ISI / (E_I_ratio_CoV_ISI + 1))
        # I_CoV_ISI_target = CoV_ISI_target['target'] / (E_I_ratio_CoV_ISI + 1)

        # CoV ISI bounds and weights
        # min_CoV_ISI = CoV_ISI_target['min']
        # max_CoV_ISI = CoV_ISI_target['max']
        CoV_ISI_target = CoV_ISI_targets.get('target', None)
        min_CoV_ISI = CoV_ISI_targets.get('min', None)
        max_CoV_ISI = CoV_ISI_targets.get('max', None)
        CoV_ISI_target_E = CoV_ISI_targets.get('target_E', None)
        CoV_ISI_target_I = CoV_ISI_targets.get('target_I', None)
        min_CoV_ISI_E = CoV_ISI_targets.get('min_E', None)
        min_CoV_ISI_I = CoV_ISI_targets.get('min_I', None)
        max_CoV_ISI_E = CoV_ISI_targets.get('max_E', None)
        max_CoV_ISI_I = CoV_ISI_targets.get('max_I', None)
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        
        
        weight = CoV_ISI_targets['weight']
        maxFitness = kwargs['maxFitness']

        spiking_data_by_unit = kwargs['network_metrics']['spiking_data']['spiking_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_CoV_ISIs = []
        val_CoV_ISIs = []

        for unit, value in spiking_data_by_unit.items():
            try:
                try: 
                    val_CoV_ISI = spiking_data_by_unit[unit]['CoV_ISI']
                except: 
                    val_CoV_ISI = spiking_data_by_unit[unit]['isi_CoV']
                val_CoV_ISIs.append(val_CoV_ISI)

                if simulated:
                    if unit in E_Gids:
                        # fitness = the_scoring_function(val_CoV_ISI, E_CoV_ISI_target, weight, maxFitness, 
                        #                                min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                        fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target_E, weight, maxFitness, 
                                                       min_val=min_CoV_ISI_E, max_val=max_CoV_ISI_E)
                    elif unit in I_Gids:
                        # fitness = the_scoring_function(val_CoV_ISI, I_CoV_ISI_target, weight, maxFitness, 
                        #                                min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                        fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target_I, weight, maxFitness, 
                                                       min_val=min_CoV_ISI_I, max_val=max_CoV_ISI_I)
                    else:
                        # Handle unclassified units
                        fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target, weight, maxFitness, 
                                                       min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                else:
                    # fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target['target'], weight, maxFitness, 
                    #                                min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                    try:
                        if unit in excitatory_neurons:
                            fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target_E, weight, maxFitness, 
                                                           min_val=min_CoV_ISI_E, max_val=max_CoV_ISI_E)
                        elif unit in inhibitory_neurons:
                            fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target_I, weight, maxFitness, 
                                                           min_val=min_CoV_ISI_I, max_val=max_CoV_ISI_I)
                        else:
                            # Handle unclassified units
                            fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target, weight, maxFitness, 
                                                           min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                    except:
                        # fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target['target'], weight, maxFitness, 
                        #                                min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                        fitness = maxFitness
                    

                fitness_CoV_ISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_CoV_ISI = {
            'fit': np.mean(fitness_CoV_ISIs) if fitness_CoV_ISIs else None,
            'value(s)': val_CoV_ISIs,
            # 'target_E': E_CoV_ISI_target,
            # 'target_I': I_CoV_ISI_target,
            'target': CoV_ISI_target,
            'min': min_CoV_ISI,
            'max': max_CoV_ISI,
            'target_E': CoV_ISI_target_E,
            'target_I': CoV_ISI_target_I,
            'min_E': min_CoV_ISI_E,
            'min_I': min_CoV_ISI_I,
            'max_E': max_CoV_ISI_E,
            'max_I': max_CoV_ISI_I,
            
            'weight': weight,
        }

        return fitness_CoV_ISI

    except Exception as e:
        print(f"Error in fit_CoV_ISI: {e}")
        return 1000

def fit_mean_Burst_Rate(data_source, mega_mode = False, **kwargs):
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
    
    try:
        print('Calculating mean burst rate fitness...')

        try:
            if not mega_mode: MeanBurstRate_targets = kwargs['targets']['bursting_data']['bursting_summary_data']['MeanBurstRate'] #TODO rerun analysis to get the correct target values
            else: MeanBurstRate_targets = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['MeanBurstRate']
        except Exception as e: #if the program fails here print a big yellow warning to fix the knon issue.
            #activate_print()
            print(f"\033[93mWarning: {e}\033[0m")
            print(f"\033[93mWarning: The target values for mean_Burst_Rate are not available. Please rerun the analysis to get the correct target values.\033[0m")
            print(f"\033[93mWarning: Setting the fitness score to 1000.\033[0m")
            #suppress_print()
            return 1000
        #E_I_ratio = 2.0 / 1.0

        #E_meanBurstRate_target = MeanBurstRate_target['target'] * (E_I_ratio / (E_I_ratio + 1))
        #I_meanBurstRate_target = MeanBurstRate_target['target'] / (E_I_ratio + 1)

        MeanBurstRate_target = MeanBurstRate_targets.get('target', None)
        # min_meanBurstRate = MeanBurstRate_targets['min']
        # max_meanBurstRate = MeanBurstRate_targets['max']
        # weight = MeanBurstRate_targets['weight']
        # maxFitness = kwargs['maxFitness']
        min_meanBurstRate = MeanBurstRate_targets.get('min', None)
        max_meanBurstRate = MeanBurstRate_targets.get('max', None)
        weight = MeanBurstRate_targets.get('weight', None)
        maxFitness = kwargs['maxFitness']

        if not mega_mode: mean_Burst_Rate = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Rate']
        else: mean_Burst_Rate = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['mean_Burst_Rate']
        
        mean_Burst_Rate_fitness = the_scoring_function(mean_Burst_Rate, MeanBurstRate_targets['target'], weight, maxFitness,)
        
        fitness_meanBurstRate = {
            'fit': mean_Burst_Rate_fitness,
            'value(s)': mean_Burst_Rate,
            'target' : MeanBurstRate_targets['target'],
            'min': min_meanBurstRate,
            'max': max_meanBurstRate,
            'weight': weight,
        }
        
        return fitness_meanBurstRate
    except Exception as e:
        print(f"Error in fit_mean_Burst_Rate: {e}")
        return 1000
    
def fit_baseline(data_source, mega_mode = False, **kwargs):
    try:
        print('Calculating baseline fitness...')

        if not mega_mode: baseline_target = kwargs['targets']['bursting_data']['bursting_summary_data']['baseline']
        else: baseline_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['baseline']
        
        target = baseline_target['target']
        min_baseline = baseline_target['min']
        max_baseline = baseline_target['max']
        weight = baseline_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: baseline = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['baseline']
        else: baseline = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['baseline']
        val_baseline = baseline

        fitness_baseline = the_scoring_function(val_baseline, target, weight, maxFitness, 
                                                min_val=min_baseline, max_val=max_baseline)

        fitness_baseline_dict = {
            'fit': fitness_baseline,
            'value': val_baseline,
            'target': target,
            'min': min_baseline,
            'max': max_baseline,
            'weight': weight,
        }

        return fitness_baseline_dict

    except Exception as e:
        print(f"Error in fit_baseline: {e}")
        return 1000

def fit_WithinBurstISI(data_source, mega_mode = False, **kwargs):
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
    
    try:
        print('Calculating WithinBurstISI fitness...')

        if not mega_mode: WithinBurstISI_targets = kwargs['targets']['bursting_data']['bursting_summary_data']['MeanWithinBurstISI']
        else: WithinBurstISI_targets = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['MeanWithinBurstISI']
        #E_I_ratio = 2.0 / 1.0

        # E_WithinBurstISI_target = WithinBurstISI_target['target'] * (E_I_ratio / (E_I_ratio + 1))
        # I_WithinBurstISI_target = WithinBurstISI_target['target'] / (E_I_ratio + 1)
        WithinBurstISI_target = WithinBurstISI_targets.get('target', None)
        # min_WithinBurstISI = WithinBurstISI_targets['min']
        # max_WithinBurstISI = WithinBurstISI_targets['max']
        
        min_WithinBurstISI = WithinBurstISI_targets.get('min', None)
        max_WithinBurstISI = WithinBurstISI_targets.get('max', None)
        
        E_WithinBurstISI_target = WithinBurstISI_targets.get('target_E', None)
        I_WithinBurstISI_target = WithinBurstISI_targets.get('target_I', None)
        E_min_WithinBurstISI = WithinBurstISI_targets.get('min_E', None)
        E_max_WithinBurstISI = WithinBurstISI_targets.get('max_E', None)
        I_min_WithinBurstISI = WithinBurstISI_targets.get('min_I', None)
        I_max_WithinBurstISI = WithinBurstISI_targets.get('max_I', None)
        
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        
        weight = WithinBurstISI_targets['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        else: bursting_data_by_unit = kwargs['network_metrics']['mega_bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_WithinBurstISIs = []
        val_WithinBurstISIs = []

        scored_by_individual = False
        if scored_by_individual:
            for unit, value in bursting_data_by_unit.items():
                try:
                    val_WithinBurstISI = bursting_data_by_unit[unit]['mean_isi_within']
                    val_WithinBurstISIs.append(val_WithinBurstISI)

                    if simulated:
                        if unit in E_Gids:
                            # fitness = the_scoring_function(val_WithinBurstISI, E_WithinBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                            fitness = the_scoring_function(val_WithinBurstISI, E_WithinBurstISI_target, weight, maxFitness, 
                                                        min_val=E_min_WithinBurstISI, max_val=E_max_WithinBurstISI)
                        elif unit in I_Gids:
                            # fitness = the_scoring_function(val_WithinBurstISI, I_WithinBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                            fitness = the_scoring_function(val_WithinBurstISI, I_WithinBurstISI_target, weight, maxFitness,
                                                            min_val=I_min_WithinBurstISI, max_val=I_max_WithinBurstISI)
                        else:
                            # Handle unclassified units
                            fitness = the_scoring_function(val_WithinBurstISI, WithinBurstISI_target, weight, maxFitness, 
                                                        min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                    else:
                        # fitness = the_scoring_function(val_WithinBurstISI, WithinBurstISI_target['target'], weight, maxFitness, 
                        #                                min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                        try:
                            if unit in excitatory_neurons:
                                fitness = the_scoring_function(val_WithinBurstISI, E_WithinBurstISI_target, weight, maxFitness, 
                                                            min_val=E_min_WithinBurstISI, max_val=E_max_WithinBurstISI)
                            elif unit in inhibitory_neurons:
                                fitness = the_scoring_function(val_WithinBurstISI, I_WithinBurstISI_target, weight, maxFitness,
                                                            min_val=I_min_WithinBurstISI, max_val=I_max_WithinBurstISI)
                            else:
                                # Handle unclassified units
                                fitness = the_scoring_function(val_WithinBurstISI, WithinBurstISI_target, weight, maxFitness, 
                                                            min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                        except:
                            # fitness = the_scoring_function(val_WithinBurstISI, WithinBurstISI_target['target'], weight, maxFitness, 
                            #                                min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                            fitness = maxFitness

                    fitness_WithinBurstISIs.append(fitness)
                except Exception as unit_error:
                    print(f"Error processing unit {unit}: {unit_error}")
                    continue  # Skip to the next unit if there's an issue
        else:
            if not mega_mode: val_WithinBurstISIs = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['MeanWithinBurstISI']
            else: val_WithinBurstISIs = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['MeanWithinBurstISI']
            
            try:
                fitness = the_scoring_function(val_WithinBurstISIs, WithinBurstISI_target, weight, maxFitness, min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
            except:
                fitness = maxFitness
            fitness_WithinBurstISIs.append(fitness)

        fitness_WithinBurstISI = {
            'fit': np.mean(fitness_WithinBurstISIs) if fitness_WithinBurstISIs else None,
            'value(s)': val_WithinBurstISIs,
            # 'target_E': E_WithinBurstISI_target,
            # 'target_I': I_WithinBurstISI_target,
            'target': WithinBurstISI_target,
            'min': min_WithinBurstISI,
            'max': max_WithinBurstISI,
            'target_E': E_WithinBurstISI_target,
            'target_I': I_WithinBurstISI_target,
            'min_E': E_min_WithinBurstISI,
            'min_I': I_min_WithinBurstISI,
            'max_E': E_max_WithinBurstISI,
            'max_I': I_max_WithinBurstISI,
            'weight': weight,
        }

        return fitness_WithinBurstISI

    except Exception as e:
        print(f"Error in fit_WithinBurstISI: {e}")
        return 1000

def fit_CovWithinBurstISI(data_source, mega_mode = False, **kwargs):
    # source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
        
    try:
        print('Calculating CovWithinBurstISI fitness...')

        if not mega_mode: CovWithinBurstISI_targets = kwargs['targets']['bursting_data']['bursting_summary_data']['CovWithinBurstISI']
        else: CovWithinBurstISI_targets = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['CovWithinBurstISI']
        # E_I_ratio_burst = 2.5 / 1.0

        # E_CovWithinBurstISI_target = CovWithinBurstISI_target['target'] * (E_I_ratio_burst / (E_I_ratio_burst + 1))
        # I_CovWithinBurstISI_target = CovWithinBurstISI_target['target'] / (E_I_ratio_burst + 1)

        CovWithinBurstISI_target = CovWithinBurstISI_targets.get('target', None)
        # min_CovWithinBurstISI = CovWithinBurstISI_targets['min']
        # max_CovWithinBurstISI = CovWithinBurstISI_targets['max']
        min_CovWithinBurstISI = CovWithinBurstISI_targets.get('min', None)
        max_CovWithinBurstISI = CovWithinBurstISI_targets.get('max', None)
        E_CovWithinBurstISI_target = CovWithinBurstISI_targets.get('target_E', None)
        I_CovWithinBurstISI_target = CovWithinBurstISI_targets.get('target_I', None)
        min_CovWithinBurstISI_I = CovWithinBurstISI_targets.get('min_I', None)
        min_CovWithinBurstISI_E = CovWithinBurstISI_targets.get('min_E', None)
        max_CovWithinBurstISI_I = CovWithinBurstISI_targets.get('max_I', None)
        max_CovWithinBurstISI_E = CovWithinBurstISI_targets.get('max_E', None)
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        
        weight = CovWithinBurstISI_targets['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        else: bursting_data_by_unit = kwargs['network_metrics']['mega_bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_CovWithinBurstISIs = []
        val_CovWithinBurstISIs = []

        scored_by_individual_units = False
        if scored_by_individual_units:
            for unit, value in bursting_data_by_unit.items():
                try:
                    val_CovWithinBurstISI = bursting_data_by_unit[unit]['cov_isi_within']
                    val_CovWithinBurstISIs.append(val_CovWithinBurstISI)

                    if simulated:
                        if unit in E_Gids:
                            # fitness = the_scoring_function(val_CovWithinBurstISI, E_CovWithinBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
                            fitness = the_scoring_function(val_CovWithinBurstISI, E_CovWithinBurstISI_target, weight, maxFitness, 
                                                        min_val=min_CovWithinBurstISI_E, max_val=max_CovWithinBurstISI_E)
                        elif unit in I_Gids:
                            # fitness = the_scoring_function(val_CovWithinBurstISI, I_CovWithinBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
                            fitness = the_scoring_function(val_CovWithinBurstISI, I_CovWithinBurstISI_target, weight, maxFitness,
                                                            min_val=min_CovWithinBurstISI_I, max_val=max_CovWithinBurstISI_I)
                    else:
                        # fitness = the_scoring_function(val_CovWithinBurstISI, CovWithinBurstISI_target['target'], weight, maxFitness, 
                        #                                min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
                        try:
                            if unit in excitatory_neurons:
                                fitness = the_scoring_function(val_CovWithinBurstISI, E_CovWithinBurstISI_target, weight, maxFitness, 
                                                            min_val=min_CovWithinBurstISI_E, max_val=max_CovWithinBurstISI_E)
                            elif unit in inhibitory_neurons:
                                fitness = the_scoring_function(val_CovWithinBurstISI, I_CovWithinBurstISI_target, weight, maxFitness, 
                                                            min_val=min_CovWithinBurstISI_I, max_val=max_CovWithinBurstISI_I)
                            else:
                                # Handle unclassified units
                                fitness = the_scoring_function(val_CovWithinBurstISI, CovWithinBurstISI_target['target'], weight, maxFitness, 
                                                            min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
                        except:
                            # fitness = the_scoring_function(val_CovWithinBurstISI, CovWithinBurstISI_target['target'], weight, maxFitness, 
                            #                                min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
                            fitness = maxFitness

                    fitness_CovWithinBurstISIs.append(fitness)
                except Exception as unit_error:
                    print(f"Error processing unit {unit}: {unit_error}")
                    continue  # Skip to the next unit if there's an issue
        else:
            if not mega_mode: val_CovWithinBurstISI = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['CoVWithinBurstISI']
            else: val_CovWithinBurstISI = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['CoVWithinBurstISI']
            val_CovWithinBurstISIs.append(val_CovWithinBurstISI)
            try:
                fitness = the_scoring_function(val_CovWithinBurstISI, CovWithinBurstISI_target, weight, maxFitness, 
                                            min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
            except:
                fitness = maxFitness
            fitness_CovWithinBurstISIs.append(fitness)

        fitness_CovWithinBurstISI = {
            'fit': np.mean(fitness_CovWithinBurstISIs) if fitness_CovWithinBurstISIs else None,
            'value(s)': val_CovWithinBurstISIs,
            # 'target_E': E_CovWithinBurstISI_target,
            # 'target_I': I_CovWithinBurstISI_target,
            'target': CovWithinBurstISI_target,
            'min': min_CovWithinBurstISI,
            'max': max_CovWithinBurstISI,
            'target_E': E_CovWithinBurstISI_target,
            'target_I': I_CovWithinBurstISI_target,
            'min_E': min_CovWithinBurstISI_E,
            'min_I': min_CovWithinBurstISI_I,
            'max_E': max_CovWithinBurstISI_E,
            'max_I': max_CovWithinBurstISI_I,
            
            'weight': weight,
        }

        return fitness_CovWithinBurstISI

    except Exception as e:
        print(f"Error in fit_CovWithinBurstISI: {e}")
        return 1000

def fit_OutsideBurstISI(data_source, mega_mode = False, **kwargs):
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
        
    try:
        print('Calculating OutsideBurstISI fitness...')

        if not mega_mode: OutsideBurstISI_targets = kwargs['targets']['bursting_data']['bursting_summary_data']['MeanOutsideBurstISI']
        else: OutsideBurstISI_targets = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['MeanOutsideBurstISI']
        # E_I_ratio_outside_mean_ISI = 5.0 / 1.0

        # E_OutsideBurstISI_target = OutsideBurstISI_target['target'] * (E_I_ratio_outside_mean_ISI / (E_I_ratio_outside_mean_ISI + 1))
        # I_OutsideBurstISI_target = OutsideBurstISI_target['target'] / (E_I_ratio_outside_mean_ISI + 1)

        OutsideBurstISI_target = OutsideBurstISI_targets.get('target', None)                    
        # min_OutsideBurstISI = OutsideBurstISI_targets['min']
        # max_OutsideBurstISI = OutsideBurstISI_targets['max']
        min_OutsideBurstISI = OutsideBurstISI_targets.get('min', None)
        max_OutsideBurstISI = OutsideBurstISI_targets.get('max', None)
        E_OutsideBurstISI_target = OutsideBurstISI_targets.get('target_E', None)
        I_OutsideBurstISI_target = OutsideBurstISI_targets.get('target_I', None)
        E_min_OutsideBurstISI = OutsideBurstISI_targets.get('min_E', None)
        E_max_OutsideBurstISI = OutsideBurstISI_targets.get('max_E', None)
        I_min_OutsideBurstISI = OutsideBurstISI_targets.get('min_I', None)
        I_max_OutsideBurstISI = OutsideBurstISI_targets.get('max_I', None)
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        
        weight = OutsideBurstISI_targets['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        else: bursting_data_by_unit = kwargs['network_metrics']['mega_bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_OutsideBurstISIs = []
        val_OutsideBurstISIs = []
        
        score_by_inidividual = False
        if score_by_inidividual:
            for unit, value in bursting_data_by_unit.items():
                try:
                    val_OutsideBurstISI = bursting_data_by_unit[unit]['mean_isi_outside']
                    val_OutsideBurstISIs.append(val_OutsideBurstISI)

                    if simulated:
                        if unit in E_Gids:
                            # fitness = the_scoring_function(val_OutsideBurstISI, E_OutsideBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                            fitness = the_scoring_function(val_OutsideBurstISI, E_OutsideBurstISI_target, weight, maxFitness, 
                                                        min_val=E_min_OutsideBurstISI, max_val=E_max_OutsideBurstISI)
                        elif unit in I_Gids:
                            # fitness = the_scoring_function(val_OutsideBurstISI, I_OutsideBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                            fitness = the_scoring_function(val_OutsideBurstISI, I_OutsideBurstISI_target, weight, maxFitness,
                                                            min_val=I_min_OutsideBurstISI, max_val=I_max_OutsideBurstISI)
                    else:
                        # fitness = the_scoring_function(val_OutsideBurstISI, OutsideBurstISI_target['target'], weight, maxFitness, 
                        #                                min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                        try:
                            if unit in excitatory_neurons:
                                fitness = the_scoring_function(val_OutsideBurstISI, E_OutsideBurstISI_target, weight, maxFitness, 
                                                            min_val=E_min_OutsideBurstISI, max_val=E_max_OutsideBurstISI)
                            elif unit in inhibitory_neurons:
                                fitness = the_scoring_function(val_OutsideBurstISI, I_OutsideBurstISI_target, weight, maxFitness,
                                                            min_val=I_min_OutsideBurstISI, max_val=I_max_OutsideBurstISI)
                            else:
                                # Handle unclassified units
                                fitness = the_scoring_function(val_OutsideBurstISI, OutsideBurstISI_target['target'], weight, maxFitness, 
                                                            min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                        except:
                            # fitness = the_scoring_function(val_OutsideBurstISI, OutsideBurstISI_target['target'], weight, maxFitness, 
                            #                                min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                            fitness = maxFitness

                    
                    if fitness == maxFitness:
                        print(f"Warning: Unit {unit} has a fitness score of {maxFitness}. This may indicate an issue with the data or the scoring function. {val_OutsideBurstISI}")
                        print()
                    fitness_OutsideBurstISIs.append(fitness)
                except Exception as unit_error:
                    print(f"Error processing unit {unit}: {unit_error}")
                    continue  # Skip to the next unit if there's an issue
        else:
            if not mega_mode: val = network_metrics['bursting_data']['bursting_summary_data']['MeanOutsideBurstISI']
            else: val = network_metrics['mega_bursting_data']['bursting_summary_data']['MeanOutsideBurstISI']
            val_OutsideBurstISIs.append(val)
            try:
                if simulated:
                    fitness = the_scoring_function(val, OutsideBurstISI_target, weight, maxFitness, 
                                                    min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                else:
                    fitness = the_scoring_function(val, OutsideBurstISI_target, weight, maxFitness, 
                                                    min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
            except:
                fitness = maxFitness
            fitness_OutsideBurstISIs.append(fitness)
            
        fitness_OutsideBurstISI = {
            'fit': np.mean(fitness_OutsideBurstISIs) if fitness_OutsideBurstISIs else None,
            'value(s)': val_OutsideBurstISIs,
            'fit_scores': fitness_OutsideBurstISIs,
            # 'target_E': E_OutsideBurstISI_target,
            # 'target_I': I_OutsideBurstISI_target,
            'target': OutsideBurstISI_target,
            'min': min_OutsideBurstISI,
            'max': max_OutsideBurstISI,
            'target_E': E_OutsideBurstISI_target,
            'target_I': I_OutsideBurstISI_target,
            'min_E': E_min_OutsideBurstISI,
            'min_I': I_min_OutsideBurstISI,
            'max_E': E_max_OutsideBurstISI,
            'max_I': I_max_OutsideBurstISI,
            
            'weight': weight,
        }

        return fitness_OutsideBurstISI

    except Exception as e:
        print(f"Error in fit_OutsideBurstISI: {e}")
        return 1000

def fit_CovOutsideBurstISI(data_source, mega_mode = False, **kwargs):
    
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
    
    try:
        print('Calculating CovOutsideBurstISI fitness...')

        if not mega_mode: CovOutsideBurstISI_targets = kwargs['targets']['bursting_data']['bursting_summary_data']['CoVOutsideBurstISI']
        else: CovOutsideBurstISI_targets = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['CoVOutsideBurstISI']
        # E_I_ratio_outside_CoV_ISI = 1.5 / 0.8

        # E_CovOutsideBurstISI_target = CovOutsideBurstISI_target['target'] * (E_I_ratio_outside_CoV_ISI / (E_I_ratio_outside_CoV_ISI + 1))
        # I_CovOutsideBurstISI_target = CovOutsideBurstISI_target['target'] / (E_I_ratio_outside_CoV_ISI + 1)

        # min_CovOutsideBurstISI = CovOutsideBurstISI_target['min']
        # max_CovOutsideBurstISI = CovOutsideBurstISI_target['max']
        CovOutsideBurstISI_min = CovOutsideBurstISI_targets.get('min', None)
        CovOutsideBurstISI_max = CovOutsideBurstISI_targets.get('max', None)
        CovOutsideBurstISI_target = CovOutsideBurstISI_targets.get('target', None)
        E_CovOutsideBurstISI_target = CovOutsideBurstISI_targets.get('target_E', None)
        I_CovOutsideBurstISI_target = CovOutsideBurstISI_targets.get('target_I', None)
        min_CovOutsideBurstISI_E = CovOutsideBurstISI_targets.get('min_E', None)
        min_CovOutsideBurstISI_I = CovOutsideBurstISI_targets.get('min_I', None)
        max_CovOutsideBurstISI_E = CovOutsideBurstISI_targets.get('max_E', None)
        max_CovOutsideBurstISI_I = CovOutsideBurstISI_targets.get('max_I', None)
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        
        weight = CovOutsideBurstISI_targets['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        else: bursting_data_by_unit = kwargs['network_metrics']['mega_bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_CovOutsideBurstISIs = []
        val_CovOutsideBurstISIs = []
        
        scored_by_individual = False
        if scored_by_individual:
            for unit, value in bursting_data_by_unit.items():
                try:
                    val_CovOutsideBurstISI = bursting_data_by_unit[unit]['cov_isi_outside']
                    val_CovOutsideBurstISIs.append(val_CovOutsideBurstISI)

                    if simulated:
                        if unit in E_Gids:
                            # fitness = the_scoring_function(val_CovOutsideBurstISI, E_CovOutsideBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)
                            fitness = the_scoring_function(val_CovOutsideBurstISI, E_CovOutsideBurstISI_target, weight, maxFitness, 
                                                        min_val=min_CovOutsideBurstISI_E, max_val=max_CovOutsideBurstISI_E)
                        elif unit in I_Gids:
                            # fitness = the_scoring_function(val_CovOutsideBurstISI, I_CovOutsideBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)
                            fitness = the_scoring_function(val_CovOutsideBurstISI, I_CovOutsideBurstISI_target, weight, maxFitness,
                                                        min_val=min_CovOutsideBurstISI_I, max_val=max_CovOutsideBurstISI_I)
                    else:
                        # fitness = the_scoring_function(val_CovOutsideBurstISI, CovOutsideBurstISI_target['target'], weight, maxFitness, 
                        #                                min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)
                        try:
                            if unit in inhibitory_neurons:
                                fitness = the_scoring_function(val_CovOutsideBurstISI, I_CovOutsideBurstISI_target, weight, maxFitness, 
                                                            min_val=min_CovOutsideBurstISI_I, max_val=max_CovOutsideBurstISI_I)
                            elif unit in excitatory_neurons:
                                fitness = the_scoring_function(val_CovOutsideBurstISI, E_CovOutsideBurstISI_target, weight, maxFitness, 
                                                            min_val=min_CovOutsideBurstISI_E, max_val=max_CovOutsideBurstISI_E)
                            else:
                                # # Handle unclassified units
                                # fitness = the_scoring_function(val_CovOutsideBurstISI, CovOutsideBurstISI_target['target'], weight, maxFitness, 
                                #                                min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)
                                fitness = the_scoring_function(val_CovOutsideBurstISI, CovOutsideBurstISI_target, weight, maxFitness,
                                                                min_val=CovOutsideBurstISI_min, max_val=CovOutsideBurstISI_max)
                        except:
                            # fitness = the_scoring_function(val_CovOutsideBurstISI, CovOutsideBurstISI_target['target'], weight, maxFitness, 
                            #                                min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)
                            fitness = maxFitness

                    fitness_CovOutsideBurstISIs.append(fitness)
                except Exception as unit_error:
                    print(f"Error processing unit {unit}: {unit_error}")
                    continue  # Skip to the next unit if there's an issue
        else:
            if not mega_mode: val = network_metrics['bursting_data']['bursting_summary_data']['CoVOutsideBurstISI']
            else: val = network_metrics['mega_bursting_data']['bursting_summary_data']['CoVOutsideBurstISI']
            val_CovOutsideBurstISIs.append(val)
            try:
                if simulated:
                    fitness = the_scoring_function(val, CovOutsideBurstISI_target, weight, maxFitness, 
                                                    min_val=CovOutsideBurstISI_min, max_val=CovOutsideBurstISI_max)
                else:
                    fitness = the_scoring_function(val, CovOutsideBurstISI_target, weight, maxFitness, 
                                                    min_val=CovOutsideBurstISI_min, max_val=CovOutsideBurstISI_max)
            except:
                fitness = maxFitness
            fitness_CovOutsideBurstISIs.append(fitness)
            

        fitness_CovOutsideBurstISI = {
            'fit': np.mean(fitness_CovOutsideBurstISIs) if fitness_CovOutsideBurstISIs else None,
            'value(s)': val_CovOutsideBurstISIs,
            # 'target_E': E_CovOutsideBurstISI_target,
            # 'target_I': I_CovOutsideBurstISI_target,
            # 'min': min_CovOutsideBurstISI,
            # 'max': max_CovOutsideBurstISI,
            'target': CovOutsideBurstISI_target,
            'min': CovOutsideBurstISI_min,
            'max': CovOutsideBurstISI_max,
            'target_E': E_CovOutsideBurstISI_target,
            'target_I': I_CovOutsideBurstISI_target,
            'min_E': min_CovOutsideBurstISI_E,
            'min_I': min_CovOutsideBurstISI_I,
            'max_E': max_CovOutsideBurstISI_E,
            'max_I': max_CovOutsideBurstISI_I,
            
            'weight': weight,
        }

        return fitness_CovOutsideBurstISI

    except Exception as e:
        print(f"Error in fit_CovOutsideBurstISI: {e}")
        return 1000

def fit_NetworkISI(data_source, mega_mode = False, **kwargs):
    
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
    
    
    try:
        print('Calculating NetworkISI fitness...')

        NetworkISI_targets = kwargs['targets']['bursting_data']['bursting_summary_data']['MeanNetworkISI']
        # E_I_ratio_network_mean_ISI = 4.0 / 1.0

        # E_NetworkISI_target = NetworkISI_target['target'] * (E_I_ratio_network_mean_ISI / (E_I_ratio_network_mean_ISI + 1))
        # I_NetworkISI_target = NetworkISI_target['target'] / (E_I_ratio_network_mean_ISI + 1)

        NetworkISI_target = NetworkISI_targets.get('target', None)
        min_NetworkISI = NetworkISI_targets['min']
        max_NetworkISI = NetworkISI_targets['max']
        weight = NetworkISI_targets['weight']
        maxFitness = kwargs['maxFitness']

        bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_NetworkISIs = []
        val_NetworkISIs = []

        for unit, value in bursting_data_by_unit.items():
            try:
                val_NetworkISI = bursting_data_by_unit[unit]['mean_isi_all']
                val_NetworkISIs.append(val_NetworkISI)

                if simulated:
                    if unit in E_Gids:
                        fitness = the_scoring_function(val_NetworkISI, E_NetworkISI_target, weight, maxFitness, 
                                                       min_val=min_NetworkISI, max_val=max_NetworkISI)
                    elif unit in I_Gids:
                        fitness = the_scoring_function(val_NetworkISI, I_NetworkISI_target, weight, maxFitness, 
                                                       min_val=min_NetworkISI, max_val=max_NetworkISI)
                else:
                    fitness = the_scoring_function(val_NetworkISI, NetworkISI_target['target'], weight, maxFitness, 
                                                   min_val=min_NetworkISI, max_val=max_NetworkISI)

                fitness_NetworkISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_NetworkISI = {
            'fit': np.mean(fitness_NetworkISIs) if fitness_NetworkISIs else None,
            'value(s)': val_NetworkISIs,
            'target_E': E_NetworkISI_target,
            'target_I': I_NetworkISI_target,
            'min': min_NetworkISI,
            'max': max_NetworkISI,
            'weight': weight,
        }

        return fitness_NetworkISI

    except Exception as e:
        print(f"Error in fit_NetworkISI: {e}")
        return 1000

def fit_CovNetworkISI(simulated=False, **kwargs):
    try:
        print('Calculating CovNetworkISI fitness...')

        CovNetworkISI_target = kwargs['targets']['bursting_data']['bursting_summary_data']['CoVNetworkISI']
        E_I_ratio_network_CoV_ISI = 2.0 / 1.0

        E_CovNetworkISI_target = CovNetworkISI_target['target'] * (E_I_ratio_network_CoV_ISI / (E_I_ratio_network_CoV_ISI + 1))
        I_CovNetworkISI_target = CovNetworkISI_target['target'] / (E_I_ratio_network_CoV_ISI + 1)

        min_CovNetworkISI = CovNetworkISI_target['min']
        max_CovNetworkISI = CovNetworkISI_target['max']
        weight = CovNetworkISI_target['weight']
        maxFitness = kwargs['maxFitness']

        bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_CovNetworkISIs = []
        val_CovNetworkISIs = []

        for unit, value in bursting_data_by_unit.items():
            try:
                val_CovNetworkISI = bursting_data_by_unit[unit]['cov_isi_all']
                val_CovNetworkISIs.append(val_CovNetworkISI)

                if simulated:
                    if unit in E_Gids:
                        fitness = the_scoring_function(val_CovNetworkISI, E_CovNetworkISI_target, weight, maxFitness, 
                                                       min_val=min_CovNetworkISI, max_val=max_CovNetworkISI)
                    elif unit in I_Gids:
                        fitness = the_scoring_function(val_CovNetworkISI, I_CovNetworkISI_target, weight, maxFitness, 
                                                       min_val=min_CovNetworkISI, max_val=max_CovNetworkISI)
                else:
                    fitness = the_scoring_function(val_CovNetworkISI, CovNetworkISI_target['target'], weight, maxFitness, 
                                                   min_val=min_CovNetworkISI, max_val=max_CovNetworkISI)

                fitness_CovNetworkISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_CovNetworkISI = {
            'fit': np.mean(fitness_CovNetworkISIs) if fitness_CovNetworkISIs else None,
            'value(s)': val_CovNetworkISIs,
            'target_E': E_CovNetworkISI_target,
            'target_I': I_CovNetworkISI_target,
            'min': min_CovNetworkISI,
            'max': max_CovNetworkISI,
            'weight': weight,
        }

        return fitness_CovNetworkISI

    except Exception as e:
        print(f"Error in fit_CovNetworkISI: {e}")
        return 1000

def fit_Number_Bursts(mega_mode = False, **kwargs):
    try:
        print('Calculating NumBursts fitness...')

        if not mega_mode: NumBursts_target = kwargs['targets']['bursting_data']['bursting_summary_data']['Number_Bursts']
        else: NumBursts_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['Number_Bursts']
        target = NumBursts_target['target']
        min_NumBursts = NumBursts_target['min']
        max_NumBursts = NumBursts_target['max']
        weight = NumBursts_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: Number_Bursts = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['Number_Bursts']
        else: Number_Bursts = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['Number_Bursts']
        
        val_NumBursts = Number_Bursts

        fitness_NumBurst = the_scoring_function(val_NumBursts, target, weight, maxFitness, 
                                                min_val=min_NumBursts, max_val=max_NumBursts)

        fitness_NumBurst_dict = {
            'fit': fitness_NumBurst,
            'value': val_NumBursts,
            'target': target,
            'min': min_NumBursts,
            'max': max_NumBursts,
            'weight': weight,
        }

        return fitness_NumBurst_dict

    except Exception as e:
        print(f"Error in fit_Number_Bursts: {e}")
        return 1000

def fit_mean_IBI(mega_mode = False, **kwargs):
    try:
        print('Calculating mean_IBI fitness...')

        if not mega_mode: mean_IBI_target = kwargs['targets']['bursting_data']['bursting_summary_data']['mean_IBI']
        else: mean_IBI_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['mean_IBI']
        target = mean_IBI_target['target']
        min_mean_IBI = mean_IBI_target['min']
        max_mean_IBI = mean_IBI_target['max']
        weight = mean_IBI_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: mean_IBI = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['mean_IBI']
        else: mean_IBI = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['mean_IBI']
        val_mean_IBI = mean_IBI

        fitness_mean_IBI = the_scoring_function(val_mean_IBI, target, weight, maxFitness, 
                                                min_val=min_mean_IBI, max_val=max_mean_IBI)

        fitness_mean_IBI_dict = {
            'fit': fitness_mean_IBI,
            'value': val_mean_IBI,
            'target': target,
            'min': min_mean_IBI,
            'max': max_mean_IBI,
            'weight': weight,
        }

        return fitness_mean_IBI_dict

    except Exception as e:
        print(f"Error in fit_mean_IBI: {e}")
        return 1000

def fit_cov_IBI(mega_mode=False, **kwargs):
    try:
        print('Calculating cov_IBI fitness...')

        if not mega_mode: cov_IBI_target = kwargs['targets']['bursting_data']['bursting_summary_data']['cov_IBI']
        else: cov_IBI_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['cov_IBI']
        target = cov_IBI_target['target']
        min_cov_IBI = cov_IBI_target['min']
        max_cov_IBI = cov_IBI_target['max']
        weight = cov_IBI_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: cov_IBI = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['cov_IBI']
        else: cov_IBI = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['cov_IBI']
        val_cov_IBI = cov_IBI

        fitness_cov_IBI = the_scoring_function(val_cov_IBI, target, weight, maxFitness, 
                                               min_val=min_cov_IBI, max_val=max_cov_IBI)

        fitness_cov_IBI_dict = {
            'fit': fitness_cov_IBI,
            'value': val_cov_IBI,
            'target': target,
            'min': min_cov_IBI,
            'max': max_cov_IBI,
            'weight': weight,
        }

        return fitness_cov_IBI_dict

    except Exception as e:
        print(f"Error in fit_cov_IBI: {e}")
        return 1000

def fit_mean_Burst_Peak(mega_mode=False, **kwargs):
    try:
        print('Calculating mean_Burst_Peak fitness...')

        if not mega_mode: mean_Burst_Peak_target = kwargs['targets']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        else: mean_Burst_Peak_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        target = mean_Burst_Peak_target['target']
        # min_mean_Burst_Peak = mean_Burst_Peak_target['min']
        # max_mean_Burst_Peak = mean_Burst_Peak_target['max']
        min_mean_Burst_Peak = mean_Burst_Peak_target.get('min', None)
        max_mean_Burst_Peak = mean_Burst_Peak_target.get('max', None)
        weight = mean_Burst_Peak_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: mean_Burst_Peak = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        else: mean_Burst_Peak = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        val_mean_Burst_Peak = mean_Burst_Peak

        fitness_mean_Burst_Peak = the_scoring_function(val_mean_Burst_Peak, target, weight, maxFitness, 
                                                       min_val=min_mean_Burst_Peak, max_val=max_mean_Burst_Peak)

        fitness_mean_Burst_Peak_dict = {
            'fit': fitness_mean_Burst_Peak,
            'value': val_mean_Burst_Peak,
            'target': target,
            'min': min_mean_Burst_Peak,
            'max': max_mean_Burst_Peak,
            'weight': weight,
        }

        return fitness_mean_Burst_Peak_dict

    except Exception as e:
        print(f"Error in fit_mean_Burst_Peak: {e}")
        return 1000

def fit_cov_Burst_Peak(mega_mode = False, **kwargs):
    try:
        print('Calculating cov_Burst_Peak fitness...')

        if not mega_mode: cov_Burst_Peak_target = kwargs['targets']['bursting_data']['bursting_summary_data']['cov_Burst_Peak']
        else: cov_Burst_Peak_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['cov_Burst_Peak']
        target = cov_Burst_Peak_target['target']
        min_cov_Burst_Peak = cov_Burst_Peak_target['min']
        max_cov_Burst_Peak = cov_Burst_Peak_target['max']
        weight = cov_Burst_Peak_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: cov_Burst_Peak = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['cov_Burst_Peak']
        else: cov_Burst_Peak = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['cov_Burst_Peak']
        val_cov_Burst_Peak = cov_Burst_Peak

        fitness_cov_Burst_Peak = the_scoring_function(val_cov_Burst_Peak, target, weight, maxFitness, 
                                                      min_val=min_cov_Burst_Peak, max_val=max_cov_Burst_Peak)

        fitness_cov_Burst_Peak_dict = {
            'fit': fitness_cov_Burst_Peak,
            'value': val_cov_Burst_Peak,
            'target': target,
            'min': min_cov_Burst_Peak,
            'max': max_cov_Burst_Peak,
            'weight': weight,
        }

        return fitness_cov_Burst_Peak_dict

    except Exception as e:
        print(f"Error in fit_cov_Burst_Peak: {e}")
        return 1000

def fit_fano_factor(mega_mode=False, **kwargs):
    try:
        print('Calculating fano_factor fitness...')

        if not mega_mode: fano_factor_target = kwargs['targets']['bursting_data']['bursting_summary_data']['fano_factor']
        else: fano_factor_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['fano_factor']
        target = fano_factor_target['target']
        min_fano_factor = fano_factor_target['min']
        max_fano_factor = fano_factor_target['max']
        weight = fano_factor_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: fano_factor = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['fano_factor']
        else: fano_factor = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['fano_factor']
        val_fano_factor = fano_factor

        fitness_fano_factor = the_scoring_function(val_fano_factor, target, weight, maxFitness, 
                                                   min_val=min_fano_factor, max_val=max_fano_factor)

        fitness_fano_factor_dict = {
            'fit': fitness_fano_factor,
            'value': val_fano_factor,
            'target': target,
            'min': min_fano_factor,
            'max': max_fano_factor,
            'weight': weight,
        }

        return fitness_fano_factor_dict

    except Exception as e:
        print(f"Error in fit_fano_factor: {e}")
        return 1000

'''dealbreakers'''
def I_frs_must_be_greater_than_E_frs(fitnessVals, network_metrics):
    '''I firing rates must be greater than E firing rates'''
    dealbroken = False
    try:
        simulated_data = network_metrics['simulated_data']
        spiking_data_by_unit = network_metrics['simulated_data']['spiking_data_by_unit']
        
        # Get population data for each population
        E_rates = []
        I_rates = []
        for unit, value in spiking_data_by_unit.items():
            if unit in simulated_data['E_Gids']:
                firing_rate = spiking_data_by_unit[unit]['FireRate']
                E_rates.append(firing_rate)
            elif unit in simulated_data['I_Gids']:
                firing_rate = spiking_data_by_unit[unit]['FireRate']
                I_rates.append(firing_rate)
                
        # Check if I firing rates are greater than E firing rates
        if np.mean(I_rates) < np.mean(E_rates):
            dealbroken = True
            print(f'I firing rates are not greater than E firing rates. I: {np.mean(I_rates)}, E: {np.mean(E_rates)}')
            
        return dealbroken
    except Exception as e:
        print(f"Error in I_frs_must_be_greater_than_E_frs: {e}")
        return False

def E_and_I_neurons_must_fire(fitnessVals, network_metrics):
    '''E and I neurons must fire'''
    dealbroken = False
    try:
        simulated_data = network_metrics['simulated_data']
        spiking_data_by_unit = network_metrics['simulated_data']['spiking_data_by_unit']
        
        #get population data for each population
        E_rates = []
        I_rates = []
        for unit, value in spiking_data_by_unit.items():
            if unit in simulated_data['E_Gids']:
                firing_rate = spiking_data_by_unit[unit]['FireRate']
                E_rates.append(firing_rate)
            elif unit in simulated_data['I_Gids']:
                firing_rate = spiking_data_by_unit[unit]['FireRate']
                I_rates.append(firing_rate)
                
        #check if E and I neurons are firing
        if np.mean(E_rates) == 0 or np.mean(I_rates) == 0:
            dealbroken = True
            print(f'E and I neurons are not firing. E: {np.mean(E_rates)}, I: {np.mean(I_rates)}')
    except Exception as e:
        print(f"Error in E_and_I_neurons_must_fire: {e}")
        dealbroken = False
        
    return dealbroken

def check_dealbreakers(fitnessVals, **kwargs):
    '''Check if any fitness values are dealbreakers'''
    network_metrics = kwargs['network_metrics']
    dealbreakers = []
    # dealbroken = I_frs_must_be_greater_than_E_frs(fitnessVals, network_metrics)
    # dealbroken = E_and_I_neurons_must_fire(fitnessVals, network_metrics)  
    dealbreakers.append(I_frs_must_be_greater_than_E_frs(fitnessVals, network_metrics))
    dealbreakers.append(E_and_I_neurons_must_fire(fitnessVals, network_metrics))
    if any(dealbreakers):
        return True
    else:
        return False  
    #return dealbroken

'''helper functions'''
def save_fitness_results(output_path, fitnessResults):
    with open(output_path, 'w') as f:
        json.dump(fitnessResults, f, indent=4)
    print(f'fitness results saved to {output_path}')

def get_fitness(data_source, **kwargs):
    '''Main fitness calculation function'''
    try:
        fitnessVals = {}
        
        # Spiking Data
        fitnessVals['rate_fit'] = fit_firing_rates(data_source, **kwargs)
        fitnessVals['CoV_rate_fit'] = fit_CoV_firing_rate(data_source, **kwargs)
        # fitnessVals['ISI_fit'] = fit_ISI(simulated=True, **kwargs)
        # fitnessVals['CoV_ISI_fit'] = fit_CoV_ISI(simulated=True, **kwargs)
        fitnessVals['ISI_fit'] = fit_ISI(data_source, **kwargs)
        fitnessVals['CoV_ISI_fit'] = fit_CoV_ISI(data_source, **kwargs)
        
        
        # aw 2025-02-03 08:08:13 - left off here. NEed to update these bursting functions
        # also need to add the mega bursting functions
        # Bursting Data
        #fitnessVals['mean_Burst_Rate_fit'] = fit_mean_Burst_Rate(simulated=True, **kwargs)
        fitnessVals['mean_Burst_Rate_fit'] = fit_mean_Burst_Rate(data_source, **kwargs)
        fitnessVals['baseline_fit'] = fit_baseline(data_source, **kwargs)
        fitnessVals['WithinBurstISI_fit'] = fit_WithinBurstISI(data_source, **kwargs)
        fitnessVals['CoVWithinBurstISI_fit'] = fit_CovWithinBurstISI(data_source, **kwargs)
        fitnessVals['OutsideBurstISI_fit'] = fit_OutsideBurstISI(data_source, **kwargs)
        fitnessVals['CoVOutsideBurstISI_fit'] = fit_CovOutsideBurstISI(data_source, **kwargs)
        # fitnessVals['NetworkISI_fit'] = fit_NetworkISI(data_source, **kwargs)
        # fitnessVals['CoVNetworkISI_fit'] = fit_CovNetworkISI(simulated=True, **kwargs)
        # fitnessVals['Number_Bursts_fit'] = fit_Number_Bursts(**kwargs)
        fitnessVals['mean_IBI_fit'] = fit_mean_IBI(**kwargs)
        fitnessVals['cov_IBI_fit'] = fit_cov_IBI(**kwargs)
        fitnessVals['mean_Burst_Peak_fit'] = fit_mean_Burst_Peak(**kwargs)
        fitnessVals['cov_Burst_Peak_fit'] = fit_cov_Burst_Peak(**kwargs)
        fitnessVals['fano_factor_fit'] = fit_fano_factor(**kwargs)
        
        # Mega Bursting Data
        fitnessVals['mega_mean_Burst_Rate_fit'] = fit_mean_Burst_Rate(data_source, mega_mode=True, **kwargs)
        fitnessVals['mega_baseline_fit'] = fit_baseline(data_source, mega_mode=True, **kwargs)
        fitnessVals['mega_WithinBurstISI_fit'] = fit_WithinBurstISI(data_source, mega_mode=True, **kwargs)
        fitnessVals['mega_CoVWithinBurstISI_fit'] = fit_CovWithinBurstISI(data_source, mega_mode=True, **kwargs)
        fitnessVals['mega_OutsideBurstISI_fit'] = fit_OutsideBurstISI(data_source, mega_mode=True, **kwargs)
        fitnessVals['mega_CoVOutsideBurstISI_fit'] = fit_CovOutsideBurstISI(data_source, mega_mode=True, **kwargs)
        # fitnessVals['mega_NetworkISI_fit'] = fit_NetworkISI(data_source, mega_mode=True, **kwargs)
        # fitnessVals['mega_CoVNetworkISI_fit'] = fit_CovNetworkISI(simulated=True, mega_mode=True, **kwargs)
        # fitnessVals['mega_Number_Bursts_fit'] = fit_Number_Bursts(mega_mode=True, **kwargs)
        fitnessVals['mega_mean_IBI_fit'] = fit_mean_IBI(mega_mode=True, **kwargs)
        fitnessVals['mega_cov_IBI_fit'] = fit_cov_IBI(mega_mode=True, **kwargs)
        fitnessVals['mega_mean_Burst_Peak_fit'] = fit_mean_Burst_Peak(mega_mode=True, **kwargs)
        fitnessVals['mega_cov_Burst_Peak_fit'] = fit_cov_Burst_Peak(mega_mode=True, **kwargs)
        #fitnessVals['mega_fano_factor_fit'] = fit_fano_factor(mega_mode=True, **kwargs)
        
        # for debugging, uncomment as needed:
        # # print warning if mega and non mega versions of the same metric have the same  'fit' and 'value'
        
        # for key in fitnessVals:
        #     if 'mega' in key:
        #         non_mega_key = key.replace('mega_', '')
        #         try:
        #             if fitnessVals[key]['fit'] == fitnessVals[non_mega_key]['fit'] and fitnessVals[key]['fit'] != 1:
        #                 print(f'Warning: {key} and {non_mega_key} have the same fit. Consider removing one of them.')
                        
        #                 value_key = fitnessVals[key].get('value', fitnessVals[key].get('value(s)'))
        #                 value_non_mega_key = fitnessVals[non_mega_key].get('value', fitnessVals[non_mega_key].get('value(s)'))
                        
        #                 if isinstance(value_key, list) and isinstance(value_non_mega_key, list):
        #                     if all(v1 == v2 for v1, v2 in zip(value_key, value_non_mega_key)):
        #                         print(f'Warning: {key} and {non_mega_key} have the same fit and value. Consider removing one of them.')
        #                 elif value_key == value_non_mega_key:
        #                     print(f'Warning: {key} and {non_mega_key} have the same fit and value. Consider removing one of them.')
        #                 # aw 2025-02-04 11:46:05 - fanofactor between mega and non mega bursts is the same.
        #                 # this is because the fano factor is calculated using the same data for both mega and non mega bursts
        #                 # it may be more appropriate to calculate the fano factor using individual spiking data for each neuron
        #         except Exception as e:
        #             print(f'Error comparing {key} and {non_mega_key}: {e}')
        #             continue

        # if any fit values are equal to 1000, change to {'fit': 1000}
        for key in fitnessVals:
            if fitnessVals[key] == 1000:
                fitnessVals[key] = {'fit': 1000}
        average_fitness = np.mean([fitnessVals[key]['fit'] for key in fitnessVals])
        
        # assert that average_fitness is an integer or float
        assert isinstance(average_fitness, (int, float)), 'average_fitness is not an integer or float'
        
        return average_fitness, fitnessVals
    except Exception as e:
        print(f"Error in get_fitness: {e}")
        return 1000, {}

def handle_existing_fitness(fitness_save_path):
    if os.path.exists(fitness_save_path):
        with open(fitness_save_path, 'r') as f:
            fitnessResults = json.load(f)
        average_fitness = fitnessResults['average_fitness']
        print(f'Fitness results already exist: {average_fitness}')
        return average_fitness
    return None

def handle_simulated_data_mode(simData, kwargs):
    """Handles the logic for 'simulated data' mode."""
    kwargs['source'] = 'simulated'
    candidate_path = kwargs['simConfig']['filename']
    fitness_save_path = f'{candidate_path}_fitness.json'
    assert simData is not None, 'Simulated data must be provided in "simulated data" mode.'
    kwargs.update({'simData': simData})
    error, kwargs = calculate_network_metrics(kwargs)
    return error, kwargs

def handle_experimental_data_mode(kwargs):
    """Handles the logic for 'experimental data' mode."""
    kwargs['source'] = 'experimental'
    implemented = False
    assert implemented, 'Experimental data source not yet implemented.'
    return None, kwargs

def calculate_and_save_fitness(kwargs):
    """Calculates fitness and saves the results."""
    try:
        average_fitness, fitnessResults = get_fitness(kwargs['source'], **kwargs)
        fitnessResults['average_fitness'] = average_fitness
        fitnessResults['maxFitness'] = kwargs['maxFitness']

        # deal_broken = check_dealbreakers(fitnessResults, **kwargs)
        # assert not deal_broken, 'Dealbreakers found in fitness results.'

        save_fitness_results(kwargs['fitness_save_path'], fitnessResults)
        return average_fitness
    except Exception as e:
        error_trace = str(e)
        fitnessResults = {
            'average_fitness': kwargs['maxFitness'],
            'maxFitness': kwargs['maxFitness'],
            'error': 'acceptable' if any(error in error_trace for error in []) else 'new',
            'error_trace': error_trace
        }
        print(f'Error calculating fitness: {e}')
        save_fitness_results(kwargs['fitness_save_path'], fitnessResults)
        return 1000
    
'''main functions'''
def calculate_network_metrics(kwargs):
    print('Calculating network activity metrics...')
    from RBS_network_models.network_analysis import get_simulated_network_activity_metrics
    network_metrics = get_simulated_network_activity_metrics(**kwargs)
        
    # Save the network metrics to a file
    # if networks_metrics is None, save an error message
    if network_metrics is None:
        print('Network activity metrics could not be calculated.')
        fitnessResults = {
            'average_fitness': kwargs['maxFitness'],
            'maxFitness': kwargs['maxFitness'],
            'error': 'Network activity metrics could not be calculated. Fitness set to maxFitness.'
        }
        print(f'Fitness set to maxFitness: {kwargs["maxFitness"]}')
        save_fitness_results(kwargs['fitness_save_path'], fitnessResults)
        kwargs['network_metrics'] = None
        error = fitnessResults['error']
        return error, kwargs
    #else, return the network_metrics
    else:
        kwargs['network_metrics'] = network_metrics
        #kwargs['mega_network_metrics'] = mega_metrics
        return None, kwargs
    
'''parallelized fitness calculation'''
import subprocess
def submit_fitness_job(simData=None, mode='optimizing', **kwargs):
    ''' Copy the initial logic of calulate_fitness() and modify it to submit a job to the cluster '''
    
    try:
        # Select mode and execute corresponding logic
        if mode == 'optimizing':
            kwargs = retrieve_sim_data_from_call_stack(simData, **kwargs)
            kwargs.pop('simData', None) # Remove simData from kwargs, it will be added back in calculate_fitness()
            #error, kwargs = handle_optimizing_mode(simData, kwargs)
            
        # elif mode == 'simulated data':
        #     error, kwargs = handle_simulated_data_mode(simData, kwargs)
        # elif mode == 'experimental data':
        #     error, kwargs = handle_experimental_data_mode(kwargs)
        else:
            raise ValueError(f'Unknown mode: {mode}')

        # Save kwargs to a file
        sim_data_path = kwargs.get('data_file_path', None)
        assert sim_data_path is not None, 'Data file path not provided.'
        kwargs_path = sim_data_path.replace('_data.json', '_kwargs.json')
        kwargs['kwargs_path'] = kwargs_path
        
    except Exception as e:
        print(f'Error calculating fitness: {e}')
        fitnessResults = {
            'average_fitness': kwargs.get('maxFitness', 1000),
            'maxFitness': kwargs.get('maxFitness', 1000),
            'error': 'general',
            'error_trace': str(e)
        }
        save_fitness_results(kwargs.get('fitness_save_path', kwargs.get('fitness_save_path', 'unknown_path.json')), fitnessResults)
        #save_fitness_results(kwargs.get('fitness_save_path', 'unknown_path.json'), fitnessResults)
        return 1000

    # Prepare sbatch or srun command
    fitness_script = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/batch_scripts/fitness_helper.py'
    import sys
    from pprint import pprint
    
    #global_kwargs_path = temp_user_args.global_kwargs_path
    #global_kwargs_path = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/temp_global_kwargs.pkl'
    
    #load pickle file from global_kwargs_path
    #import pickle
        
    # # Ensure the file exists
    # if os.path.exists(global_kwargs_path):
    #     try:
    #         print(f'Loading global kwargs from {global_kwargs_path}')
    #         with open(global_kwargs_path, 'rb') as f:
    #             global_kwargs = pickle.load(f)
    #         print("Global kwargs loaded successfully.")
    #         print(global_kwargs)  # Print the loaded data for debugging purposes
    #     except Exception as e:
    #         print(f"Error loading pickle file: {e}")
    # else:
    #    print(f"Pickle file does not exist at: {global_kwargs_path}")
    
    from __main__ import global_kwargs
    nodes_per_sim = global_kwargs.get('nodes_per_sim', 1)
    cores_per_sim = global_kwargs.get('cores_per_sim', 1)
    cores_per_task = global_kwargs.get('cores_per_task', 1)
    tasks_per_sim = global_kwargs.get('tasks_per_sim', 1)
    #available_cores = tasks_per_sim * cores_per_task
    
    nodes_per_sim = str(global_kwargs.get('nodes_per_sim', 1))  # Convert to string
    cores_per_task = str(global_kwargs.get('cores_per_task', 1))  # Convert to string
    tasks_per_sim = str(global_kwargs.get('tasks_per_sim', 1))  # Convert to string
    
    slurm_command = [
        "srun", "--exclusive", 
        "-N", nodes_per_sim,
        "-n", '1',
        #"--cpus-per-task", f'{cores_per_sim}',
        "--cpus-per-task", f'{cores_per_task}',
        "python3", fitness_script, sim_data_path, kwargs_path
    ]
    
    pprint(f"Submitting fitness job with command: {slurm_command}")

    # Submit the job
    subprocess.run(slurm_command)


    # def submit_fitness_job(kwargs):
    #     try:
    #         sim_data_path = kwargs.get('data_file_path', None)
    #         assert sim_data_path is not None, 'Data file path not provided.'
    #         kwargs_path = kwargs.get('kwargs_path', sim_data_path.replace('_data.json', '_kwargs.json'))

    #         # Prepare sbatch or srun command
    #         fitness_script = "/path/to/fitness_calculation.py"
    #         slurm_command = [
    #             "srun", "--exclusive", "--cpus-per-task=1",
    #             "python3", fitness_script, sim_data_path, kwargs_path
    #         ]

    #         # Submit the job
    #         subprocess.run(slurm_command)
    #     except Exception as e:
    #         print(f'Error submitting fitness job: {e}')
    #         return 1000 