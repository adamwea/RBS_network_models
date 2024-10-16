import os
import json
import numpy as np

# Import all the fitness functions
from simulation_fitness_functions.fit_baseline import fit_baseline
from simulation_fitness_functions.fit_burst_frequency import fit_burst_frequency
from simulation_fitness_functions.fit_threshold import fit_threshold
from simulation_fitness_functions.fit_firing_rates import fit_E_firing_rate, fit_I_firing_rate
from simulation_fitness_functions.fit_ISI import fit_E_ISI, fit_I_ISI
from simulation_fitness_functions.fit_burst_amplitude import fit_big_burst_amplitude, fit_small_burst_amplitude, fit_bimodal_burst_amplitude
from simulation_fitness_functions.fit_network_IBI import fit_IBI 
from simulation_fitness_functions.fit_sustain import fit_sustain
from simulation_fitness_functions.fit_rate_slope import fit_rate_slope
from simulation_analysis_functions.network_activity_metrics import get_simulated_network_activity_metrics
from simulation_analysis_functions.individual_neuron_metrics import get_individual_neuron_metrics

def fitnessFunc(simObj):
    
    def get_fitness():
        '''Main fitness calculation function'''
        fitnessVals = {}

        # Priority 1
        fitnessVals['E_rate_fit'] = fit_E_firing_rate(neuron_metrics, **kwargs)
        fitnessVals['I_rate_fit'] = fit_I_firing_rate(neuron_metrics, **kwargs)
        fitnessVals['E_ISI_fit'] = fit_E_ISI(neuron_metrics, **kwargs)
        fitnessVals['I_ISI_fit'] = fit_I_ISI(neuron_metrics, **kwargs)

        # Priority 2
        fitnessVals['baseline_fit'] = fit_baseline(net_activity_metrics, **kwargs)

        # Priority 3
        fitnessVals['IBI_fitness'] = fit_IBI(net_activity_metrics, **kwargs)
        fitnessVals['burst_frequency_fitness'] = fit_burst_frequency(net_activity_metrics, **kwargs)

        # Priority 4
        fitnessVals['big_burst_fit'] = fit_big_burst_amplitude(net_activity_metrics, **kwargs)
        fitnessVals['small_burst_fit'] = fit_small_burst_amplitude(net_activity_metrics, **kwargs)
        fitnessVals['thresh_fit'] = fit_threshold(net_activity_metrics, **kwargs)

        # Priority 5
        fitnessVals['bimodal_burst_fit'] = fit_bimodal_burst_amplitude(net_activity_metrics, **kwargs)
        fitnessVals['slope_fit'] = fit_rate_slope(net_activity_metrics, **kwargs)
        fitnessVals['sustain_fit'] = fit_sustain(net_activity_metrics, **kwargs)

        # Calculate average fitness and prioritize
        prioritize_fitness(fitnessVals, **kwargs)
        average_fitness, avg_scaled_fitness = fitness_summary_metrics(fitnessVals)

        # Save fitness results
        save_fitness_results(fitnessVals, average_fitness, avg_scaled_fitness)
        
        return average_fitness, avg_scaled_fitness, fitnessVals

    def prioritize_fitness(fitnessVals, **kwargs):
        '''Assign priorities and handle fitness values with maxFitness.'''
        print('Prioritizing fitness values...')
        maxFitness = kwargs['maxFitness']
        priorities = [
            ['E_rate_fit', 'I_rate_fit', 'E_ISI_fit', 'I_ISI_fit'],  # Priority 1
            ['baseline_fit'],  # Priority 2
            ['IBI_fitness', 'burst_frequency_fitness', 'big_burst_fit', 'small_burst_fit', 'thresh_fit', 'bimodal_burst_fit', 'slope_fit']  # Priority 3
        ]
        for priority in priorities:
            if any(fitnessVals[fit]['Fit'] == maxFitness for fit in priority):
                for lower_priority in priorities[priorities.index(priority) + 1:]:
                    for fit in lower_priority:
                        fitnessVals[fit]['Fit'] = maxFitness
                        fitnessVals[fit]['deprioritized'] = True
                break

    def fitness_summary_metrics(fitnessVals):
        '''Calculate and summarize the fitness metrics.'''
        fitness_values = {key: fitnessVals[key]['Fit'] for key in fitnessVals if 'Fit' in fitnessVals[key]}
        fitness_values = [v for v in fitness_values.values() if v is not None]
        average_fitness = np.mean(fitness_values)

        min_value, max_value = min(fitness_values), max(fitness_values)
        if max_value > min_value:
            normalized_fitness_values = [(v - min_value) / (max_value - min_value) for v in fitness_values]
        else:
            normalized_fitness_values = [1 for _ in fitness_values]

        avg_scaled_fitness = np.mean(normalized_fitness_values)
        print(f'Average Fitness: {average_fitness}, Average Scaled Fitness: {avg_scaled_fitness}')
        return average_fitness, avg_scaled_fitness

    def save_fitness_results(fitnessVals, average_fitness, avg_scaled_fitness):
        '''Save fitness results to a file.'''
        fitnessResults = {key: value for key, value in fitnessVals.items()}
        fitnessResults['average_fitness'] = average_fitness
        fitnessResults['average_scaled_fitness'] = avg_scaled_fitness
        fitnessResults['maxFitness'] = kwargs['maxFitness']

        output_path = batch_saveFolder or fitness_save_path
        if exp_mode:
            destination = os.path.join(output_path, f'{simLabel}_Fitness.json')
        else:
            gen_folder = simLabel.split('_cand')[0]
            destination = os.path.join(output_path, gen_folder, f'{simLabel}_Fitness.json')

        with open(destination, 'w') as f:
            json.dump(fitnessResults, f, indent=4)
        print(f'Fitness results saved to {destination}')

    # Main logic of the calculate_fitness function
    print('Calculating network activity metrics...')
    net_activity_metrics = get_simulated_network_activity_metrics(simData=simObj.simData)
    print('Calculating individual neuron metrics...')
    neuron_metrics = get_individual_neuron_metrics(data_file_path, exp_mode=exp_mode)

    # Get the fitness
    average_fitness, avg_scaled_fitness, fitnessVals = get_fitness()

    return average_fitness
