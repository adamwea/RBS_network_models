import numpy as np
from modules.helper_functions.load_clean_sim_object import load_clean_sim_object
import netpyne

def get_individual_neuron_metrics(data_file_path, exp_mode=False):
    def get_spike_data_from_simulated_data():
        load_clean_sim_object(data_file_path)
        spike_data = netpyne.sim.analysis.prepareSpikeData()
        E_neurons = [ind for ind in spike_data['cellGids'] if spike_data['cellPops'][ind] == 'E']
        I_neurons = [ind for ind in spike_data['cellGids'] if spike_data['cellPops'][ind] == 'I']
        return spike_data, E_neurons, I_neurons

    def get_spike_data_from_experimental_data():
        real_spike_data = np.load(data_file_path, allow_pickle=True)
        data = real_spike_data['spike_array']
        firing_rates = {}
        total_time = len(data[0]) / 10000
        for i in range(len(data)):
            firing_rates[i] = sum(data[i]) / total_time

        I_neurons = np.array([i for i in firing_rates if firing_rates[i] > np.percentile(list(firing_rates.values()), 70)])
        E_neurons = np.array([i for i in firing_rates if firing_rates[i] <= np.percentile(list(firing_rates.values()), 70)])

        return firing_rates, E_neurons, I_neurons

    try:
        if not exp_mode:
            spike_data, E_neurons, I_neurons = get_spike_data_from_simulated_data()
        else:
            firing_rates, E_neurons, I_neurons = get_spike_data_from_experimental_data()

        E_average_ISIs, I_average_ISIs = {}, {}
        E_average_firing_rates, I_average_firing_rates = {}, {}

        neuron_indices = np.unique(spike_data['spkInds'])
        cellPops = spike_data['cellPops']

        for neuron_index in neuron_indices:
            spike_times = np.array(spike_data['spkTimes']) / 1000
            spike_times = spike_times[spike_data['spkInds'] == neuron_index]
            firing_rate = len(spike_times) / (spike_times[-1] - spike_times[0]) if len(spike_times) > 1 else None
            cellType = cellPops[neuron_index]

            if 'E' in cellType:
                E_average_firing_rates[neuron_index] = firing_rate
            elif 'I' in cellType:
                I_average_firing_rates[neuron_index] = firing_rate

            ISIs = np.diff(spike_times) if len(spike_times) > 1 else None
            average_ISI = np.mean(ISIs) if ISIs is not None else None
            if 'E' in cellType:
                E_average_ISIs[neuron_index] = average_ISI
            elif 'I' in cellType:
                I_average_ISIs[neuron_index] = average_ISI

        neuron_metrics = {
            'E_average_firing_rates': E_average_firing_rates,
            'I_average_firing_rates': I_average_firing_rates,
            'E_average_ISIs': E_average_ISIs,
            'I_average_ISIs': I_average_ISIs,
            'E_neurons': E_neurons,
            'I_neurons': I_neurons,
        }

        return neuron_metrics

    except Exception as e:
        print(f'Error calculating individual firing rates and average ISIs: {e}')
        return None