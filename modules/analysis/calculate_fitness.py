import os
import json
from modules.analysis.analyze_network_activity import get_simulated_network_activity_metrics
import numpy as np
#from simulate._config_files.fitnessFuncArgs import fitnessFuncArgs
from modules.analysis.extract_simulated_data import retrieve_sim_data_from_call_stack

'''Fitness functions for the network activity metrics'''
def fit_firing_rates(simulated=False, **kwargs):
    print('Calculating firing rate fitness...')
    MeanFireRate_target = kwargs['targets']['spiking_data']['spiking_summary_data']['MeanFireRate']
    E_I_ratio = 5  # 1:5 ratio of E to I neurons
    E_fr_target = MeanFireRate_target['target'] * (E_I_ratio / (E_I_ratio + 1))
    I_fr_target = MeanFireRate_target['target'] / (E_I_ratio + 1)
    min_FR = MeanFireRate_target['min']
    max_FR = MeanFireRate_target['max']
    weight = MeanFireRate_target['weight']
    maxFitness = kwargs['maxFitness']
    
    #MeanFireRate = kwargs['net_activity_metrics']['spiking_data']['spiking_summary_data']['MeanFireRate']
    #val_FR = MeanFireRate
    spiking_data_by_unit = kwargs['network_metrics']['spiking_data']['spiking_data_by_unit']
    fitness_FRs = []
    for unit, value in spiking_data_by_unit.items():
        val_FR = spiking_data_by_unit[unit]['FireRate']
        if simulated is False: 
            target = MeanFireRate_target['target'] #experimental target
            fitness = the_scoring_function(val_FR, target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
        else:
            E_gids = kwargs['network_metrics']['simulated_data']['E_Gids']
            I_gids = kwargs['network_metrics']['simulated_data']['I_Gids']
            if unit in E_gids:
                E_fitness = the_scoring_function(val_FR, E_fr_target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
                fitness = E_fitness
            elif unit in I_gids:
                I_fitness = the_scoring_function(val_FR, I_fr_target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
                fitness = I_fitness
            #fitness = np.mean([E_fitness, I_fitness])
        fitness_FRs.append(fitness)
    fitness_FR = np.mean(fitness_FRs)
    #print(f'Firing rate fitness: {fitness_FRs}')
    return fitness_FR

def fit_CoV_firing_rate(**kwargs):
    print('Calculating CoV firing rate fitness...')
    CoVFireRate_target = kwargs['targets']['spiking_data']['spiking_summary_data']['CoVFireRate']
    target = CoVFireRate_target['target']
    min_CoV = CoVFireRate_target['min']
    max_CoV = CoVFireRate_target['max']
    weight = CoVFireRate_target['weight']
    maxFitness = kwargs['maxFitness']
    
    CoVFireRate = kwargs['targets']['spiking_data']['spiking_summary_data']['CoVFireRate']
    val_CoV = CoVFireRate
    fitness_CoV = the_scoring_function(val_CoV, target, weight, maxFitness, min_val=min_CoV, max_val=max_CoV)
    print(f'CoV firing rate fitness: {fitness_CoV}')
    return fitness_CoV
    
def fit_ISI(**kwargs):
    print('Calculating ISI fitness...')
    MeanISI_target = kwargs['net_activity_targets']['spiking_data']['spiking_summary_data']['MeanISI']
    target = MeanISI_target['target']
    min_ISI = MeanISI_target['min']
    max_ISI = MeanISI_target['max']
    weight = MeanISI_target['weight']
    maxFitness = kwargs['maxFitness']
    
    spiking_data_by_unit = kwargs['net_activity_metrics']['spiking_data']['spiking_data_by_unit']
    fitness_ISIs = []
    for unit, value in spiking_data_by_unit.items():
        val_ISI = spiking_data_by_unit[unit]['meanISI']
        fitness = the_scoring_function(val_ISI, target, weight, maxFitness, min_val=min_ISI, max_val=max_ISI)
        fitness_ISIs.append(fitness)
    fitness_ISI = np.mean(fitness_ISIs)
    print(f'ISI fitness: {fitness_ISI}')
    return fitness_ISI

def fit_CoV_ISI(**kwargs):
    print('Calculating CoV ISI fitness...')
    CoV_ISI_target = kwargs['net_activity_targets']['spiking_data']['spiking_summary_data']['CoV_ISI']
    target = CoV_ISI_target['target']
    min_CoV = CoV_ISI_target['min']
    max_CoV = CoV_ISI_target['max']
    weight = CoV_ISI_target['weight']
    maxFitness = kwargs['maxFitness']
    
    spiking_data_by_unit = kwargs['net_activity_metrics']['spiking_data']['spiking_data_by_unit']
    fitness_CoV_ISIs = []
    for unit, value in spiking_data_by_unit.items():
        val_CoV_ISI = spiking_data_by_unit[unit]['CoV_ISI']
        fitness = the_scoring_function(val_CoV_ISI, target, weight, maxFitness, min_val=min_CoV, max_val=max_CoV)
        fitness_CoV_ISIs.append(fitness)
    fitness_CoV_ISI = np.mean(fitness_CoV_ISIs)
    print(f'CoV ISI fitness: {fitness_CoV_ISI}')
    return fitness_CoV_ISI

def fit_baseline(**kwargs):
    print('Calculating baseline fitness...')
    baseline_target = kwargs['net_activity_targets']['bursting_data']['bursting_summary_data']['baseline']
    target = baseline_target['target']
    min_baseline = baseline_target['min']
    max_baseline = baseline_target['max']
    weight = baseline_target['weight']
    maxFitness = kwargs['maxFitness']
    
    baseline = kwargs['net_activity_metrics']['bursting_data']['bursting_summary_data']['baseline']
    val_baseline = baseline
    fitness_baseline = the_scoring_function(val_baseline, target, weight, maxFitness, min_val=min_baseline, max_val=max_baseline)
    print(f'Baseline fitness: {fitness_baseline}')
    return fitness_baseline

def fit_WithinBurstISI(**kwargs):
    print('Calculating WithinBurstISI fitness...')
    WithinBurstISI_target = kwargs['net_activity_targets']['bursting_data']['bursting_summary_data']['MeanWithinBurstISI']
    target = WithinBurstISI_target['target']
    min_WithinBurstISI = WithinBurstISI_target['min']
    max_WithinBurstISI = WithinBurstISI_target['max']
    weight = WithinBurstISI_target['weight']
    maxFitness = kwargs['maxFitness']
    
    bursting_data_by_unit = kwargs['net_activity_metrics']['bursting_data']['bursting_data_by_unit']
    fitness_WithinBurstISIs = []
    for unit, value in bursting_data_by_unit.items():
        val_WithinBurstISI = bursting_data_by_unit[unit]['mean_isi_within']
        fitness = the_scoring_function(val_WithinBurstISI, target, weight, maxFitness, min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
        fitness_WithinBurstISIs.append(fitness)
    fitness_WithinBurstISI = np.mean(fitness_WithinBurstISIs)
    print(f'WithinBurstISI fitness: {fitness_WithinBurstISI}')
    return fitness_WithinBurstISI

def fit_CovWithinBurstISI(**kwargs):
    print('Calculating CovWithinBurstISI fitness...')
    CovWithinBurstISI_target = kwargs['net_activity_targets']['bursting_data']['bursting_summary_data']['CovWithinBurstISI']
    target = CovWithinBurstISI_target['target']
    min_CovWithinBurstISI = CovWithinBurstISI_target['min']
    max_CovWithinBurstISI = CovWithinBurstISI_target['max']
    weight = CovWithinBurstISI_target['weight']
    maxFitness = kwargs['maxFitness']
    
    bursting_data_by_unit = kwargs['net_activity_metrics']['bursting_data']['bursting_data_by_unit']
    fitness_CovWithinBurstISIs = []
    for unit, value in bursting_data_by_unit.items():
        val_CovWithinBurstISI = bursting_data_by_unit[unit]['cov_isi_within']
        fitness = the_scoring_function(val_CovWithinBurstISI, target, weight, maxFitness, min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
        fitness_CovWithinBurstISIs.append(fitness)
    fitness_CovWithinBurstISI = np.mean(fitness_CovWithinBurstISIs)
    print(f'CovWithinBurstISI fitness: {fitness_CovWithinBurstISI}')
    return fitness_CovWithinBurstISI

def fit_OutsideBurstISI(**kwargs):
    print('Calculating OutsideBurstISI fitness...')
    OutsideBurstISI_target = kwargs['net_activity_targets']['bursting_data']['bursting_summary_data']['MeanOutsideBurstISI']
    target = OutsideBurstISI_target['target']
    min_OutsideBurstISI = OutsideBurstISI_target['min']
    max_OutsideBurstISI = OutsideBurstISI_target['max']
    weight = OutsideBurstISI_target['weight']
    maxFitness = kwargs['maxFitness']
    
    bursting_data_by_unit = kwargs['net_activity_metrics']['bursting_data']['bursting_data_by_unit']
    fitness_OutsideBurstISIs = []
    for unit, value in bursting_data_by_unit.items():
        val_OutsideBurstISI = bursting_data_by_unit[unit]['mean_isi_outside']
        fitness = the_scoring_function(val_OutsideBurstISI, target, weight, maxFitness, min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
        fitness_OutsideBurstISIs.append(fitness)
    fitness_OutsideBurstISI = np.mean(fitness_OutsideBurstISIs)
    print(f'OutsideBurstISI fitness: {fitness_OutsideBurstISI}')
    return fitness_OutsideBurstISI

def fit_CovOutsideBurstISI(**kwargs):
    print('Calculating CoVOutsideBurstISI fitness...')
    CovOutsideBurstISI_target = kwargs['net_activity_targets']['bursting_data']['bursting_summary_data']['CoVOutsideBurstISI']
    target = CovOutsideBurstISI_target['target']
    min_CovOutsideBurstISI = CovOutsideBurstISI_target['min']
    max_CovOutsideBurstISI = CovOutsideBurstISI_target['max']
    weight = CovOutsideBurstISI_target['weight']
    maxFitness = kwargs['maxFitness']
    
    bursting_data_by_unit = kwargs['net_activity_metrics']['bursting_data']['bursting_data_by_unit']
    fitness_CovOutsideBurstISIs = []
    for unit, value in bursting_data_by_unit.items():
        val_CovOutsideBurstISI = bursting_data_by_unit[unit]['cov_isi_outside']
        fitness = the_scoring_function(val_CovOutsideBurstISI, target, weight, maxFitness, min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)
        fitness_CovOutsideBurstISIs.append(fitness)
    fitness_CovOutsideBurstISI = np.mean(fitness_CovOutsideBurstISIs)
    print(f'CovOutsideBurstISI fitness: {fitness_CovOutsideBurstISI}')
    return fitness_CovOutsideBurstISI

def fit_NetworkISI(**kwargs):
    print('Calculating NetworkISI fitness...')
    NetworkISI_target = kwargs['net_activity_targets']['bursting_data']['bursting_summary_data']['MeanNetworkISI']
    target = NetworkISI_target['target']
    min_NetworkISI = NetworkISI_target['min']
    max_NetworkISI = NetworkISI_target['max']
    weight = NetworkISI_target['weight']
    maxFitness = kwargs['maxFitness']
    
    bursting_data_by_unit = kwargs['net_activity_metrics']['bursting_data']['bursting_data_by_unit']
    fitness_NetworkISIs = []
    for unit, value in bursting_data_by_unit.items():
        val_NetworkISI = bursting_data_by_unit[unit]['mean_isi_all']
        fitness = the_scoring_function(val_NetworkISI, target, weight, maxFitness, min_val=min_NetworkISI, max_val=max_NetworkISI)
        fitness_NetworkISIs.append(fitness)
    fitness_NetworkISI = np.mean(fitness_NetworkISIs)
    print(f'NetworkISI fitness: {fitness_NetworkISI}')
    return fitness_NetworkISI

def fit_CovNetworkISI(**kwargs):
    print('Calculating CovNetworkISI fitness...')
    CovNetworkISI_target = kwargs['net_activity_targets']['bursting_data']['bursting_summary_data']['CoVNetworkISI']
    target = CovNetworkISI_target['target']
    min_CovNetworkISI = CovNetworkISI_target['min']
    max_CovNetworkISI = CovNetworkISI_target['max']
    weight = CovNetworkISI_target['weight']
    maxFitness = kwargs['maxFitness']
    
    bursting_data_by_unit = kwargs['net_activity_metrics']['bursting_data']['bursting_data_by_unit']
    fitness_CovNetworkISIs = []
    for unit, value in bursting_data_by_unit.items():
        val_CovNetworkISI = bursting_data_by_unit[unit]['cov_isi_all']
        fitness = the_scoring_function(val_CovNetworkISI, target, weight, maxFitness, min_val=min_CovNetworkISI, max_val=max_CovNetworkISI)
        fitness_CovNetworkISIs.append(fitness)
    fitness_CovNetworkISI = np.mean(fitness_CovNetworkISIs)
    print(f'CovNetworkISI fitness: {fitness_CovNetworkISI}')
    return fitness_CovNetworkISI

def fit_Number_Bursts(**kwargs):
    print('Calculating NumBursts fitness...')
    NumBursts_target = kwargs['net_activity_targets']['bursting_data']['bursting_summary_data']['Number_Bursts']
    target = NumBursts_target['target']
    min_NumBursts = NumBursts_target['min']
    max_NumBursts = NumBursts_target['max']
    weight = NumBursts_target['weight']
    maxFitness = kwargs['maxFitness']
    
    # Method 1: not sure which method to use here, but I think using the bursting_data_by_unit is the wrong way to go.
    # bursting_data_by_unit = kwargs['net_activity_metrics']['bursting_data']['bursting_data_by_unit']
    # fitness_NumBursts = []
    # for unit, value in bursting_data_by_unit.items():
    #     num_bursts = len(bursting_data_by_unit[unit]['bursts'])
    #     val_NumBursts = num_bursts
    #     fitness = the_scoring_function(val_NumBursts, target, weight, maxFitness, min_val=min_NumBursts, max_val=max_NumBursts)
    #     fitness_NumBursts.append(fitness)
    # fitness_NumBurst = np.mean(fitness_NumBursts)
    
    # Method 2: using the network level data
    Number_Bursts = kwargs['net_activity_metrics']['bursting_data']['bursting_summary_data']['Number_Bursts']
    val_NumBursts = Number_Bursts
    fitness_NumBurst = the_scoring_function(val_NumBursts, target, weight, maxFitness, min_val=min_NumBursts, max_val=max_NumBursts)    
    print(f'NumBursts fitness: {fitness_NumBurst}')
    return fitness_NumBurst

def fit_mean_IBI(**kwargs):
    print('Calculating mean_IBI fitness...')
    mean_IBI_target = kwargs['net_activity_targets']['bursting_data']['bursting_summary_data']['mean_IBI']
    target = mean_IBI_target['target']
    min_mean_IBI = mean_IBI_target['min']
    max_mean_IBI = mean_IBI_target['max']
    weight = mean_IBI_target['weight']
    maxFitness = kwargs['maxFitness']
    
    mean_IBI = kwargs['net_activity_metrics']['bursting_data']['bursting_summary_data']['mean_IBI']
    val_mean_IBI = mean_IBI
    fitness_mean_IBI = the_scoring_function(val_mean_IBI, target, weight, maxFitness, min_val=min_mean_IBI, max_val=max_mean_IBI)
    print(f'mean_IBI fitness: {fitness_mean_IBI}')
    return fitness_mean_IBI

def fit_cov_IBI(**kwargs):
    print('Calculating cov_IBI fitness...')
    cov_IBI_target = kwargs['net_activity_targets']['bursting_data']['bursting_summary_data']['cov_IBI']
    target = cov_IBI_target['target']
    min_cov_IBI = cov_IBI_target['min']
    max_cov_IBI = cov_IBI_target['max']
    weight = cov_IBI_target['weight']
    maxFitness = kwargs['maxFitness']
    
    cov_IBI = kwargs['net_activity_metrics']['bursting_data']['bursting_summary_data']['cov_IBI']
    val_cov_IBI = cov_IBI
    fitness_cov_IBI = the_scoring_function(val_cov_IBI, target, weight, maxFitness, min_val=min_cov_IBI, max_val=max_cov_IBI)
    print(f'cov_IBI fitness: {fitness_cov_IBI}')
    return fitness_cov_IBI

def fit_mean_Burst_Peak(**kwargs):
    print('Calculating mean_Burst_Peak fitness...')
    mean_Burst_Peak_target = kwargs['net_activity_targets']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
    target = mean_Burst_Peak_target['target']
    min_mean_Burst_Peak = mean_Burst_Peak_target['min']
    max_mean_Burst_Peak = mean_Burst_Peak_target['max']
    weight = mean_Burst_Peak_target['weight']
    maxFitness = kwargs['maxFitness']
    
    mean_Burst_Peak = kwargs['net_activity_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
    val_mean_Burst_Peak = mean_Burst_Peak
    fitness_mean_Burst_Peak = the_scoring_function(val_mean_Burst_Peak, target, weight, maxFitness, min_val=min_mean_Burst_Peak, max_val=max_mean_Burst_Peak)
    print(f'mean_Burst_Peak fitness: {fitness_mean_Burst_Peak}')
    return fitness_mean_Burst_Peak

def fit_cov_Burst_Peak(**kwargs):
    print('Calculating cov_Burst_Peak fitness...')
    cov_Burst_Peak_target = kwargs['net_activity_targets']['bursting_data']['bursting_summary_data']['cov_Burst_Peak']
    target = cov_Burst_Peak_target['target']
    min_cov_Burst_Peak = cov_Burst_Peak_target['min']
    max_cov_Burst_Peak = cov_Burst_Peak_target['max']
    weight = cov_Burst_Peak_target['weight']
    maxFitness = kwargs['maxFitness']
    
    cov_Burst_Peak = kwargs['net_activity_metrics']['bursting_data']['bursting_summary_data']['cov_Burst_Peak']
    val_cov_Burst_Peak = cov_Burst_Peak
    fitness_cov_Burst_Peak = the_scoring_function(val_cov_Burst_Peak, target, weight, maxFitness, min_val=min_cov_Burst_Peak, max_val=max_cov_Burst_Peak)
    print(f'cov_Burst_Peak fitness: {fitness_cov_Burst_Peak}')
    return fitness_cov_Burst_Peak

def fit_fano_factor(**kwargs):
    print('Calculating fano_factor fitness...')
    fano_factor_target = kwargs['net_activity_targets']['bursting_data']['bursting_summary_data']['fano_factor']
    target = fano_factor_target['target']
    min_fano_factor = fano_factor_target['min']
    max_fano_factor = fano_factor_target['max']
    weight = fano_factor_target['weight']
    maxFitness = kwargs['maxFitness']
    
    fano_factor = kwargs['net_activity_metrics']['bursting_data']['bursting_summary_data']['fano_factor']
    val_fano_factor = fano_factor
    fitness_fano_factor = the_scoring_function(val_fano_factor, target, weight, maxFitness, min_val=min_fano_factor, max_val=max_fano_factor)
    print(f'fano_factor fitness: {fitness_fano_factor}')
    return fitness_fano_factor

'''Main Scoring Function'''        
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
    
    # Set default min and max values if not provided
    if min_val is None:
        min_val = float('-inf')  # Allow all values below positive infinity
    if max_val is None:
        max_val = float('inf')   # Allow all values above negative infinity

    # Calculate fitness score
    if min_val <= val <= max_val:
        return min(np.exp(abs(target_val - val) / weight), maxFitness)
    else:
        return maxFitness

def fitnessFunc(simData=None, **kwargs):       
    
    def get_fitness():
        '''Main fitness calculation function'''
        fitnessVals = {}
        data_source = kwargs['source']
        if data_source == 'experimental':
            
            #Priority 1: Spiking Data
            fitnessVals['rate_fit'] = fit_firing_rates(**kwargs)
            fitnessVals['CoV_rate_fit'] = fit_CoV_firing_rate(**kwargs)
            fitnessVals['ISI_fit'] = fit_ISI(**kwargs)
            fitnessVals['CoV_ISI_fit'] = fit_CoV_ISI(**kwargs)
            
            #Priority 2: Bursting Data
            fitnessVals['baseline_fit'] = fit_baseline(**kwargs)
            fitnessVals['WithinBurstISI_fit'] = fit_WithinBurstISI(**kwargs)
            fitnessVals['CoVWithinBurstISI_fit'] = fit_CovWithinBurstISI(**kwargs)
            fitnessVals['OutsideBurstISI_fit'] = fit_OutsideBurstISI(**kwargs)
            fitnessVals['CoVOutsideBurstISI_fit'] = fit_CovOutsideBurstISI(**kwargs)
            fitnessVals['NetworkISI_fit'] = fit_NetworkISI(**kwargs)
            fitnessVals['CoVNetworkISI_fit'] = fit_CovNetworkISI(**kwargs)
            #fitnessVals['NumUnits_fit'] = fit_NumUnits(**kwargs) no need to fit this
            fitnessVals['Number_Bursts_fit'] = fit_Number_Bursts(**kwargs)
            fitnessVals['mean_IBI_fit'] = fit_mean_IBI(**kwargs)
            fitnessVals['cov_IBI_fit'] = fit_cov_IBI(**kwargs)
            fitnessVals['mean_Burst_Peak_fit'] = fit_mean_Burst_Peak(**kwargs)
            fitnessVals['cov_Burst_Peak_fit'] = fit_cov_Burst_Peak(**kwargs)
            fitnessVals['fano_factor_fit'] = fit_fano_factor(**kwargs)
        elif data_source == 'simulated':
            
            #Priority 1: Spiking Data
            fitnessVals['rate_fit'] = fit_firing_rates(simulated=True, **kwargs)
            fitnessVals['CoV_rate_fit'] = fit_CoV_firing_rate(**kwargs)
            fitnessVals['ISI_fit'] = fit_ISI(**kwargs)
            fitnessVals['CoV_ISI_fit'] = fit_CoV_ISI(**kwargs)
            
            #Priority 2: Bursting Data
            fitnessVals['baseline_fit'] = fit_baseline(**kwargs)
            fitnessVals['WithinBurstISI_fit'] = fit_WithinBurstISI(**kwargs)
            fitnessVals['CoVWithinBurstISI_fit'] = fit_CovWithinBurstISI(**kwargs)
            fitnessVals['OutsideBurstISI_fit'] = fit_OutsideBurstISI(**kwargs)
            fitnessVals['CoVOutsideBurstISI_fit'] = fit_CovOutsideBurstISI(**kwargs)
            fitnessVals['NetworkISI_fit'] = fit_NetworkISI(**kwargs)
            fitnessVals['CoVNetworkISI_fit'] = fit_CovNetworkISI(**kwargs)
            #fitnessVals['NumUnits_fit'] = fit_NumUnits(**kwargs) #no need to fit this
            fitnessVals['Number_Bursts_fit'] = fit_Number_Bursts(**kwargs)
            fitnessVals['mean_IBI_fit'] = fit_mean_IBI(**kwargs)
            fitnessVals['cov_IBI_fit'] = fit_cov_IBI(**kwargs)
            fitnessVals['mean_Burst_Peak_fit'] = fit_mean_Burst_Peak(**kwargs)
            fitnessVals['cov_Burst_Peak_fit'] = fit_cov_Burst_Peak(**kwargs)
            fitnessVals['fano_factor_fit'] = fit_fano_factor(**kwargs)

        #average_fitness, avg_scaled_fitness = fitness_summary_metrics(fitnessVals) #TODO - revise how I do this when I loop in Nfactors
        average_fitness = np.mean([fitnessVals[key] for key in fitnessVals])

        # Save fitness results in .json file
        #save_fitness_results(fitnessVals, average_fitness, avg_scaled_fitness)
        fitnessResults = {key: value for key, value in fitnessVals.items()}
        fitnessResults['average_fitness'] = average_fitness
        #fitnessResults['average_scaled_fitness'] = avg_scaled_fitness
        fitnessResults['maxFitness'] = kwargs['maxFitness']
        output_path = kwargs['fitness_save_path']
        #destination = os.path.join(output_path, f'{kwargs["simLabel"]}_Fitness.json')
        with open(output_path, 'w') as f:
            json.dump(fitnessResults, f, indent=4)
        print(f'fitness results saved to {output_path}')        
        return average_fitness, fitnessVals


   
    '''Main logic of the calculate_fitness function'''
    # Check if the function is being called during simulation - if so, retrieve expanded simData from the call stack
    if simData is not None:
        #during_simulation = True
        #kwargs['simData'] = simData
        kwargs['source'] = 'simulated'
        kwargs = retrieve_sim_data_from_call_stack(simData, **kwargs) 
    else:
        #during_simulation = False
        kwargs['source'] = 'experimental'
    
    #Network activity metrics
    print('Calculating network activity metrics...')
    network_metrics = get_simulated_network_activity_metrics(**kwargs)
    if network_metrics is None:
        print('Network activity metrics could not be calculated.')
        return 1000 # Return a high fitness value to indicate poor performance
    kwargs['network_metrics'] = network_metrics
    
    # Get the fitness
    average_fitness, fitnessVals = get_fitness()

    return average_fitness
    
'''Deprecated functions'''
# def fitness_summary_metrics(fitnessVals):
#     '''Calculate and summarize the fitness metrics.'''
#     fitness_values = {key: fitnessVals[key]['Fit'] for key in fitnessVals if 'Fit' in fitnessVals[key]}
#     fitness_values = [v for v in fitness_values.values() if v is not None]
#     average_fitness = np.mean(fitness_values)

#     min_value, max_value = min(fitness_values), max(fitness_values)
#     if max_value > min_value:
#         normalized_fitness_values = [(v - min_value) / (max_value - min_value) for v in fitness_values]
#     else:
#         normalized_fitness_values = [1 for _ in fitness_values]

#     avg_scaled_fitness = np.mean(normalized_fitness_values)
#     print(f'Average Fitness: {average_fitness}, Average Scaled Fitness: {avg_scaled_fitness}')
#     return average_fitness, avg_scaled_fitness

# def prioritize_fitness(fitnessVals, **kwargs):
#     '''Assign priorities and handle fitness values with maxFitness.'''
#     print('Prioritizing fitness values...')
#     maxFitness = kwargs['maxFitness']
#     priorities = [
#         ['E_rate_fit', 'I_rate_fit', 'E_ISI_fit', 'I_ISI_fit'],  # Priority 1
#         ['baseline_fit'],  # Priority 2
#         ['IBI_fitness', 'burst_frequency_fitness', 'big_burst_fit', 'small_burst_fit', 'thresh_fit', 'bimodal_burst_fit', 'slope_fit']  # Priority 3
#     ]
#     for priority in priorities:
#         if any(fitnessVals[fit]['Fit'] == maxFitness for fit in priority):
#             for lower_priority in priorities[priorities.index(priority) + 1:]:
#                 for fit in lower_priority:
#                     fitnessVals[fit]['Fit'] = maxFitness
#                     fitnessVals[fit]['deprioritized'] = True
#             break

# def save_fitness_results(fitnessVals, average_fitness, avg_scaled_fitness):
#     '''Save fitness results to a file.'''
#     fitnessResults = {key: value for key, value in fitnessVals.items()}
#     fitnessResults['average_fitness'] = average_fitness
#     fitnessResults['average_scaled_fitness'] = avg_scaled_fitness
#     fitnessResults['maxFitness'] = kwargs['maxFitness']

#     output_path = batch_saveFolder or fitness_save_path
#     if exp_mode:
#         destination = os.path.join(output_path, f'{simLabel}_Fitness.json')
#     else:
#         gen_folder = simLabel.split('_cand')[0]
#         destination = os.path.join(output_path, gen_folder, f'{simLabel}_Fitness.json')

#     with open(destination, 'w') as f:
#         json.dump(fitnessResults, f, indent=4)
#     print(f'Fitness results saved to {destination}')