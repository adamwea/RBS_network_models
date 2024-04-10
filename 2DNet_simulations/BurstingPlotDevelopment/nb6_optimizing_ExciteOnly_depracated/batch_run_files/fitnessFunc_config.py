##FITNESS FUNCTION TARGETS 
pops = {}
#firing rate targets
pops['rate_targets'] = {}
pops['rate_targets']['E'] = {'target': 7.5, 'width': 2.5, 'min': 1}
pops['rate_targets']['I'] = {'target': 30, 'width': 10, 'min': 2}
#burst peak targets
pops['burts_peak_targets'] = {'target': 15, 'width': 10, 'min': 1}
#burst IBI targets
pops['IBI_targets'] = {'target': 3000, 'width': 2000 , 'min': 1000} #ms
#baseline targets
pops['baseline_targets'] = {'target': 1.5, 'width': 1.5 , 'min': 0} #ms
#firing rate variance targets
pops['rate_slope'] = {'target': 0, 'width': 0.5, 'min': 0}

# fitness function
fitnessFuncArgs = {}
fitnessFuncArgs['pops'] = pops
fitnessFuncArgs['maxFitness'] = 1000

def fitnessFunc(simData, **kwargs):
    try: from batch_run_files.plot_config import plot_sim_figs
    except: from plot_config import plot_sim_figs
    ##find relevant batch object in call stack
    # (I dont understand how this works, figure it out later)
    import inspect

    def find_batch_object_and_sim_label():
        # Get the current frame
        current_frame = inspect.currentframe()

        # Iterate through the call stack
        while current_frame:
            # Check each local variable in the frame
            for name, obj in current_frame.f_locals.items():
                # If the object is of type Batch, return it and its simLabel
                if type(obj).__name__ == 'Batch':
                    simLabel = obj.cfg.simLabel if obj else None
                    return obj, simLabel

            # If not, move to the next frame
            current_frame = current_frame.f_back

        # If no Batch object is found, print a message and return None
        print("Batch object not found in the caller frames.")
        return None, None

    # Use the function to get the Batch object and simLabel
    batch, simLabel = find_batch_object_and_sim_label()

    # If no simLabel is found, print a message
    assert simLabel is not None, "SimLabel not found in the caller frames."
    assert batch is not None, "Batch object not found in the caller frames."
    
    #from netpyne import sim
    import numpy as np
    import sys
    sys.path.insert(0, '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5_optimizing/batch_run_files')
    from aw_batch_tools import measure_network_activity
    #from netpyne import sim
    from scipy.stats import linregress
    #rasterData = sim.analysis.prepareRaster()
    maxFitness = kwargs['maxFitness']
    net_activity_params = {'binSize': .03*1000, 'gaussianSigma': .12*1000, 'thresholdBurst': 1.0}
    binSize = net_activity_params['binSize']
    gaussianSigma = net_activity_params['gaussianSigma']
    thresholdBurst = net_activity_params['thresholdBurst']
    #output_path = '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5_optimizing/output'
    output_path = batch.saveFolder

    #
    try:
        #prepare raster data
        #rasterData = sim.analysis.prepareRaster()
        rasterData = simData
        maxFitness = kwargs['maxFitness']
        assert len(rasterData['spkt']) > 0, 'Error: rasterData has no elements. burstPeak, baseline, slopeFitness, and IBI fitness set to 1000.'                       
        
        # Generate the network activity plot with a size of (10, 5)
        net_metrics = measure_network_activity(
            rasterData, 
            binSize=binSize, 
            gaussianSigma=gaussianSigma, 
            thresholdBurst=thresholdBurst, 
            #figSize=(network_activity_width, network_activity_height), 
            #saveFig=network_activity_path
        )
        burstPeakValues = net_metrics['burstPeakValues']
        burstPeakTimes = net_metrics['burstPeakTimes'] #not needed
        IBIs = net_metrics['IBIs']
        #firingRate = net_metrics['firingRate'] #solved using other method
        #timeVector = net_metrics['timeVector'] #not needed
        baseline = net_metrics['baseline']

        # Get mean and std of firing rate, make sure it stays within 2.5 std of mean
        try:
            # Firing rate fitness
            pops = kwargs['pops']
            rate_slope= pops['rate_slope']
            maxFitness = kwargs['maxFitness']

            # Get the firing rate from the network metrics
            firingRate = net_metrics['firingRate']

            # Calculate the trendline of the firing rate
            slope, intercept, r_value, p_value, std_err = linregress(range(len(firingRate)), firingRate)

            # Calculate the fitness as the absolute difference between the slope and the target slope
            slopeFitness = abs(rate_slope['target'] - slope)/rate_slope['width'] if slope > rate_slope['min'] else maxFitness

            # Print the slope and its fitness
            print('Slope of firing rate: %.3f, Fitness: %.3f' % (slope, slopeFitness))
        except Exception as e:
            print(f'Error calculating firing rate slope fitness.')
            print(f'Error: {e}')
            # Set fitness values to maxFitness
            slopeFitness = maxFitness 
        
        # Calculate the fitness for each value in burstPeakValues
        try: 
            #burstPeakValue Fitness
            pops = kwargs['pops']
            pops_peaks = pops['burts_peak_targets']
            print(kwargs)
            maxFitness = kwargs['maxFitness']
            #popFitnessBurstPeak = [None for i in pops.items()]
            assert len(burstPeakValues) > 0, 'Error: burstPeakValues has no elements. BurstVal_fitness set to maxFitness.'
            popFitnessBurstPeak = [
                min(np.exp(abs(pops_peaks['target'] - value) / pops_peaks['width']), maxFitness)
                if value > pops_peaks['min'] else maxFitness for value in burstPeakValues
            ]

            # Calculate the mean fitness
            BurstVal_fitness = np.mean(popFitnessBurstPeak)

            # Calculate the average burstPeak
            avg_burstPeak = np.mean(burstPeakValues)

            # Print the average burstPeak and its fitness
            print('Average burstPeak: %.3f, Fitness: %.3f' % (avg_burstPeak, BurstVal_fitness))
        except Exception as e:
            print(f'Error calculating burst peak fitness.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            BurstVal_fitness = maxFitness
            pass

        # IBI Fitness
        try:
            assert len(burstPeakTimes) > 0, 'Error: burstPeakTimes has less than 1 element. IBI could be calculated.'
            assert len(burstPeakValues) > 1, 'Error: burstPeakValues has less than 2 elements. IBI could be calculated.'
            assert len(IBIs) > 0, 'Error: IBIs has no elements. IBI could not be calculated.'
            pops = kwargs['pops']
            pops_IBI = pops['IBI_targets']
            maxFitness = kwargs['maxFitness']

            popFitnessIBI = [
                min(np.exp(abs(pops_IBI['target'] - value) / pops_IBI['width']), maxFitness)
                if value > pops_IBI['min'] else maxFitness for value in IBIs
            ]
            IBI_fitness = np.mean(popFitnessIBI)

            #Calc average IBI
            avg_IBI = np.mean(IBIs)

            #print average IBI and its fitness
            print('Average IBI: %.3f, Fitness: %.3f' % (avg_IBI, IBI_fitness))
        except Exception as e:
            print(f'Error calculating IBI fitness.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            IBI_fitness = maxFitness
            pass

        # Baseline Fitness
        try:
            assert baseline is not None, 'Error: baseline is None. Baseline could not be calculated.'
            #assert len(baseline) > 0, 'Error: baseline has no elements. Baseline could not be calculated.'
            pops = kwargs['pops']
            pops_baseline = pops['baseline_targets']
            maxFitness = kwargs['maxFitness']
            baseline_target = pops_baseline['target']
            baseline_width = pops_baseline['width']
            baseline_min = pops_baseline['min']

            baselineFitness = min(np.exp(abs(baseline_target - baseline) / baseline_width), maxFitness) if baseline > baseline_min else maxFitness

            #print average baseline and its fitness
            print('baseline: %.3f, Fitness: %.3f' % (baseline, baselineFitness)) 
        except:
            print(f'Error calculating baseline fitness.')
            print(f'Error: {e}')
            #set fitness values to maxFitness
            maxFitness = kwargs['maxFitness']
            baselineFitness = maxFitness
            pass      

    except Exception as e:
        print(f'Error calculating network activity metrics.')
        print(f'Error: {e}')
        #set fitness values to maxFitness
        maxFitness = kwargs['maxFitness']
        BurstVal_fitness = maxFitness
        IBI_fitness = maxFitness
        baselineFitness = maxFitness
        slopeFitness = maxFitness
        pass

    #Firing Rate Fitness
    try:
        pops = kwargs['pops']
        pops_rate = pops['rate_targets']
        maxFitness = kwargs['maxFitness']
        popFitness = [None for i in pops.items()]
        # This is a list comprehension that calculates the fitness for each population in the simulation.
        popFitness = [
            min(np.exp(abs(v['target'] - simData['popRates'][k]) / v['width']), maxFitness)
            if simData["popRates"][k] > v['min'] else maxFitness
            for k, v in pops_rate.items()
        ]
        rate_fitness = np.mean(popFitness)
        popInfo = '; '.join(['%s rate=%.1f fit=%1.f'%(p,r,f) for p,r,f in zip(list(simData['popRates'].keys()), list(simData['popRates'].values()), popFitness)])
        print('  '+popInfo)
    except Exception as e:
        print(f'Error calculating firing rate fitness.')
        #print(f'Error: {e}')
        #set fitness values to maxFitness
        maxFitness = kwargs['maxFitness']
        rate_fitness = maxFitness
        pass

    
    #average fitness
    fitness = (rate_fitness + BurstVal_fitness + IBI_fitness + baselineFitness + slopeFitness) / 5

    ##normalized and scaled average fitness - mitigate the effect of outliers
    # Assuming fitness values are stored in a list
    fitness_values = [rate_fitness, BurstVal_fitness, IBI_fitness, baselineFitness, slopeFitness]

    # Calculate min and max values
    min_value = min(fitness_values)
    max_value = max(fitness_values)

    # Normalize the fitness values
    normalized_fitness_values = [(value - min_value) / (max_value - min_value) for value in fitness_values]
    if np.isnan(normalized_fitness_values[0]):
        #set all values to 1
        normalized_fitness_values = [1, 1, 1, 1, 1]
        #normalized_fitness_values[0] = 1
    # Scale the normalized values back to the original range (0-1000)
    scaled_fitness_values = [value * 1000 for value in normalized_fitness_values]

    # Calculate the average of the scaled fitness values
    average_scaled_fitness = sum(scaled_fitness_values) / len(scaled_fitness_values)

    #save fitness values
    fitnessResults = {}
    fitnessResults['rate_fitness'] = rate_fitness
    fitnessResults['BurstVal_fitness'] = BurstVal_fitness
    fitnessResults['IBI_fitness'] = IBI_fitness
    fitnessResults['baselineFitness'] = baselineFitness
    fitnessResults['slopeFitness'] = slopeFitness
    fitnessResults['fitness'] = fitness
    fitnessResults['average_scaled_fitness'] = average_scaled_fitness
    fitnessResults['maxFitness'] = maxFitness

    #output_path = '/mnt/disk15tb/adam/git_workspace/netpyne_2DNetworkSimulations/2DNet_simulations/BurstingPlotDevelopment/nb5_optimizing/output'
    output_path = batch.saveFolder

    fitness_thresh = 500
    fitness_check = max(fitness, average_scaled_fitness)
    if fitness_check < fitness_thresh:   
    #if len(rasterData['spkt']) > 0:
        #plot figures
        plot_sim_figs(output_path, simLabel)

    #save fitness results to file
    import json
    #get folder from simLabel
    gen_folder = simLabel.split('_cand')[0]
    with open(f'{output_path}/{gen_folder}/{simLabel}_Fitness.json', 'w') as f:
        json.dump(fitnessResults, f)
    #return fitness
    return average_scaled_fitness