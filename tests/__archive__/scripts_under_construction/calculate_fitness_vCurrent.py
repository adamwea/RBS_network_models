from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.fitness_helper import *

def fitnessFunc(simData=None, mode='optimizing', **kwargs):
    """
    Main logic of the calculate_fitness function.
    Ensures the function does not crash and always returns a fitness value.
    """
        
    '''execute the main logic of the calculate_fitness function'''
    try:
        # Select mode and execute corresponding logic
        if mode == 'optimizing':
            error, kwargs = handle_optimizing_mode(simData, kwargs)
        elif mode == 'simulated data':
            error, kwargs = handle_simulated_data_mode(simData, kwargs)
        elif mode == 'experimental data':
            error, kwargs = handle_experimental_data_mode(kwargs)
        else:
            raise ValueError(f'Unknown mode: {mode}')

        # Save kwargs to a file
        # sim_data_path = kwargs.get('data_file_path', None)
        # assert sim_data_path is not None, 'Data file path not provided.'
        # kwargs_path = sim_data_path.replace('_data.json', '_kwargs.json')
        # kwargs['kwargs_path'] = kwargs_path
        # with open(kwargs_path, 'w') as f:
        #     json.dump(kwargs, f, indent=4)
        
        # Handle potential errors from network metrics calculation
        if error is not None:
            #return error
            return 1000
        
        return calculate_and_save_fitness(kwargs)

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
    
    
    # from fitness_helper import *

    # def fitnessFunc(simData=None, mode='optimizing', **kwargs):
        
    #     '''Main logic of the calculate_fitness function'''
    #     try: #ensure that the function does not crash - always return a fitness value
            
    #         # Select run mode
    #         # default mode is 'optimizing' - i.e. the function is being called during optimization batch runs
    #         # 'simulated data' mode is used when the function is being called in isolation testing on simulated data
    #         # 'experimental data' mode is used when the function is being called in isolation testing on experimental data
            
    #         #parse initialization by mode, get network metrics for each mode
    #         #This is where the function is called in the optimization batch runs
    #         #This will be the first time fitness is calculated 
    #         # (unless I'm rerunning the exact same optimization)
            
    #         if mode == 'optimizing':

    #             #implemented = False
    #             #assert implemented, 'Optimizing mode not yet implemented.'       
    #             # Check if the function is being called during simulation - if so, retrieve expanded simData from the call stack
    #             if simData is not None:
    #                 kwargs['source'] = 'simulated'
    #                 #import workspace.RBS_network_simulations._archive.temp_user_args as temp_user_args
    #                 import batch_scripts.temp_user_args as temp_user_args
                    
    #                 # # handle existing fitness results
    #                 recalculate_fitness = temp_user_args.USER_recalculate_fitness
    #                 if not recalculate_fitness:
    #                     #from workspace.RBS_network_simulation_optimization_tools.workspace.optimization_projects.CDKL5_DIV21_dep.extract_simulated_data import get_candidate_and_job_path_from_call_stack
    #                     from batch_scripts.extract_simulated_data import get_candidate_and_job_path_from_call_stack
                        
                        
    #                     candidate_path, job_path = get_candidate_and_job_path_from_call_stack()
    #                     fitness_save_path = f'{candidate_path}_fitness.json'
    #                     existing_fitness = handle_existing_fitness(fitness_save_path)
    #                     if existing_fitness is not None:
    #                         return existing_fitness
    #                     else: pass # just puttin this here so that its obv
    #                                 # that fitness is being calculated for the first time if this 
    #                                 # block is not entered

    #                 # Get the candidate and job paths from the call stack
    #                 kwargs = retrieve_sim_data_from_call_stack(simData, **kwargs)
    #                 error, kwargs = calculate_network_metrics(kwargs)
    #         elif mode == 'simulated data':
    #             kwargs['source'] = 'simulated'
    #             candidate_path = kwargs['simConfig']['filename']
    #             fitness_save_path = f'{candidate_path}_fitness.json'
                
    #             # Calculate network metrics - handle potential errors
    #             assert simData is not None, 'Simulated data must be provided in "simulated data" mode.' #obvs
    #             kwargs.update({'simData': simData})
    #             error, kwargs = calculate_network_metrics(kwargs)
    #             if error is not None:
    #                 return error            
                
    #         elif mode == 'experimental data':
    #             kwargs['source'] = 'experimental'
    #             #TODO: make sure this is fully implemented before use.
    #             implemented = False
    #             assert implemented, 'Experimental data source not yet implemented.'
    
    
    #         # # Calculate network metrics - handle potential errors
    #         # error, kwargs = calculate_network_metrics(kwargs)
    #         # if error is not None:
    #         #     return error

    #         # Get the fitness - handle known errors
    #         try:
    #             average_fitness, fitnessResults = get_fitness(kwargs['source'], **kwargs)
    #             fitnessResults['average_fitness'] = average_fitness
    #             fitnessResults['maxFitness'] = kwargs['maxFitness']
    #             save_fitness_results(kwargs['fitness_save_path'], fitnessResults)
                
    #             # Check for dealbreakers in the fitness results
    #             #kwargs.update({'network_metrics': network_
    #             deal_broken = check_dealbreakers(fitnessResults, **kwargs)
    #             try: assert not deal_broken, 'Dealbreakers found in fitness results.'
    #             except: return 1000
                
    #             # Return the average fitness
    #             return average_fitness
    #         except Exception as e:
    #             error_trace = str(e)
    #             fitnessResults = {
    #                 'average_fitness': kwargs['maxFitness'],
    #                 'maxFitness': kwargs['maxFitness'],
    #                 'error': 'acceptable' if any(error in error_trace for error in []) else 'new',
    #                 'error_trace': error_trace
    #             }
    #             print(f'Error calculating fitness: {e}')
    #             save_fitness_results(kwargs['fitness_save_path'], fitnessResults)
    #             return 1000
            
    #     # General error handling if all else fails
    #     except Exception as e:
    #         print(f'Error calculating fitness: {e}')
    #         fitnessResults = {
    #             'average_fitness': kwargs['maxFitness'],
    #             'maxFitness': kwargs['maxFitness'],
    #             'error': 'general',
    #             'error_trace': str(e)
    #         }
    #         save_fitness_results(kwargs['fitness_save_path'], fitnessResults)
    #         return 1000