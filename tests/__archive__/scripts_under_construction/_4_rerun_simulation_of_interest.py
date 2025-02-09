global PROGRESS_SLIDES_PATH, SIMULATION_RUN_PATH, REFERENCE_DATA_NPY, CONVOLUTION_PARAMS, DEBUG_MODE
from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import *
from fitness_helper import *
from fitting.calculate_fitness_vCurrent import fitnessFunc
from time import time
from analysis_helper import *
from netpyne import sim
add_repo_root_to_sys_path()

# from netpyne import specs
# cfg = specs.SimConfig()
# print('cfg.py script completed successfully.')


# ===================================================================================================
SIMULATION_RUN_PATH = (    
    #'/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_3/gen_3_cand_173_data.json'
    # '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_11_data.json'
    #'/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_1/gen_1_cand_114_data.json'
    #'/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run5_CDKL5_seeded_evol_5/gen_1/gen_1_cand_1_data.json'
    '/pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_network_simulation_outputs/CDKL5_DIV21/241208_Run8_CDKL5_seeded_evol_8_reloaded/gen_0/gen_0_cand_1_data.json'
    )
REFERENCE_DATA_NPY = (
    '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/fitting/experimental_data_features/network_metrics/CDKL5-E6D_T2_C1_05212024_240611_M06844_Network_000076_network_metrics_well000.npy'
    )
CONVOLUTION_PARAMS = (
    #"workspace/RBS_network_simulations/workspace/optimization_projects/CDKL5_DIV21/convolution_params/241202_convolution_params.py"
    "/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/fitting/convolution_params/241202_convolution_params.py"
    )
cfg_script_path = 'batching/241211_cfg.py'
param_script_path = 'fitting/evol_parameter_spaces/241202_adjusted_evol_params.py'
target_script_path = 'fitting/experimental_data_features/fitness_args_20241205_022033.py'
sim_cfg_path = SIMULATION_RUN_PATH.replace('_data.json', '_cfg.json')
netParams_data_path = SIMULATION_RUN_PATH.replace('_data.json', '_netParams.json')

# options
duration_seconds = 30
DEBUG_MODE = False
save_data = False
# ===================================================================================================

'''main script'''
def reprocess_simulation(
    SIMULATION_RUN_PATH, 
    REFERENCE_DATA_NPY, 
    CONVOLUTION_PARAMS, 
    cfg_script_path=None, 
    param_script_path=None, 
    target_script_path=None, 
    sim_cfg_path=None, 
    netParams_data_path=None, 
    duration_seconds=1, 
    permuted_cfg_path=None, 
    save_data=False, 
    overwrite_cfgs=False):
    
    #
    sim_data_path = SIMULATION_RUN_PATH
    
    # surpress all prints
    #suppress_print() # surpress all prints

    #assert simulation run path is .json file with '_data' in the name
    assert '_data' in sim_data_path and '.json' in sim_data_path
    privileged_print(f"\nSimulation run path: {sim_data_path}")

    #rerun simulation of interest
    privileged_print("Rerunning simulation of interest...")
    start = time()
    def rerun_simulation_of_interest(sim_data_path, duration_seconds, permuted_cfg_path=None, save_data=True, overwrite_cfgs=False):
        # load simulation files 
        # def load_simulation(sim_data_path, permuted_cfg_path=None):
        #     #sim.clearAll()
        #     print(f'simLabel: {sim.cfg.simLabel}')
        #     output = sim.load(sim_data_path, output=True)  #output is none when there are cells in the sim_data I guess?
        #     if permuted_cfg_path is not None: 
        #         print(f'simLabel: {sim.cfg.simLabel}')
        #         sim.loadSimCfg(permuted_cfg_path)
        #         # #sim.net.createCells()
        #         # #sim.clearConns()
        #         # sim.clearObj("conns")
        #         # sim.net.connectCells()
                
        #         # #TODO need to write a function that recreates cells in place. use .util logic/functions.
        #         # # hold shape and size constant.
                
        #         # #TODO: make a slide about this for Roy.
                
                
        #         # sim.net.setupRecording()
        #         print(f'simLabel: {sim.cfg.simLabel}') 
        #     filename = sim.cfg.filename
        # #sim_data_path = SIMULATION_RUN_PATH
        # load_simulation(sim_data_path, permuted_cfg_path=permuted_cfg_path)
        
        # load simulation files
        def load_simulation(sim_data_path):
            try:
                sim.clearAll() #this will fail if sim has no net yet
            except: pass
            sim.load(sim_data_path)
            if permuted_cfg_path is not None:
                sim.loadSimCfg(permuted_cfg_path)
        load_simulation(sim_data_path)
        
        # modify simulation as needed
        instructions = {
            'duration_seconds': duration_seconds,      
            }        
        def modify_simulation(instructions):        
            # rebuild simulation as needed            
            # print simulation label            
            print(f'simLabel: {sim.cfg.simLabel}')
            
            ## things that shouldn't affect the simulation
            
            # #activate core neuron, which is optimized for large networks and hpcs
            # sim.cfg.coreneuron = True            
            # t_vec = sim.h.Vector()             # coreneuron needs to be told to record time explicitly
            # t_vec.record(sim.h._ref_t)
            sim.cfg.dump_coreneuron_model = False # not really sure what this does... it dumps the model instead of running it?
            
            # # deactivate cvode and use fast imem for now
            sim.cfg.cvode_active = False    # pretty sure this must be false for coreneuron to work - which is fine because coreneuron i dont think I need this?
            #                                 # copied from documentation:
            #                                 # Multi order variable time step integration method which may be used in place of the default staggered fixed time step method. 
            #                                 # The performance benefits can be substantial (factor of more than 10) for problems in which all states vary slowly for long periods of time between fast spikes.
            # sim.cfg.use_fast_imem = False   # this doesnt work with coreneuron, but it might help speed up the simulation. 
            #                                 # 
            
            # ## cache efficient is a good idea in general
            # sim.cfg.cache_efficient = True
            
            # ## outputs and stuff
            # sim.cfg.validateNetParams = True #this only plays a role when re-building the net, which we may or may not do.
            sim.cfg.printSynsAfterRule = True
            sim.cfg.saveTiming = True # this creates a time vector in the sim data...which I wish the documentation would have made more obvious
            # sim.cfg.printRunTime = True
            # sim.cfg.savePickle = False # saving pkls just seems like a waste of time lately
            # sim.cfg.saveCSV = False            
            # sim.cfg.timestampFilename = False # this might be useful for version control, but lets keep this false for now
            
            # ## things that might affect the simulation
            sim.cfg.allowSelfConns = False
            sim.cfg.oneSynPerNetcon = True    
            sim.cfg.allowConnsWithWeight0 = True
            
            ## turns out that modifying the cfg object alone doesn't actually change the simulation.
            ## need to modify intantiated objects in the sim object using modifyCells, modifyConns, modifySynMechs, etc.
            ## they expect params similar to those in the netParams object.
            
            #sim.net.connectCells()
            
            netParams = sim.net.params
            # params = {
            #     'conds': sim.cfg.__dict__.copy(),
            # }
            #sim.net.
            #sim.net.modifyCells(params)
            
            # clear and rebuild the net
            # #params = sim.cfg.__dict__.copy()
            # #params = sim.net.params.__dict__.copy()
            # cellParams = sim.net.params.cellParams
            # for pop, params in cellParams.items():
            #     #params = cell_pop
            #     sim.net.modifyCells(params)
                
            # cellConds ={
            #     # if desired, I can filter synmechs modified by any tag i guess. 
            #     # e.g. {'pop': 'PYR', 'ynorm': [0.1, 0.6]} targets connections of cells from the ‘PYR’ population with normalized depth within 0.1 and 0.6.
            # }
            
            # synMechParams = sim.net.params.synMechParams
            # for pop, params in synMechParams.items():
            #     #params = cell_pop
            #     params['cellConds'] = {'pop': 'E'}
            #     params['']                
            #     sim.net.modifySynMechs(params)
                
            #     params['cellConds'] = {'pop': 'I'}
            #     sim.net.modifySynMechs(params)
            #     #sim.net.modifySynMechs()
                
            # for pop, params in cellParams.items():
            #     #params = cell_pop
            #     sim.net.modifyConns(params)
                
            # # sim.net.createPops()
            # # sim.net.createCells()
            # # sim.net.connectCells()

            #modify duration of the simulation. Intuitively, modulating this last makes sense.
            sim.cfg.duration = duration_seconds*1e3
            
            return t_vec if 't_vec' in locals() else None

        t_vec = modify_simulation(instructions)
        
        
        #run simulation
        sim.simulate()
        
        # post process as needed
        def post_simulation_processing(tvec):
            sim.gatherData()
            # if sim.cfg.coreneuron: #save time vector to sim data - which for some reason is lost by coreneuron
            #     time_vector = np.array(t_vec)
            #     sim.allSimData.t = time_vector    
        post_simulation_processing(t_vec)

        #save data
        saved_files = None
        if save_data:
            activate_print()
            saved_files = sim.saveData()
            print("Data saved.")
            suppress_print()
        return saved_files
    saved_files = rerun_simulation_of_interest(sim_data_path, duration_seconds, permuted_cfg_path=permuted_cfg_path, save_data=save_data, overwrite_cfgs=overwrite_cfgs)
    from pprint import pprint
    activate_print()
    pprint(saved_files)
    
    # update sim_data_path if necessary
    def update_sim_data_path(saved_files):
        # if any of the saved files are _data.json, update sim_data_path
        if any('_data.json' in f for f in saved_files):
            new_path = [f for f in saved_files if '_data.json' in f][0]
        return new_path
    if saved_files is not None: sim_data_path = update_sim_data_path(saved_files)   
    print("Simulation rerun complete.")
    print(f"Time taken: {time()-start} seconds")
    suppress_print()

    # re-fit simulation of interest
    privileged_print("Refitting simulation of interest...")
    start = time()
    from fitting.calculate_fitness_vCurrent import fitnessFunc
    def refit_simulation_of_interest(sim_data_path, target_script_path, saved_files=None):   
        #sim.clearAll()
        #sim.load(sim_data_path)
        simData = sim.allSimData
        convolution_params = import_module_from_path(CONVOLUTION_PARAMS)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        target_script_path = os.path.join(script_dir, target_script_path)
        fitnessFuncArgs = import_module_from_path(target_script_path).fitnessFuncArgs
        def get_fitness_file_path(sim_data_path, saved_files):
            # if saved_files is not None:
            #     #find the file with data and json in the name
            #     data_file = [f for f in saved_files if '_data' in f and 'json' in f][0]
            #     #replace _data with _fitness
            #     fitness_file = data_file.replace('_data', '_fitness')
            # else:
            fitness_file = sim_data_path.replace('_data', '_fitness')
            return fitness_file
        fitness_save_path = get_fitness_file_path(sim_data_path, saved_files)        
        kwargs = {
            'simConfig': sim.cfg,
            'conv_params': convolution_params.conv_params,
            'popData': sim.net.allPops,
            'cellData': sim.net.allCells, #not actually used in the fitness calculation, but whatever
            'targets': fitnessFuncArgs['targets'],
            'maxFitness': fitnessFuncArgs['maxFitness'],
            'features': fitnessFuncArgs['features'],
            'fitness_save_path': fitness_save_path,
        }
        average_fitness = fitnessFunc(simData, mode='simulated data', **kwargs)
        return average_fitness
    average_fitness = refit_simulation_of_interest(sim_data_path, target_script_path, saved_files=saved_files)
    privileged_print(f"Average fitness: {average_fitness}")
    privileged_print("Refit complete.")
    privileged_print(f"Time taken: {time()-start} seconds")

    # re-plot simulation of interest - re-generate summary plot and all associated plots
    privileged_print("Replotting simulation of interest...")
    start = time()
    def replot_simulation_of_interest(sim_data_path, convolution_params_path, reference_data_npy, average_fitness):
        start_network_metrics = time()
        def get_network_metrics():
            convolution_params = import_module_from_path(CONVOLUTION_PARAMS)
            #get network metrics
            kwargs = {
                'simData': sim.allSimData,
                'simConfig': sim.cfg,
                'conv_params': convolution_params.conv_params,
                'popData': sim.net.allPops,
                'cellData': sim.net.allCells, #not actually used in the fitness calculation, but whatever
            }
            error, kwargs = calculate_network_metrics(kwargs)
            return error, kwargs
        error, kwargs = get_network_metrics()
        if error: return error
        privileged_print("\tNetwork metrics calculated - kwargs dict created.")
        privileged_print(f'\tTime taken: {time()-start_network_metrics} seconds')

        #plot raster
        start_raster = time()
        def plot_simulated_raster_wrapper(ax = None, subplot=False, **kwargs):
            network_metrics = kwargs['network_metrics']
            spiking_data_by_unit = network_metrics['spiking_data']['spiking_data_by_unit']
            popData = sim.net.allPops
            E_gids = popData['E']['cellGids']
            I_gids = popData['I']['cellGids']
            
            if ax is None:
                fig, ax_raster = plt.subplots(1, 1, figsize=(16, 4.5))
            else:
                ax_raster = ax
                subplot = True
                
            ax_raster = plot_raster(ax_raster, spiking_data_by_unit, E_gids=E_gids, I_gids=I_gids, data_type='simulated')
            
            #break if subplot
            if subplot:
                #plt.close()
                return ax_raster

            if DEBUG_MODE:
                #save local for debugging
                #dev_dir = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts'
                dev_dir = os.path.dirname(os.path.realpath(__file__))
                plt.savefig(os.path.join(dev_dir, '_raster_plot.png'), dpi=300)
                #save local for debugging
                
            #save wherever data is saved
            #sim_data_path = SIMULATION_RUN_PATH
            raster_plot_path = sim_data_path.replace('_data', '_raster_plot')
            #remove file type and replace with png
            raster_plot_path = raster_plot_path.replace('.json', '.png')
            plt.savefig(raster_plot_path, dpi=300)
            print(f"Raster plot saved to {raster_plot_path}")
            
            #save as pdf
            raster_plot_path = sim_data_path.replace('_data', '_raster_plot')
            raster_plot_path = raster_plot_path.replace('.json', '.pdf')
            plt.savefig(raster_plot_path)
            print(f"Raster plot saved to {raster_plot_path}")
            plt.close()
            
            raster_plots_paths = [
                raster_plot_path,
                raster_plot_path.replace('.pdf', '.png'),
            ]
            
            return raster_plots_paths
        raster_plot_paths = plot_simulated_raster_wrapper(**kwargs)
        privileged_print("\tIndividual raster plots saved.")
        privileged_print(f'\tTime taken: {time()-start_raster} seconds')

        #plot bursting summary
        start_bursting = time()
        def plot_simulated_bursting_wrapper(ax = None, subplot=False, **kwargs):
            if ax is None:
                fig, new_ax = plt.subplots(1, 1)
                fig.set_size_inches(16, 4.5)
            else:
                new_ax = ax
                subplot = True #if ax is passed in, then we are plotting on a subplot
            
            #
            conv_params = kwargs['conv_params']
            SpikeTimes = kwargs['network_metrics']['spiking_data']['spiking_times_by_unit']
            new_ax, _ = plot_network_activity_aw(new_ax, SpikeTimes, **conv_params) #TODO need to make sure this function agrees with mandar. would be best if we shared a function here.
            new_ax.set_title('Bursting summary')
            new_ax.set_xlabel('Time (s)')
            new_ax.set_ylabel('Fire rate (Hz)')
            
            #break if subplot
            if subplot:
                #plt.close()
                return new_ax
            
            plt.tight_layout()
            
            if DEBUG_MODE:
                # Save local for debugging
                dev_dir = os.path.dirname(os.path.realpath(__file__))
                plt.savefig(os.path.join(dev_dir, '_bursting_plot.png'), dpi=300)
                # Save local for debugging
            
            # save wherever data is saved
            #sim_data_path = SIMULATION_RUN_PATH
            bursting_plot_path = sim_data_path.replace('_data', '_bursting_plot')
            #remove file type and replace with png
            bursting_plot_path = bursting_plot_path.replace('.json', '.png')
            plt.savefig(bursting_plot_path, dpi=300)
            print(f"Bursting plot saved to {bursting_plot_path}")
            
            #save as pdf
            bursting_plot_path = sim_data_path.replace('_data', '_bursting_plot')
            bursting_plot_path = bursting_plot_path.replace('.json', '.pdf')
            plt.savefig(bursting_plot_path)
            print(f"Bursting plot saved to {bursting_plot_path}")
            plt.close()
            
            bursting_plot_paths = [
                bursting_plot_path,
                bursting_plot_path.replace('.pdf', '.png'),
            ]
            
            return bursting_plot_paths    
        bursting_plot_paths = plot_simulated_bursting_wrapper(**kwargs)
        privileged_print("\tIndividual bursting plots saved.")
        privileged_print(f'\tTime taken: {time()-start_bursting} seconds')

        # combine plots into a single summary plot
        start_summary = time()
        def plot_simulation_summary(**kwargs):
            fig, ax = plt.subplots(2, 1, figsize=(16, 9))
            raster_plot_ax, bursting_plot_ax = ax
            subplot = True
            raster_plot_ax = plot_simulated_raster_wrapper(ax=raster_plot_ax, subplot=subplot, **kwargs)
            bursting_plot_ax = plot_simulated_bursting_wrapper(ax=bursting_plot_ax, subplot=subplot, **kwargs)
            
            #make both plots share the same x-axis
            raster_plot_ax.get_shared_x_axes().join(raster_plot_ax, bursting_plot_ax)    
            plt.tight_layout()
            
            if DEBUG_MODE:
                # Save local for debugging
                dev_dir = os.path.dirname(os.path.realpath(__file__))
                plt.savefig(os.path.join(dev_dir, '_summary_plot.png'), dpi=300)
                # Save local for debugging
                
            # save wherever data is saved
            #sim_data_path = SIMULATION_RUN_PATH
            summary_plot_path = sim_data_path.replace('_data', '_summary_plot')
            #remove file type and replace with png
            summary_plot_path = summary_plot_path.replace('.json', '.png')
            plt.savefig(summary_plot_path, dpi=300)
            print(f"Summary plot saved to {summary_plot_path}")
            
            #save as pdf
            summary_plot_path = sim_data_path.replace('_data', '_summary_plot')
            summary_plot_path = summary_plot_path.replace('.json', '.pdf')
            plt.savefig(summary_plot_path)  
            print(f"Summary plot saved to {summary_plot_path}")
            plt.close()
            
            summary_plot_paths = [
                summary_plot_path,
                summary_plot_path.replace('.pdf', '.png'),
            ]    
            return summary_plot_paths
        plot_simulation_summary(**kwargs)
        privileged_print("\tSimulation summary plot saved.")
        privileged_print(f'\tTime taken: {time()-start_summary} seconds')

        # comparison plot against reference data snippet
        start_comparison = time()
        def plot_comparision_plot(ax_list=None, subplot=False, **kwargs):
            #activate_print() #for debugging
            #fig, ax = plt.subplots(4, 1, figsize=(16, 9))
            if ax_list is None:
                fig, ax_list = plt.subplots(4, 1, figsize=(16, 9))
                sim_raster_ax, sim_bursting_ax, ref_raster_ax, ref_bursting_ax = ax_list
            else:
                #assert that there be 4 axes in the ax list
                assert len(ax_list) == 4, "There must be 4 axes in the ax_list."
                sim_raster_ax, sim_bursting_ax, ref_raster_ax, ref_bursting_ax = ax_list
                subplot = True
            
            #plot simulated raster
            sim_raster_ax = plot_simulated_raster_wrapper(ax=sim_raster_ax, subplot=True, **kwargs)
            
            #plot simulated bursting
            sim_bursting_ax = plot_simulated_bursting_wrapper(ax=sim_bursting_ax, subplot=True, **kwargs)
            
            # plot reference raster
            def plot_reference_raster_wrapper(ax=None, subplot=False, sim_data_length=None, **kwargs):
                #load npy ref data
                print(ax)
                print(subplot)
                #print(kwargs)
                
                #load npy ref data
                ref_data = np.load(
                    REFERENCE_DATA_NPY, 
                    allow_pickle=True
                    ).item()
                network_metrics = ref_data
                spiking_data_by_unit = network_metrics['spiking_data']['spiking_data_by_unit'].copy()
                unit_ids = [unit_id for unit_id in spiking_data_by_unit.keys()]
                
                # Trim reference data to match the length of simulated data
                if sim_data_length is not None:
                    for unit_id in unit_ids:
                        #spiking_data_by_unit[unit_id] = spiking_data_by_unit[unit_id][:sim_data_length]
                        #max_spike_time_index = np.argmax(spiking_data_by_unit[unit_id])
                        spike_times = spiking_data_by_unit[unit_id]['spike_times']
                        spike_times = spike_times[spike_times < sim_data_length]
                        spiking_data_by_unit[unit_id]['spike_times'] = spike_times
                
                if ax is None:
                    fig, ax_raster = plt.subplots(1, 1, figsize=(16, 4.5))
                else:
                    ax_raster = ax
                    subplot = True
                    
                ax_raster = plot_raster(ax_raster, spiking_data_by_unit, unit_ids=unit_ids, data_type='experimental')
                
                if subplot:
                    #plt.close()
                    print("subplot")
                    return ax_raster
                
                if DEBUG_MODE:
                    #save local for debugging
                    #dev_dir = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts'
                    dev_dir = os.path.dirname(os.path.realpath(__file__))
                    plt.savefig(os.path.join(dev_dir, '_ref_raster_plot.png'), dpi=300)
                    #save as pdf
                    plt.savefig(os.path.join(dev_dir, '_ref_raster_plot.pdf'))
                    #save local for debugging
                
                
                #TODO: semi unfinished compared to similar function for simulated raster
            #sim_data_length = len(kwargs['network_metrics']['spiking_data']['spiking_data_by_unit'][0])
            sim_data_length = kwargs['simConfig']['duration']/1000 #in seconds
            ref_raster_ax = plot_reference_raster_wrapper(ax=ref_raster_ax, subplot=True, sim_data_length = sim_data_length, **kwargs)
            
            #plot reference bursting
            def plot_reference_bursting_wrapper(ax=None, subplot=False, sim_data_length=None, **kwargs):
                ref_data = np.load(REFERENCE_DATA_NPY, allow_pickle=True).item()
                network_metrics = ref_data
                conv_params = kwargs['conv_params']
                SpikeTimes = network_metrics['spiking_data']['spiking_times_by_unit']
                if ax is None:
                    fig, new_ax = plt.subplots(1, 1)
                    fig.set_size_inches(16, 4.5)
                else:
                    new_ax = ax
                    subplot = True
                    
                if sim_data_length is not None:
                    for unit_id in SpikeTimes.keys():
                        spike_times = SpikeTimes[unit_id]
                        spike_times = spike_times[spike_times < sim_data_length]
                        SpikeTimes[unit_id] = spike_times
                
                new_ax, _ = plot_network_activity_aw(new_ax, SpikeTimes, **conv_params) #TODO need to make sure this function agrees with mandar. would be best if we shared a function here.
                
                new_ax.set_title('Bursting summary')
                new_ax.set_xlabel('Time (s)')
                new_ax.set_ylabel('Fire rate (Hz)')
                
                #break if subplot
                if subplot:
                    #plt.close()
                    return new_ax
                
                plt.tight_layout()
                
                if DEBUG_MODE:
                    # Save local for debugging
                    dev_dir = os.path.dirname(os.path.realpath(__file__))
                    plt.savefig(os.path.join(dev_dir, '_ref_bursting_plot.png'), dpi=300)
                    # plot as pdf
                    plt.savefig(os.path.join(dev_dir, '_ref_bursting_plot.pdf'))
                    # Save local for debugging
                
                # TODO: semi unfinished compared to similar function for simulated bursting
            sim_data_length = kwargs['simConfig']['duration']/1000 #in seconds
            ref_bursting_ax = plot_reference_bursting_wrapper(ax=ref_bursting_ax, subplot=True, sim_data_length=sim_data_length, **kwargs)
            
            # Ensure y-axis is the same for bursting plots
            sim_bursting_ylim = sim_bursting_ax.get_ylim()
            ref_bursting_ylim = ref_bursting_ax.get_ylim()
            
            # Calculate the combined y-axis limits
            combined_ylim = [
                min(sim_bursting_ylim[0], ref_bursting_ylim[0]) * 0.8,
                max(sim_bursting_ylim[1], ref_bursting_ylim[1]) * 1.2
            ]
            
            # Set the same y-axis limits for both axes
            sim_bursting_ax.set_ylim(combined_ylim)
            ref_bursting_ax.set_ylim(combined_ylim)
            
            # Make all four plots share the same x-axis as simulated raster
            sim_raster_ax.get_shared_x_axes().join(sim_raster_ax, sim_bursting_ax)
            sim_raster_ax.get_shared_x_axes().join(sim_raster_ax, ref_raster_ax)
            sim_raster_ax.get_shared_x_axes().join(sim_raster_ax, ref_bursting_ax)  

            #
            if subplot:
                #plt.tight_layout()
                #plt.close()
                return [sim_raster_ax, sim_bursting_ax, ref_raster_ax, ref_bursting_ax] 
            
            plt.tight_layout()
            
            if DEBUG_MODE:
                # Save local for debugging
                dev_dir = os.path.dirname(os.path.realpath(__file__))
                fig.savefig(os.path.join(dev_dir, '_comparison_plot.png'), dpi=300)
                # Save local for debugging
            
            # save wherever data is saved
            #sim_data_path = SIMULATION_RUN_PATH
            comparison_plot_path = sim_data_path.replace('_data', '_comparison_plot')
            #remove file type and replace with png
            comparison_plot_path = comparison_plot_path.replace('.json', '.png')
            fig.savefig(comparison_plot_path, dpi=300)
            print(f"Comparison plot saved to {comparison_plot_path}")
            
            #save as pdf
            comparison_plot_path = sim_data_path.replace('_data', '_comparison_plot')
            comparison_plot_path = comparison_plot_path.replace('.json', '.pdf')
            fig.savefig(comparison_plot_path)
            print(f"Comparison plot saved to {comparison_plot_path}")
            plt.close()
        plot_comparision_plot(**kwargs)
        privileged_print("\tComparison plot saved.")
        privileged_print(f'\tTime taken: {time()-start_comparison} seconds')

        # build comparison summary slide
        start_summary_slide = time()
        def build_comparision_summary_slide(sim_data_path):
            fig, ax_list = plt.subplots(4, 1, figsize=(16, 9))
            
            #plot comparison plot
            plot_comparision_plot(ax_list=ax_list, subplot=True, **kwargs)
            
            #remove x-axis labels from all plots
            for ax in ax_list:
                ax.set_xlabel('')
                
            #remove titles from all plots
            for ax in ax_list:
                ax.set_title('')
                
            #remove x-axis ticks from all plots except the bottom one
            for ax in ax_list[:-1]:
                ax.set_xticks([])
                
            # add 'simulated' and 'reference' labels above each raster plot
            ax_list[0].text(0.5, 1.1, 'Simulated', ha='center', va='center', transform=ax_list[0].transAxes, fontsize=12)
            ax_list[2].text(0.5, 1.1, 'Reference', ha='center', va='center', transform=ax_list[2].transAxes, fontsize=12)
            
            # add time(s) label to the bottom plot
            ax_list[-1].set_xlabel('Time (s)')
                    
            #create space to the right of plots for text
            fig.subplots_adjust(right=0.6)
            
            #create some space at the bottom for one line of text
            fig.subplots_adjust(bottom=0.1)
            
            #add text
            spiking_summary = kwargs['network_metrics']['spiking_data']['spiking_summary_data']
            bursting_summary = kwargs['network_metrics']['bursting_data']['bursting_summary_data']
            simulated_spiking_data = kwargs['network_metrics']['simulated_data']
            #simulated_bursting_data = kwargs['network_metrics']['simulated_data']
            
            text = (
                # f"Summary of comparison between simulated and reference data:\n"
                # f"Simulated raster plot is shown in the top left, reference raster plot is shown in the bottom left.\n"
                # f"Simulated bursting plot is shown in the top right, reference bursting plot is shown in the bottom right.\n"
                f"Simulated spiking summary:\n"
                f"  - Mean firing rate: {spiking_summary['MeanFireRate']} Hz\n"
                f"  - Coefficient of variation: {spiking_summary['CoVFireRate']}\n"
                f"  - Mean ISI: {spiking_summary['MeanISI']} s\n"
                f"  - Coefficient of variation of ISI: {spiking_summary['CoV_ISI']}\n"
                f"  - Mean firing rate E: {simulated_spiking_data['MeanFireRate_E']} Hz\n"
                f"  - Mean CoV FR E: {simulated_spiking_data['CoVFireRate_E']}\n"
                f"  - Mean firing rate I: {simulated_spiking_data['MeanFireRate_I']} Hz\n"
                f"  - Mean CoV FR I: {simulated_spiking_data['CoVFireRate_I']}\n"
                f"  - Mean ISI E: {simulated_spiking_data['MeanISI_E']} s\n"
                f"  - CoV ISI E: {simulated_spiking_data['CoV_ISI_E']}\n"
                f"  - Mean ISI I: {simulated_spiking_data['MeanISI_I']} s\n"
                f"  - CoV ISI I: {simulated_spiking_data['CoV_ISI_I']}\n"
                
                f"Simulated bursting summary:\n"        
                #number of units
                f"  - Number of units: {bursting_summary['NumUnits']}\n"
                #baseline
                f"  - Baseline: {bursting_summary['baseline']} Hz\n"
                #fanofactor
                f"  - Fanofactor: {bursting_summary['fano_factor']}\n"
                #num bursts
                f"  - Number of bursts: {bursting_summary['Number_Bursts']}\n"
                #mean burst rate
                f"  - Mean burst rate: {bursting_summary['mean_Burst_Rate']} Hz\n" #TODO: this is not in the data        
                #mean burst peak
                f"  - Mean burst peak: {bursting_summary['mean_Burst_Peak']} Hz\n"
                #burst peak CoV
                f"  - CoV burst peak: {bursting_summary['cov_Burst_Peak']}\n"
                #mean IBI
                f"  - Mean IBI: {bursting_summary['mean_IBI']} s\n"
                #CoV IBI
                f"  - CoV IBI: {bursting_summary['cov_IBI']}\n"
                #mean within burst isi
                f"  - Mean within burst ISI: {bursting_summary['MeanWithinBurstISI']} s\n"
                #CoV within burst isi
                f"  - CoV within burst ISI: {bursting_summary['CoVWithinBurstISI']}\n"
                #mean outside burst isi
                f"  - Mean outside burst ISI: {bursting_summary['MeanOutsideBurstISI']} s\n"
                #CoV outside burst isi
                f"  - CoV outside burst ISI: {bursting_summary['CoVOutsideBurstISI']}\n"
                #mean whole network isi
                f"  - Mean network ISI: {bursting_summary['MeanNetworkISI']} s\n"
                #CoV whole network isi
                f"  - CoV network ISI: {bursting_summary['CoVNetworkISI']}\n"
                )
            
            #append reference metrics to text
            ref_data = np.load(REFERENCE_DATA_NPY, allow_pickle=True).item()
            ref_spiking_summary = ref_data['spiking_data']['spiking_summary_data']
            ref_bursting_summary = ref_data['bursting_data']['bursting_summary_data']    
            text += (
                f"\n"
                f"\n"
                f"Reference spiking summary:\n"
                f"  - Mean firing rate: {ref_spiking_summary['MeanFireRate']} Hz\n"
                f"  - Coefficient of variation: {ref_spiking_summary['CoVFireRate']}\n"
                f"  - Mean ISI: {ref_spiking_summary['MeanISI']} s\n"
                f"  - Coefficient of variation of ISI: {ref_spiking_summary['CoV_ISI']}\n"
                
                f"Reference bursting summary:\n"
                #number of units
                f"  - Number of units: {ref_bursting_summary['NumUnits']}\n"
                #baseline
                f"  - Baseline: {ref_bursting_summary['baseline']} Hz\n"
                #fanofactor
                f"  - Fanofactor: {ref_bursting_summary['fano_factor']}\n"
                #num bursts
                f"  - Number of bursts: {ref_bursting_summary['Number_Bursts']}\n"
                #mean burst rate
                #f"  - Mean burst rate: {ref_bursting_summary['mean_Burst_Rate']} Hz\n" #TODO: this is not in the data        
                #mean burst peak
                f"  - Mean burst peak: {ref_bursting_summary['mean_Burst_Peak']} Hz\n"
                #burst peak CoV
                f"  - CoV burst peak: {ref_bursting_summary['cov_Burst_Peak']}\n"
                #mean IBI
                f"  - Mean IBI: {ref_bursting_summary['mean_IBI']} s\n"
                #CoV IBI
                f"  - CoV IBI: {ref_bursting_summary['cov_IBI']}\n"
                #mean within burst isi
                f"  - Mean within burst ISI: {ref_bursting_summary['MeanWithinBurstISI']} s\n"
                #CoV within burst isi
                f"  - CoV within burst ISI: {ref_bursting_summary['CoVWithinBurstISI']}\n"
                #mean outside burst isi
                f"  - Mean outside burst ISI: {ref_bursting_summary['MeanOutsideBurstISI']} s\n"
                #CoV outside burst isi
                f"  - CoV outside burst ISI: {ref_bursting_summary['CoVOutsideBurstISI']}\n"
                #mean whole network isi
                f"  - Mean network ISI: {ref_bursting_summary['MeanNetworkISI']} s\n"
                #CoV whole network isi
                f"  - CoV network ISI: {ref_bursting_summary['CoVNetworkISI']}\n"
                )
            
            #add average fitness to text
            text += (
                f"\n"
                f"\n"
                f"\nAverage fitness: {average_fitness}"
                ) 
            
            #add text to the right of the plots
            fig.text(0.65, 0.5, text, ha='left', va='center', fontsize=9)
            
            #add data path at the bottom of the slide
            #fig.text(0.5, 0.05, f"Data path: {SIMULATION_RUN_PATH}", ha='center', va='center', fontsize=10)
            #go a little lower and add data path
            fig.text(0.5, 0.02, f"Data path: {sim_data_path}", ha='center', va='center', fontsize=9) 
            
            if DEBUG_MODE:
                #save local for debugging
                dev_dir = os.path.dirname(os.path.realpath(__file__))
                fig.savefig(os.path.join(dev_dir, '_comparison_summary_slide.png'), dpi=300)
                #save as pdf
            
            # save wherever data is saved
            sim_data_path = sim_data_path
            comparison_summary_slide_path = sim_data_path.replace('_data', '_comparison_summary_slide')
            #remove file type and replace with png
            comparison_summary_slide_path = comparison_summary_slide_path.replace('.json', '.png')
            fig.savefig(comparison_summary_slide_path, dpi=300)
            privileged_print(f"Comparison summary slide saved to {comparison_summary_slide_path}")
            
            #save as pdf
            comparison_summary_slide_path = sim_data_path.replace('_data', '_comparison_summary_slide')
            comparison_summary_slide_path = comparison_summary_slide_path.replace('.json', '.pdf')
            fig.savefig(comparison_summary_slide_path)
            privileged_print(f"Comparison summary slide saved to {comparison_summary_slide_path}")
            plt.close()
        build_comparision_summary_slide(sim_data_path)
        privileged_print("\tComparison summary slide saved.")
        privileged_print(f'\tTime taken: {time()-start_summary_slide} seconds')
    replot_simulation_of_interest(sim_data_path, CONVOLUTION_PARAMS, REFERENCE_DATA_NPY, average_fitness)
    privileged_print("Replotting complete.")
    privileged_print(f"Time taken: {time()-start} seconds")

if __name__ == '__main__':
    reprocess_simulation(
        SIMULATION_RUN_PATH, 
        REFERENCE_DATA_NPY, 
        CONVOLUTION_PARAMS, 
        cfg_script_path=cfg_script_path, 
        param_script_path=param_script_path, 
        target_script_path=target_script_path, 
        sim_cfg_path=sim_cfg_path, 
        netParams_data_path=netParams_data_path, 
        duration_seconds=duration_seconds,
        save_data=save_data,
        overwrite_cfgs=True,
        )