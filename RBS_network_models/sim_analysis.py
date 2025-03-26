# notes ===================================================================================================
'''
'''
# imports ===================================================================================================
#from RBS_network_models.developing.CDKL5.DIV21.src.fitnessFunc import fitnessFunc
from time import time
#from RBS_network_models.CDKL5.DIV21.src.fitnessFunc import fitnessFunc
from RBS_network_models.fitnessFunc import fitnessFunc_v2
from netpyne import sim
import numpy as np
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import get_simulated_network_activity_metrics
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import compute_network_metrics
import matplotlib.pyplot as plt
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_raster
import os
# funcs ===================================================================================================
''' newer functions '''
def process_simulation_v3(kwargs): # aw 2025-03-12 09:51:10 - updated to use new fitnessFunc

    #assertions
    assert 'simData' in kwargs, "simData must be provided in kwargs."
    assert 'popData' in kwargs, "popData must be provided in kwargs."
    assert 'cellData' in kwargs, "cellData must be provided in kwargs."
    
    # unpack kwargs
    sim_data_path = kwargs.get('sim_data_path', None)
    conv_params = kwargs.get('conv_params', None)
    mega_params = kwargs.get('mega_params', None)
    #fitnessFuncArgs = kwargs.get('fitnessFuncArgs', None)
    reference_data_npy = kwargs.get('reference_data_path', None)
    debug_mode = kwargs.get('debug_mode', False)
    
    # add to kwargs
    kwargs['debug_mode'] = debug_mode
    kwargs['max_workers'] = 16
    kwargs['plot_wfs'] = False
    kwargs['burst_sequencing'] = False
    kwargs['run_parallel'] = True
    kwargs['dtw_temp'] = None
    # kwargs['simData'] = sim.allSimData
    # kwargs['cellData'] = sim.net.allCells
    # kwargs['popData'] = sim.net.allPops
    
    # get network_metrics
    # TODO: left off here with updating this func. 
    simulated_network_metrics = compute_network_metrics(
        #conv_params=conv_params,
        #mega_params=mega_params,
        source='simulated',
        **kwargs
    )
    
    perm_network_data = []
    perm_network_data.append(simulated_network_metrics)
    from RBS_network_models.sensitivity_analysis import plot_permutations
    #remove 'sim_data_path' from kwargs
    kwargs.pop('sim_data_path')
    plot_permutations(perm_network_data, kwargs)
    
    # # re-fit simulation of interest
    # refit = False # TODO: need to refractor this. Shouldnt be calculating network metrics twice.
    # if refit:
    #     start = time()
    #     print("Calculating average fitness...")
    #     average_fitness = fit_simulation(
    #         sim_data_path,
    #         conv_params=conv_params,
    #         mega_params=mega_params,
    #         #fitnessFuncArgs=fitnessFuncArgs,
    #         ) 
        
    #     # TODO: There seems to be some inconsistency in FR calculated by network metrics vs NETPYNE - need to investigate this.
    #     print(f"Average fitness: {average_fitness}")
    #     print("Refit complete.")
    #     print(f"Time taken: {time()-start} seconds")
    # else:
    #     average_fitness = None

    # # re-plot simulation of interest - re-generate summary plot and all associated plots
    # print("Replotting simulation of interest...")
    # start = time()
    # pkwargs = {
    #     'sim_data_path': sim_data_path,
    #     'average_fitness': average_fitness,
    #     'conv_params': conv_params,
    #     'mega_params': mega_params,
    #     #'fitnessFuncArgs': fitnessFuncArgs,
    #     'reference_data_npy': reference_data_npy,
    #     'trim_start': 5,
    #     'DEBUG_MODE': debug_mode,
    #     }
    
    # comparison_summary_slide_paths = plot_simulation_v2(pkwargs)
    # print("Replotting complete.")
    # print(f"Time taken: {time()-start} seconds")
    
    #return comparison_summary_slide_paths

def plot_simulation_v2(pwkargs):
    
    
    # sub-functions ===================================================================================================
    
    # main ===================================================================================================
    
    # assertions
    assert sim_data_path is not None, "sim_data_path must be provided."
    assert conv_params is not None, "conv_params must be provided."
    assert fitnessFuncArgs is not None, "fitnessFuncArgs must be provided."
    assert reference_data_npy is not None, "reference_data_npy must be provided."
    assert mega_params is not None, "mega_params must be provided."
    
    # get network metrics
    # start_network_metrics = time()
    # kwargs = get_simulated_network_metrics_wrapper(
    #     conv_params=conv_params,
    #     mega_params=mega_params,
    #     sim_data_path=sim_data_path,        
    # )
    # #if error: return error
    # # privileged_print("\tNetwork metrics calculated - kwargs dict created.")
    # # privileged_print(f'\tTime taken: {time()-start_network_metrics} seconds')
    # print("\tNetwork metrics calculated - kwargs dict created.")
    # print(f'\tTime taken: {time()-start_network_metrics} seconds')

    #plot raster
    start_raster = time()
    def plot_simulated_raster_wrapper(ax = None, subplot=False, trim_start = None, **kwargs):
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
        
        # # if trim_start, trim first x seconds from the start of the simulation
        # if trim_start is not None and trim_start > 0 and trim_start < ax_raster.get_xlim()[1]:
        #     ax_raster.set_xlim(trim_start, ax_raster.get_xlim()[1])
        # elif trim_start is not None and trim_start > 0 and trim_start > ax_raster.get_xlim()[1]:
        #     modified_trim = ax_raster.get_xlim()[1]*0.1
        #     ax_raster.set_xlim(modified_trim, ax_raster.get_xlim()[1])
        #     print('boop')
        
        #plt.tight_layout()
        #fig.tight_layout()
        # print('tight bbox:')
        # print(ax_raster.get_tightbbox())
        # ax_raster.set_xlim(ax_raster.get_xlim()[0], ax_raster.get_xlim()[1])    
        
        #set the tightest possible xlim for the data in the raster plot
        # for some reason when raster plot is generated, left side is slightly negative and right side is slightly more positive than the actual data.
        # so using xlim doesnt work here.
        # #x_data = ax_raster.lines[0].get_xdata()
        # true_max_x = 0
        # true_min_x = 0
        # for line in ax_raster.lines:
        #     x_data = line.get_xdata()
        #     max_x = max(x_data)
        #     min_x = min(x_data)
        #     if max_x > true_max_x:
        #         true_max_x = max_x
        #     if min_x < true_min_x:
        #         true_min_x = min_x
        # ax_raster.set_xlim(true_min_x, true_max_x)
        
        # ##set the tightest possible ylim for the data in the raster plot
        # true_max_y = 0
        # true_min_y = 0
        # for line in ax_raster.lines:
        #     y_data = line.get_ydata()
        #     max_y = max(y_data)
        #     min_y = min(y_data)
        #     if max_y > true_max_y:
        #         true_max_y = max_y
        #     if min_y < true_min_y:
        #         true_min_y = min_y
        # ax_raster.set_ylim(true_min_y, true_max_y)
        # #print(f'true min y: {min(y_data)}')
                    
        
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
        if '.json' in raster_plot_path:
            raster_plot_path = raster_plot_path.replace('.json', '.png')
        elif '.pkl' in raster_plot_path:
            raster_plot_path = raster_plot_path.replace('.pkl', '.png')
        #raster_plot_path = raster_plot_path.replace('.json', '.png')
        plt.savefig(raster_plot_path, dpi=300)
        print(f"Raster plot saved to {raster_plot_path}")
        
        #save as pdf
        raster_plot_path = sim_data_path.replace('_data', '_raster_plot')
        #raster_plot_path = raster_plot_path.replace('.json', '.pdf')
        if '.json' in raster_plot_path:
            raster_plot_path = raster_plot_path.replace('.json', '.pdf')
        elif '.pkl' in raster_plot_path:
            raster_plot_path = raster_plot_path.replace('.pkl', '.pdf')
        plt.savefig(raster_plot_path)
        print(f"Raster plot saved to {raster_plot_path}")
        plt.close()
        
        raster_plots_paths = [
            raster_plot_path,
            raster_plot_path.replace('.pdf', '.png'),
        ]
        
        return raster_plots_paths
    raster_plot_paths = plot_simulated_raster_wrapper(trim_start = trim_start, **kwargs)
    # privileged_print("\tIndividual raster plots saved.")
    # privileged_print(f'\tTime taken: {time()-start_raster} seconds')
    print("\tIndividual raster plots saved.")
    print(f'\tTime taken: {time()-start_raster} seconds')

    #plot bursting summary
    start_bursting = time()
    def plot_simulated_bursting_wrapper(ax = None, subplot=False, trim_start = None, **kwargs):
        if ax is None:
            #fig, new_ax = plt.subplots(1, 1)
            fig, new_ax = plt.subplots(2, 1, figsize=(16, 9))
            fig.set_size_inches(16, 4.5)
        else:
            new_ax = ax
            subplot = True #if ax is passed in, then we are plotting on a subplot
        
        #
        # conv_params = kwargs['conv_params']
        # SpikeTimes = kwargs['network_metrics']['spiking_data']['spiking_times_by_unit']
        #from DIV21.utils.fitness_helper import plot_network_activity_aw
        #from RBS_network_models.developing.utils.analysis_helper import plot_network_activity_aw
        #from RBS_network_models.network_analysis import plot_network_activity_aw
        #bursting_ax, _ = plot_network_activity_aw(new_ax, SpikeTimes, **conv_params) #TODO need to make sure this function agrees with mandar. would be best if we shared a function here.
        # bursting_ax = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['ax']
        # mega_ax = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['ax']
        if 'ax' in kwargs['network_metrics']['bursting_data']:
            bursting_ax = kwargs['network_metrics']['bursting_data']['ax']
        else:
            bursting_ax = None
        if 'ax' in kwargs['network_metrics']['mega_bursting_data']:
            mega_ax = kwargs['network_metrics']['mega_bursting_data']['ax']
        else:
            mega_ax = None
        
        # # HACK
        # mega_conv_params = kwargs['conv_params'].copy()
        # mega_conv_params['binSize'] *= 5
        # mega_conv_params['gaussianSigma'] *= 15
        #mega_ax, _ = plot_network_activity_aw(new_ax, SpikeTimes, **mega_conv_params) #TODO need to make sure this function agrees with mandar. would be best if we shared a function here.
        
        from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_network_bursting_experimental
        #new_ax = plot_network_bursting_experimental(new_ax, bursting_ax, mega_ax=mega_ax)
        new_ax[0] = plot_network_bursting_experimental(new_ax[0], bursting_ax) #TODO: rename this func. It works for both experimental and simulated bursting
        new_ax[1] = plot_network_bursting_experimental(new_ax[1], mega_ax, mode='mega')
        
        # from RBS_network_models.network_analysis import plot_network_summary
        # new_ax = plot_network_summary(new_ax, bursting_ax, mega_ax=mega_ax)            
        
        # # if trim_start, trim first x seconds from the start of the simulation
        # if trim_start is not None and trim_start > 0 and trim_start < new_ax.get_xlim()[1]:
        #     new_ax.set_xlim(trim_start, new_ax.get_xlim()[1])
        # elif trim_start is not None and trim_start > 0 and trim_start > new_ax.get_xlim()[1]:
        #     modified_trim = new_ax.get_xlim()[1]*0.1
        #     new_ax.set_xlim(modified_trim, new_ax.get_xlim()[1])    
        
        # new_ax.set_title('Bursting summary')
        # new_ax.set_xlabel('Time (s)')
        # new_ax.set_ylabel('Fire rate (Hz)')
        new_ax[0].set_title('Bursting summary')
        #new_ax[0].set_xlabel('Time (s)')
        new_ax[0].set_ylabel('Fire rate (Hz)')
        new_ax[1].set_title('Mega bursting summary')
        new_ax[1].set_xlabel('Time (s)')
        new_ax[1].set_ylabel('Fire rate (Hz)')
        
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
        #bursting_plot_path = bursting_plot_path.replace('.json', '.png')
        if '.json' in bursting_plot_path:
            bursting_plot_path = bursting_plot_path.replace('.json', '.png')
        elif '.pkl' in bursting_plot_path:
            bursting_plot_path = bursting_plot_path.replace('.pkl', '.png')
        plt.savefig(bursting_plot_path, dpi=300)
        print(f"Bursting plot saved to {bursting_plot_path}")
        
        #save as pdf
        bursting_plot_path = sim_data_path.replace('_data', '_bursting_plot')
        #bursting_plot_path = bursting_plot_path.replace('.json', '.pdf')
        if '.json' in bursting_plot_path:
            bursting_plot_path = bursting_plot_path.replace('.json', '.pdf')
        elif '.pkl' in bursting_plot_path:
            bursting_plot_path = bursting_plot_path.replace('.pkl', '.pdf')
        plt.savefig(bursting_plot_path)
        print(f"Bursting plot saved to {bursting_plot_path}")
        plt.close()
        
        bursting_plot_paths = [
            bursting_plot_path,
            bursting_plot_path.replace('.pdf', '.png'),
        ]
        
        return bursting_plot_paths    
    bursting_plot_paths = plot_simulated_bursting_wrapper(trim_start = trim_start, **kwargs)
    # privileged_print("\tIndividual bursting plots saved.")
    # privileged_print(f'\tTime taken: {time()-start_bursting} seconds')
    print("\tIndividual bursting plots saved.")
    print(f'\tTime taken: {time()-start_bursting} seconds')

    # combine plots into a single summary plot
    start_summary = time()
    def plot_simulation_summary(trim_start = None, **kwargs):
        #fig, ax = plt.subplots(2, 1, figsize=(16, 9))
        # fig, ax = plt.subplots(2, 1, figsize=(16, 9))
        # fig2, ax[1] = plt.subplot(2, 1, 2)
        fig, ax = plt.subplots(3, 1, figsize=(16, 9))
        raster_plot_ax = ax[0]
        bursting_plot_ax = ax[1:]

        subplot = True
        raster_plot_ax = plot_simulated_raster_wrapper(ax=raster_plot_ax, subplot=subplot, 
                                                       #trim_start = trim_start, 
                                                       **kwargs)
        bursting_plot_ax = plot_simulated_bursting_wrapper(ax=bursting_plot_ax, subplot=subplot, 
                                                           #trim_start = trim_start, 
                                                           **kwargs)
        
        #make both plots share the same x-axis
        #raster_plot_ax.get_shared_x_axes().join(raster_plot_ax, bursting_plot_ax[0], bursting_plot_ax[1])
        bursting_plot_ax[0].set_xlim(raster_plot_ax.get_xlim())
        bursting_plot_ax[1].set_xlim(raster_plot_ax.get_xlim())
        print(f"simulation summary plot xlims set to {raster_plot_ax.get_xlim()[0]} to {raster_plot_ax.get_xlim()[1]}")
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
        #summary_plot_path = summary_plot_path.replace('.json', '.png')
        if '.json' in summary_plot_path:
            summary_plot_path = summary_plot_path.replace('.json', '.png')
        elif '.pkl' in summary_plot_path:
            summary_plot_path = summary_plot_path.replace('.pkl', '.png')
        plt.savefig(summary_plot_path, dpi=300)
        print(f"Summary plot saved to {summary_plot_path}")
        
        #save as pdf
        summary_plot_path = sim_data_path.replace('_data', '_summary_plot')
        #summary_plot_path = summary_plot_path.replace('.json', '.pdf')
        if '.json' in summary_plot_path:
            summary_plot_path = summary_plot_path.replace('.json', '.pdf')
        elif '.pkl' in summary_plot_path:
            summary_plot_path = summary_plot_path.replace('.pkl', '.pdf')
        plt.savefig(summary_plot_path)  
        print(f"Summary plot saved to {summary_plot_path}")
        plt.close()
        
        summary_plot_paths = [
            summary_plot_path,
            summary_plot_path.replace('.pdf', '.png'),
        ]    
        return summary_plot_paths
    plot_simulation_summary(trim_start=trim_start, **kwargs)
    # privileged_print("\tSimulation summary plot saved.")
    # privileged_print(f'\tTime taken: {time()-start_summary} seconds')
    print("\tSimulation summary plot saved.")
    print(f'\tTime taken: {time()-start_summary} seconds')

    # comparison plot against reference data snippet
    start_comparison = time()
    #activate_print()
    def plot_comparision_plot(ax_list=None, subplot=False, trim_start = None, **kwargs):
        #activate_print() #for debugging
        #fig, ax = plt.subplots(4, 1, figsize=(16, 9))
        if ax_list is None:
            #fig, ax_list = plt.subplots(4, 1, figsize=(16, 9))
            fig, ax_list = plt.subplots(6, 1, figsize=(16, 9))
            #sim_raster_ax, sim_bursting_ax, ref_raster_ax, ref_bursting_ax = ax_list
            sim_raster_ax, sim_bursting_ax, sim_mega_bursting_ax, ref_raster_ax, ref_bursting_ax, ref_mega_bursting_ax = ax_list
        else:
            #assert that there be 4 axes in the ax list
            #assert len(ax_list) == 4, "There must be 4 axes in the ax_list."
            assert len(ax_list) == 6, "There must be 6 axes in the ax_list."
            #sim_raster_ax, sim_bursting_ax, ref_raster_ax, ref_bursting_ax = ax_list
            sim_raster_ax, sim_bursting_ax, sim_mega_bursting_ax, ref_raster_ax, ref_bursting_ax, ref_mega_bursting_ax = ax_list
            subplot = True
        
        #plot simulated raster
        sim_raster_ax = plot_simulated_raster_wrapper(ax=sim_raster_ax, subplot=True, trim_start=trim_start, **kwargs)
        
        #plot simulated bursting
        sim_bursting_axes = ax_list[1:3]
        #sim_bursting_ax = plot_simulated_bursting_wrapper(ax=sim_bursting_ax, subplot=True, trim_start=trim_start, **kwargs)
        sim_bursting_axes = plot_simulated_bursting_wrapper(ax=sim_bursting_axes, subplot=True, trim_start=trim_start, **kwargs)
        
        # plot reference raster
        def plot_reference_raster_wrapper(ax=None, subplot=False, sim_data_length=None, **kwargs):
            #load npy ref data
            print(ax)
            print(subplot)
            #print(kwargs)
            
            #load npy ref data
            ref_data = np.load(
                #REFERENCE_DATA_NPY,
                reference_data_npy, 
                allow_pickle=True
                ).item()
            network_metrics = ref_data
            
            #HACK patch for different ref data format            
            try: spiking_data_by_unit = network_metrics['spiking_data']['spiking_data_by_unit'].copy()
            except: spiking_data_by_unit = network_metrics['spiking_data']['spiking_metrics_by_unit'].copy()
            
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
        def plot_reference_bursting_wrapper(ax=None, subplot=False, sim_data_length=None, trim_start = None, **kwargs):
            #ref_data = np.load(REFERENCE_DATA_NPY, allow_pickle=True).item()
            ref_data = np.load(
                #REFERENCE_DATA_NPY,
                reference_data_npy, 
                allow_pickle=True).item()
            network_metrics = ref_data
            conv_params = kwargs['conv_params']
            SpikeTimes = network_metrics['spiking_data']['spiking_times_by_unit']
            if ax is None:
                #fig, new_ax = plt.subplots(1, 1)
                fig, new_ax = plt.subplots(2, 1)
                fig.set_size_inches(16, 4.5)
            else:
                new_ax = ax
                subplot = True
                
            if sim_data_length is not None:
                for unit_id in SpikeTimes.keys():
                    spike_times = SpikeTimes[unit_id]
                    spike_times = spike_times[spike_times < sim_data_length]
                    SpikeTimes[unit_id] = spike_times
            
            #from DIV21.utils.fitness_helper import plot_network_activity_aw
            #from RBS_network_models.developing.utils.analysis_helper import plot_network_activity_aw
            #from RBS_network_models.network_analysis import plot_network_activity_aw
            #new_ax[0], _ = plot_network_activity_aw(new_ax, SpikeTimes, **conv_params) #TODO need to make sure this function agrees with mandar. would be best if we shared a function here.
            #new_ax[1], _ = plot_network_activity_aw(new_ax, SpikeTimes, **conv_params) #TODO need to make sure this function agrees with mandar. would be best if we shared a function here.
            # new_ax.set_title('Bursting summary')
            # new_ax.set_xlabel('Time (s)')
            # new_ax.set_ylabel('Fire rate (Hz)')
            
            
            bursting_ax = network_metrics['bursting_data']['bursting_summary_data']['ax']
            mega_ax = network_metrics['mega_bursting_data']['bursting_summary_data']['ax']
            
            #plot
            from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_network_bursting_experimental
            new_ax[0] = plot_network_bursting_experimental(new_ax[0], bursting_ax) #TODO: rename this func. It works for both experimental and simulated bursting
            new_ax[1] = plot_network_bursting_experimental(new_ax[1], mega_ax, mode='mega')
            
            #adjust axes titles and labels
            new_ax[0].set_title('Bursting summary')
            #new_ax[0].set_xlabel('Time (s)')
            new_ax[0].set_ylabel('Fire rate (Hz)')
            new_ax[1].set_title('Mega bursting summary')
            new_ax[1].set_xlabel('Time (s)')
            new_ax[1].set_ylabel('Fire rate (Hz)')           
            
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
        ref_bursting_axes = ax_list[4:6]
        ref_bursting_axes = plot_reference_bursting_wrapper(ax=ref_bursting_axes, subplot=True, sim_data_length=sim_data_length, trim_start = trim_start, **kwargs)
        
        # apply trim_start if trim_start is not None and trim_start > 0:
        # only apply to sim raster plot - allow x axis alignmnet code below to handle the rest
        if trim_start is not None and trim_start > 0:
            print("Trimming start of simulated data")
            # ensure trim_start is less than or equal to 10% of the x-axis range, else just trim first 10% of the x-axis range
            sim_duration = sim_raster_ax.get_xlim()[1] - sim_raster_ax.get_xlim()[0]
            if trim_start > sim_duration * 0.1:
                trim_start = sim_duration * 0.1
            sim_raster_ax.set_xlim(trim_start, sim_raster_ax.get_xlim()[1])
            print(f'Trimmed simulated raster plot range: {sim_raster_ax.get_xlim()[0]} to {sim_raster_ax.get_xlim()[1]}')
            
        
        # ensure xaxis of refernce plots matches simulated raster
        #print("Setting xlim of reference plots to match simulated raster")
        ref_raster_ax.set_xlim(sim_raster_ax.get_xlim())
        ref_bursting_ax.set_xlim(sim_raster_ax.get_xlim())
        ref_mega_bursting_ax.set_xlim(sim_raster_ax.get_xlim())
        sim_bursting_ax.set_xlim(sim_raster_ax.get_xlim())
        sim_mega_bursting_ax.set_xlim(sim_raster_ax.get_xlim())
        
        
        # Ensure y-axis is the same for bursting plots
        #print("Adjusting y-axis limits based on x-axis limits")
        def adjust_ylim_based_on_xlim(ax):
            x_data = ax.lines[0].get_xdata()
            y_data = ax.lines[0].get_ydata()
            xlim = ax.get_xlim()
            filtered_y_data = [y for x, y in zip(x_data, y_data) if xlim[0] <= x <= xlim[1]]
            if filtered_y_data:
                ax.set_ylim(min(filtered_y_data) * 0.8, max(filtered_y_data) * 1.2)

        adjust_ylim_based_on_xlim(sim_bursting_ax)
        adjust_ylim_based_on_xlim(ref_bursting_ax)  
        
        
        # Calculate the combined y-axis limits
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
        sim_mega_bursting_ax.set_ylim(combined_ylim)
        sim_mega_bursting_ax.set_ylim(combined_ylim)
        
        # # Make all four plots share the same x-axis as simulated raster
        # sim_raster_ax.get_shared_x_axes().join(sim_raster_ax, sim_bursting_ax)
        # sim_raster_ax.get_shared_x_axes().join(sim_raster_ax, ref_raster_ax)
        # sim_raster_ax.get_shared_x_axes().join(sim_raster_ax, ref_bursting_ax)
        
        # remove x-axis labels from all plots except the bottom one
        sim_raster_ax.set_xlabel('')
        sim_bursting_ax.set_xlabel('')
        sim_mega_bursting_ax.set_xlabel('')
        ref_raster_ax.set_xlabel('')
        ref_bursting_ax.set_xlabel('')
        
        # remove all titles
        sim_mega_bursting_ax.set_title('')
        ref_mega_bursting_ax.set_title('')
        sim_raster_ax.set_title('')
        sim_bursting_ax.set_title('')
        ref_bursting_ax.set_title('')
        ref_raster_ax.set_title('')
        
        #re-write all y-axis labels
        sim_raster_ax.set_ylabel('Neuron GID')
        sim_bursting_ax.set_ylabel('Fire rate (Hz)')
        sim_mega_bursting_ax.set_ylabel('Fire rate (Hz)')
        ref_raster_ax.set_ylabel('Neuron ID')
        ref_bursting_ax.set_ylabel('Fire rate (Hz)')  
        sim_mega_bursting_ax.set_ylabel('Fire rate (Hz)')
        
        #align all y-axis labels to the left margin
        x_coord = -0.045
        sim_raster_ax.yaxis.set_label_coords(x_coord, 0.5)
        sim_bursting_ax.yaxis.set_label_coords(x_coord, 0.5)
        sim_mega_bursting_ax.yaxis.set_label_coords(x_coord, 0.5)
        ref_raster_ax.yaxis.set_label_coords(x_coord, 0.5)
        ref_bursting_ax.yaxis.set_label_coords(x_coord, 0.5)
        ref_mega_bursting_ax.yaxis.set_label_coords(x_coord, 0.5)
        
        # remove x-axis ticks from all plots except the bottom one
        sim_raster_ax.set_xticks([])
        sim_bursting_ax.set_xticks([])
        sim_mega_bursting_ax.set_xticks([])
        ref_raster_ax.set_xticks([])
        ref_bursting_ax.set_xticks([])
        
        #
        if subplot:
            #plt.tight_layout()
            #plt.close()
            #return [sim_raster_ax, sim_bursting_ax, ref_raster_ax, ref_bursting_ax]
            return ax_list
        
        #plt.tight_layout()
        fig.tight_layout()
        
        if DEBUG_MODE:
            print("Saving comparison plot for debugging")
            # Save local for debugging
            dev_dir = os.path.dirname(os.path.realpath(__file__))
            fig.savefig(os.path.join(dev_dir, '_comparison_plot.png'), dpi=300)
            # Save local for debugging
        
        # save wherever data is saved
        #sim_data_path = SIMULATION_RUN_PATH
        comparison_plot_path = sim_data_path.replace('_data', '_comparison_plot')
        #remove file type and replace with png
        #comparison_plot_path = comparison_plot_path.replace('.json', '.png')
        if '.json' in comparison_plot_path:
            comparison_plot_path = comparison_plot_path.replace('.json', '.png')
        elif '.pkl' in comparison_plot_path:
            comparison_plot_path = comparison_plot_path.replace('.pkl', '.png')
        fig.savefig(comparison_plot_path, dpi=300)
        print(f"Comparison plot saved to {comparison_plot_path}")
        
        #save as pdf
        comparison_plot_path = sim_data_path.replace('_data', '_comparison_plot')
        #comparison_plot_path = comparison_plot_path.replace('.json', '.pdf')
        if '.json' in comparison_plot_path:
            comparison_plot_path = comparison_plot_path.replace('.json', '.pdf')
        elif '.pkl' in comparison_plot_path:
            comparison_plot_path = comparison_plot_path.replace('.pkl', '.pdf')
        fig.savefig(comparison_plot_path)
        print(f"Comparison plot saved to {comparison_plot_path}")
        plt.close()
    plot_comparision_plot(trim_start=trim_start, **kwargs)
    # privileged_print("\tComparison plot saved.")
    # privileged_print(f'\tTime taken: {time()-start_comparison} seconds')
    print("\tComparison plot saved.")
    print(f'\tTime taken: {time()-start_comparison} seconds')

    # build comparison summary slide
    start_summary_slide = time()
    def build_comparision_summary_slide(sim_data_path, trim_start = None,):
        fig, ax_list = plt.subplots(6, 1, figsize=(16, 18))
        
        #plot comparison plot
        plot_comparision_plot(ax_list=ax_list, subplot=True, trim_start=trim_start, **kwargs)
        
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
        # ax_list[0].text(0.5, 1.1, 'Simulated', ha='center', va='center', transform=ax_list[0].transAxes, fontsize=12)
        # ax_list[3].text(0.5, 1.1, 'Reference', ha='center', va='center', transform=ax_list[2].transAxes, fontsize=12)
        ax_list[0].title.set_text('Simulated')
        ax_list[3].title.set_text('Reference')
        
        # add time(s) label to the bottom plot
        ax_list[-1].set_xlabel('Time (s)')
                
        #create space to the right of plots for text
        fig.subplots_adjust(right=0.6)
        
        #create some space at the bottom for one line of text
        fig.subplots_adjust(bottom=0.1)
        
        #add text
        spiking_summary = kwargs['network_metrics']['spiking_data']['spiking_summary_data']
        bursting_summary = kwargs['network_metrics']['bursting_data']['bursting_summary_data']
        mega_bursting_summary = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']
        simulated_spiking_data = kwargs['network_metrics']['simulated_data']
        
        #TODO: if meanBurstrate is not in the spiking summary, then the following will fail, add nan in that case for now
        if 'mean_Burst_Rate' not in bursting_summary:
            bursting_summary['mean_Burst_Rate'] = np.nan                
        
        text = (
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
            f"\n"
            
            f"Simulated bursting summary:\n"
            f"  Bursting summary:\n"        
            f"  - Number of units: {bursting_summary['NumUnits']}\n"
            f"  - Baseline: {bursting_summary['baseline']} Hz\n"
            f"  - Fanofactor: {bursting_summary['fano_factor']}\n"
            f"  - Number of bursts: {bursting_summary['Number_Bursts']}\n"
            f"  - Mean burst rate: {bursting_summary['mean_Burst_Rate']} bursts/second\n"     
            f"  - Mean burst peak: {bursting_summary['mean_Burst_Peak']} Hz\n"
            f"  - CoV burst peak: {bursting_summary['cov_Burst_Peak']}\n"
            f"  - Mean IBI: {bursting_summary['mean_IBI']} s\n"
            f"  - CoV IBI: {bursting_summary['cov_IBI']}\n"
            f"  - Mean within burst ISI: {bursting_summary['MeanWithinBurstISI']} s\n"
            f"  - CoV within burst ISI: {bursting_summary['CoVWithinBurstISI']}\n"
            f"  - Mean outside burst ISI: {bursting_summary['MeanOutsideBurstISI']} s\n"
            f"  - CoV outside burst ISI: {bursting_summary['CoVOutsideBurstISI']}\n"
            f"  - Mean network ISI: {bursting_summary['MeanNetworkISI']} s\n"
            f"  - CoV network ISI: {bursting_summary['CoVNetworkISI']}\n"
            f"\n"
            
            f"  Mega Bursting summary:\n"
            f"  - Number of units: {mega_bursting_summary['NumUnits']}\n"
            f"  - Baseline: {mega_bursting_summary['baseline']} Hz\n"
            f"  - Fanofactor: {mega_bursting_summary['fano_factor']}\n"
            f"  - Number of bursts: {mega_bursting_summary['Number_Bursts']}\n"
            f"  - Mean burst rate: {mega_bursting_summary['mean_Burst_Rate']} bursts/second\n"
            f"  - Mean burst peak: {mega_bursting_summary['mean_Burst_Peak']} Hz\n"
            f"  - CoV burst peak: {mega_bursting_summary['cov_Burst_Peak']}\n"
            f"  - Mean IBI: {mega_bursting_summary['mean_IBI']} s\n"
            f"  - CoV IBI: {mega_bursting_summary['cov_IBI']}\n"
            f"  - Mean within burst ISI: {mega_bursting_summary['MeanWithinBurstISI']} s\n"
            f"  - CoV within burst ISI: {mega_bursting_summary['CoVWithinBurstISI']}\n"
            f"  - Mean outside burst ISI: {mega_bursting_summary['MeanOutsideBurstISI']} s\n"
            f"  - CoV outside burst ISI: {mega_bursting_summary['CoVOutsideBurstISI']}\n"
            f"  - Mean network ISI: {mega_bursting_summary['MeanNetworkISI']} s\n"
            f"  - CoV network ISI: {mega_bursting_summary['CoVNetworkISI']}\n"
            f"\n"
        )
        
        #append reference metrics to text
        ref_data = np.load(
            reference_data_npy, 
            allow_pickle=True).item()
        ref_spiking_summary = ref_data['spiking_data']['spiking_summary_data']
        ref_bursting_summary = ref_data['bursting_data']['bursting_summary_data']
        ref_mega_bursting_summary = ref_data['mega_bursting_data']['bursting_summary_data']    
        text += (
            f"Reference spiking summary:\n"
            f"  - Mean firing rate: {ref_spiking_summary['MeanFireRate']} Hz\n"
            f"  - Coefficient of variation: {ref_spiking_summary['CoVFireRate']}\n"
            f"  - Mean ISI: {ref_spiking_summary['MeanISI']} s\n"
            f"  - Coefficient of variation of ISI: {ref_spiking_summary['CoV_ISI']}\n"
            f"\n"
            
            f"Reference bursting summary:\n"
            f"  - Number of units: {ref_bursting_summary['NumUnits']}\n"
            f"  - Baseline: {ref_bursting_summary['baseline']} Hz\n"
            f"  - Fanofactor: {ref_bursting_summary['fano_factor']}\n"
            f"  - Number of bursts: {ref_bursting_summary['Number_Bursts']}\n"
            f"  - Mean burst rate: {ref_bursting_summary['mean_Burst_Rate']} bursts/second\n"
            f"  - Mean burst peak: {ref_bursting_summary['mean_Burst_Peak']} Hz\n"
            f"  - CoV burst peak: {ref_bursting_summary['cov_Burst_Peak']}\n"
            f"  - Mean IBI: {ref_bursting_summary['mean_IBI']} s\n"
            f"  - CoV IBI: {ref_bursting_summary['cov_IBI']}\n"
            f"  - Mean within burst ISI: {ref_bursting_summary['MeanWithinBurstISI']} s\n"
            f"  - CoV within burst ISI: {ref_bursting_summary['CoVWithinBurstISI']}\n"
            f"  - Mean outside burst ISI: {ref_bursting_summary['MeanOutsideBurstISI']} s\n"
            f"  - CoV outside burst ISI: {ref_bursting_summary['CoVOutsideBurstISI']}\n"
            f"  - Mean network ISI: {ref_bursting_summary['MeanNetworkISI']} s\n"
            f"  - CoV network ISI: {ref_bursting_summary['CoVNetworkISI']}\n"
            f"\n"
            
            f"Reference mega bursting summary:\n"
            f"  - Number of units: {ref_mega_bursting_summary['NumUnits']}\n"
            f"  - Baseline: {ref_mega_bursting_summary['baseline']} Hz\n"
            f"  - Fanofactor: {ref_mega_bursting_summary['fano_factor']}\n"
            f"  - Number of bursts: {ref_mega_bursting_summary['Number_Bursts']}\n"
            f"  - Mean burst rate: {ref_mega_bursting_summary['mean_Burst_Rate']} bursts/second\n"
            f"  - Mean burst peak: {ref_mega_bursting_summary['mean_Burst_Peak']} Hz\n"
            f"  - CoV burst peak: {ref_mega_bursting_summary['cov_Burst_Peak']}\n"
            f"  - Mean IBI: {ref_mega_bursting_summary['mean_IBI']} s\n"
            f"  - CoV IBI: {ref_mega_bursting_summary['cov_IBI']}\n"
            f"  - Mean within burst ISI: {ref_mega_bursting_summary['MeanWithinBurstISI']} s\n"
            f"  - CoV within burst ISI: {ref_mega_bursting_summary['CoVWithinBurstISI']}\n"
            f"  - Mean outside burst ISI: {ref_mega_bursting_summary['MeanOutsideBurstISI']} s\n"
            f"  - CoV outside burst ISI: {ref_mega_bursting_summary['CoVOutsideBurstISI']}\n"
            f"  - Mean network ISI: {ref_mega_bursting_summary['MeanNetworkISI']} s\n"
            f"  - CoV network ISI: {ref_mega_bursting_summary['CoVNetworkISI']}\n"
            f"\n"
        )
        #add average fitness to text
        # text += (
        #     f"**Average fitness: {average_fitness}**"
        #     ) 
        
        #add text to the right of the plots
        fig.text(0.65, 0.5, text, ha='left', va='center', fontsize=11.5)
        
        # print average fitness in bold just above data path
        fig.text(0.5, 0.04, f"Average fitness: {average_fitness}", ha='center', va='center', fontsize=11.5, fontweight='bold')
        
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
        #comparison_summary_slide_path = comparison_summary_slide_path.replace('.json', '.png')
        if '.json' in comparison_summary_slide_path:
            comparison_summary_slide_path = comparison_summary_slide_path.replace('.json', '.png')
        elif '.pkl' in comparison_summary_slide_path:
            comparison_summary_slide_path = comparison_summary_slide_path.replace('.pkl', '.png')
        fig.savefig(comparison_summary_slide_path, dpi=300)
        #privileged_print(f"Comparison summary slide saved to {comparison_summary_slide_path}")
        print(f"Comparison summary slide saved to {comparison_summary_slide_path}")
        
        #save as pdf
        comparison_summary_slide_path = sim_data_path.replace('_data', '_comparison_summary_slide')
        #comparison_summary_slide_path = comparison_summary_slide_path.replace('.json', '.pdf')
        if '.json' in comparison_summary_slide_path:
            comparison_summary_slide_path = comparison_summary_slide_path.replace('.json', '.pdf')
        elif '.pkl' in comparison_summary_slide_path:
            comparison_summary_slide_path = comparison_summary_slide_path.replace('.pkl', '.pdf')
        fig.savefig(comparison_summary_slide_path)
        #privileged_print(f"Comparison summary slide saved to {comparison_summary_slide_path}")
        print(f"Comparison summary slide saved to {comparison_summary_slide_path}")
        plt.close()
        
        # return comparison_summary_slide_paths
        comparison_summary_slide_paths = [
            comparison_summary_slide_path.replace('.pdf', '.png'),
            comparison_summary_slide_path.replace('.png', '.pdf'),
        ]
        return comparison_summary_slide_paths
    comparison_summary_slide_paths = build_comparision_summary_slide(sim_data_path, trim_start = trim_start)
    # privileged_print("\tComparison summary slide saved.")
    # privileged_print(f'\tTime taken: {time()-start_summary_slide} seconds')
    print("\tComparison summary slide saved.")
    print(f'\tTime taken: {time()-start_summary_slide} seconds')
    
    return comparison_summary_slide_paths

def process_simulation_v2(kwargs):

    # unpack kwargs
    sim_data_path = kwargs.get('sim_data_path', None)
    conv_params = kwargs.get('conv_params', None)
    mega_params = kwargs.get('mega_params', None)
    fitnessFuncArgs = kwargs.get('fitnessFuncArgs', None)
    reference_data_npy = kwargs.get('reference_data_path', None)
    debug_mode = kwargs.get('debug_mode', False)
    
    # re-fit simulation of interest
    refit = False # TODO: need to refractor this. Shouldnt be calculating network metrics twice.
    if refit:
        start = time()
        print("Calculating average fitness...")
        average_fitness = fit_simulation(
            sim_data_path,
            conv_params=conv_params,
            mega_params=mega_params,
            fitnessFuncArgs=fitnessFuncArgs,
            ) 
        
        # TODO: There seems to be some inconsistency in FR calculated by network metrics vs NETPYNE - need to investigate this.
        print(f"Average fitness: {average_fitness}")
        print("Refit complete.")
        print(f"Time taken: {time()-start} seconds")
    else:
        average_fitness = None

    # re-plot simulation of interest - re-generate summary plot and all associated plots
    print("Replotting simulation of interest...")
    start = time()
    pkwargs = {
        'sim_data_path': sim_data_path,
        'average_fitness': average_fitness,
        'conv_params': conv_params,
        'mega_params': mega_params,
        'fitnessFuncArgs': fitnessFuncArgs,
        'reference_data_npy': reference_data_npy,
        'trim_start': 5,
        'DEBUG_MODE': debug_mode,
        }
    
    comparison_summary_slide_paths = plot_simulation_v2(pkwargs)
    print("Replotting complete.")
    print(f"Time taken: {time()-start} seconds")
    
    return comparison_summary_slide_paths

''' older functions '''
# older funcs ===================================================================================================
def process_simulation(
    sim_data_path, 
    REFERENCE_DATA_NPY,
    DEBUG_MODE=False,
    conv_params = None,
    mega_params = None,
    fitnessFuncArgs = None,
    ):

    # re-fit simulation of interest
    start = time()
    #print("Refitting simulation of interest...")
    print("Calculating average fitness...")
    average_fitness = fit_simulation(
        sim_data_path,
        conv_params=conv_params,
        mega_params=mega_params,
        fitnessFuncArgs=fitnessFuncArgs,
        ) 
    
    # TODO: There seems to be some inconsistency in FR calculated by network metrics vs NETPYNE - need to investigate this.
    print(f"Average fitness: {average_fitness}")
    print("Refit complete.")
    print(f"Time taken: {time()-start} seconds")

    # re-plot simulation of interest - re-generate summary plot and all associated plots
    print("Replotting simulation of interest...")
    start = time()
    comparison_summary_slide_paths = plot_simulation(
        sim_data_path,
        average_fitness,
        conv_params=conv_params,
        mega_params=mega_params,
        fitnessFuncArgs=fitnessFuncArgs,
        reference_data_npy=REFERENCE_DATA_NPY,
        trim_start=5,
        DEBUG_MODE=DEBUG_MODE,
        )
    # privileged_print("Replotting complete.")
    # privileged_print(f"Time taken: {time()-start} seconds")
    print("Replotting complete.")
    print(f"Time taken: {time()-start} seconds")
    
    return comparison_summary_slide_paths

def fit_simulation(
    sim_data_path,
    conv_params = None,
    mega_params = None,
    fitnessFuncArgs = None,
    ):
    
    # assertions
    assert sim_data_path is not None, "sim_data_path must be provided."
    assert conv_params is not None, "conv_params must be provided."
    assert mega_params is not None, "mega_params must be provided."
    assert fitnessFuncArgs is not None, "fitnessFuncArgs must be provided."
    
    # load sim data
    if not hasattr(sim, 'allSimData'):
        sim.load(sim_data_path)
    simData = sim.allSimData

    fitness_save_path = sim_data_path.replace('_data', '_fitness')
    if '.pkl' in fitness_save_path: fitness_save_path = fitness_save_path.replace('.pkl', '.json')
    kwargs = {
        'simConfig': sim.cfg,
        #'conv_params': convolution_params.conv_params,
        'conv_params': conv_params,
        'mega_params': mega_params,
        'popData': sim.net.allPops,
        'cellData': sim.net.allCells, #not actually used in the fitness calculation, but whatever
        'targets': fitnessFuncArgs['targets'],
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'features': fitnessFuncArgs['features'],
        'fitness_save_path': fitness_save_path,
        'break_deals': False #no need to do deal_breakers when doing refitting or sensitivity analysis
    }
    average_fitness = fitnessFunc(simData, mode='simulated', **kwargs)
    return average_fitness

def get_simulated_network_metrics_wrapper(
    conv_params = None,
    mega_params = None,
    sim_data_path = None,
    ):
    
    #convolution_params = import_module_from_path(CONVOLUTION_PARAMS)
    #from DIV21.src.conv_params import conv_params
    #from RBS_network_models.CDKL5.DIV21.src.conv_params import conv_params
    #from RBS_network_models.developing.CDKL5.DIV21.src.conv_params import conv_params
    assert conv_params is not None, "conv_params must be provided."
    assert sim_data_path is not None, "sim_data_path must be provided."    
    assert mega_params is not None, "mega_params must be provided."
    
    #get network metrics
    kwargs = {
        'simData': sim.allSimData,
        'simConfig': sim.cfg,
        #'conv_params': convolution_params.conv_params,
        'conv_params': conv_params,
        'mega_params': mega_params,
        'popData': sim.net.allPops,
        'cellData': sim.net.allCells, #not actually used in the fitness calculation, but whatever
    }
    #from DIV21.utils.fitness_helper import calculate_network_metrics
    #from RBS_network_models.developing.utils.fitness_helper import calculate_network_metrics
    #from RBS_network_models.utils.fitness_helper import calculate_network_metrics
    #error, kwargs = calculate_network_metrics(kwargs)
    
    network_metrics = get_simulated_network_activity_metrics(**kwargs)
    kwargs['network_metrics'] = network_metrics
    
    # save network metrics
    network_metrics_path = sim_data_path.replace('_data', '_network_metrics')
    if '.pkl' in network_metrics_path:
        network_metrics_path = network_metrics_path.replace('.pkl', '.npy')
    elif '.json' in network_metrics_path:
        network_metrics_path = network_metrics_path.replace('.json', '.npy')
    np.save(network_metrics_path, kwargs)
    # with open(network_metrics_path, 'w') as f:
    #     json.dump(kwargs, f, indent=4)
    return kwargs

def plot_simulation(sim_data_path,  
                    average_fitness,
                    conv_params=None,
                    mega_params=None,
                    fitnessFuncArgs=None,
                    reference_data_npy=None,
                    trim_start=0,
                    DEBUG_MODE=False,
                    ):
    
    # assertions
    assert sim_data_path is not None, "sim_data_path must be provided."
    assert conv_params is not None, "conv_params must be provided."
    assert fitnessFuncArgs is not None, "fitnessFuncArgs must be provided."
    assert reference_data_npy is not None, "reference_data_npy must be provided."
    assert mega_params is not None, "mega_params must be provided."
    
    # get network metrics
    start_network_metrics = time()
    kwargs = get_simulated_network_metrics_wrapper(
        conv_params=conv_params,
        mega_params=mega_params,
        sim_data_path=sim_data_path,        
    )
    #if error: return error
    # privileged_print("\tNetwork metrics calculated - kwargs dict created.")
    # privileged_print(f'\tTime taken: {time()-start_network_metrics} seconds')
    print("\tNetwork metrics calculated - kwargs dict created.")
    print(f'\tTime taken: {time()-start_network_metrics} seconds')

    #plot raster
    start_raster = time()
    def plot_simulated_raster_wrapper(ax = None, subplot=False, trim_start = None, **kwargs):
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
        
        # # if trim_start, trim first x seconds from the start of the simulation
        # if trim_start is not None and trim_start > 0 and trim_start < ax_raster.get_xlim()[1]:
        #     ax_raster.set_xlim(trim_start, ax_raster.get_xlim()[1])
        # elif trim_start is not None and trim_start > 0 and trim_start > ax_raster.get_xlim()[1]:
        #     modified_trim = ax_raster.get_xlim()[1]*0.1
        #     ax_raster.set_xlim(modified_trim, ax_raster.get_xlim()[1])
        #     print('boop')
        
        #plt.tight_layout()
        #fig.tight_layout()
        # print('tight bbox:')
        # print(ax_raster.get_tightbbox())
        # ax_raster.set_xlim(ax_raster.get_xlim()[0], ax_raster.get_xlim()[1])    
        
        #set the tightest possible xlim for the data in the raster plot
        # for some reason when raster plot is generated, left side is slightly negative and right side is slightly more positive than the actual data.
        # so using xlim doesnt work here.
        # #x_data = ax_raster.lines[0].get_xdata()
        # true_max_x = 0
        # true_min_x = 0
        # for line in ax_raster.lines:
        #     x_data = line.get_xdata()
        #     max_x = max(x_data)
        #     min_x = min(x_data)
        #     if max_x > true_max_x:
        #         true_max_x = max_x
        #     if min_x < true_min_x:
        #         true_min_x = min_x
        # ax_raster.set_xlim(true_min_x, true_max_x)
        
        # ##set the tightest possible ylim for the data in the raster plot
        # true_max_y = 0
        # true_min_y = 0
        # for line in ax_raster.lines:
        #     y_data = line.get_ydata()
        #     max_y = max(y_data)
        #     min_y = min(y_data)
        #     if max_y > true_max_y:
        #         true_max_y = max_y
        #     if min_y < true_min_y:
        #         true_min_y = min_y
        # ax_raster.set_ylim(true_min_y, true_max_y)
        # #print(f'true min y: {min(y_data)}')
                    
        
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
        if '.json' in raster_plot_path:
            raster_plot_path = raster_plot_path.replace('.json', '.png')
        elif '.pkl' in raster_plot_path:
            raster_plot_path = raster_plot_path.replace('.pkl', '.png')
        #raster_plot_path = raster_plot_path.replace('.json', '.png')
        plt.savefig(raster_plot_path, dpi=300)
        print(f"Raster plot saved to {raster_plot_path}")
        
        #save as pdf
        raster_plot_path = sim_data_path.replace('_data', '_raster_plot')
        #raster_plot_path = raster_plot_path.replace('.json', '.pdf')
        if '.json' in raster_plot_path:
            raster_plot_path = raster_plot_path.replace('.json', '.pdf')
        elif '.pkl' in raster_plot_path:
            raster_plot_path = raster_plot_path.replace('.pkl', '.pdf')
        plt.savefig(raster_plot_path)
        print(f"Raster plot saved to {raster_plot_path}")
        plt.close()
        
        raster_plots_paths = [
            raster_plot_path,
            raster_plot_path.replace('.pdf', '.png'),
        ]
        
        return raster_plots_paths
    raster_plot_paths = plot_simulated_raster_wrapper(trim_start = trim_start, **kwargs)
    # privileged_print("\tIndividual raster plots saved.")
    # privileged_print(f'\tTime taken: {time()-start_raster} seconds')
    print("\tIndividual raster plots saved.")
    print(f'\tTime taken: {time()-start_raster} seconds')

    #plot bursting summary
    start_bursting = time()
    def plot_simulated_bursting_wrapper(ax = None, subplot=False, trim_start = None, **kwargs):
        if ax is None:
            #fig, new_ax = plt.subplots(1, 1)
            fig, new_ax = plt.subplots(2, 1, figsize=(16, 9))
            fig.set_size_inches(16, 4.5)
        else:
            new_ax = ax
            subplot = True #if ax is passed in, then we are plotting on a subplot
        
        #
        # conv_params = kwargs['conv_params']
        # SpikeTimes = kwargs['network_metrics']['spiking_data']['spiking_times_by_unit']
        #from DIV21.utils.fitness_helper import plot_network_activity_aw
        #from RBS_network_models.developing.utils.analysis_helper import plot_network_activity_aw
        #from RBS_network_models.network_analysis import plot_network_activity_aw
        #bursting_ax, _ = plot_network_activity_aw(new_ax, SpikeTimes, **conv_params) #TODO need to make sure this function agrees with mandar. would be best if we shared a function here.
        # bursting_ax = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['ax']
        # mega_ax = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['ax']
        if 'ax' in kwargs['network_metrics']['bursting_data']:
            bursting_ax = kwargs['network_metrics']['bursting_data']['ax']
        else:
            bursting_ax = None
        if 'ax' in kwargs['network_metrics']['mega_bursting_data']:
            mega_ax = kwargs['network_metrics']['mega_bursting_data']['ax']
        else:
            mega_ax = None
        
        # # HACK
        # mega_conv_params = kwargs['conv_params'].copy()
        # mega_conv_params['binSize'] *= 5
        # mega_conv_params['gaussianSigma'] *= 15
        #mega_ax, _ = plot_network_activity_aw(new_ax, SpikeTimes, **mega_conv_params) #TODO need to make sure this function agrees with mandar. would be best if we shared a function here.
        
        from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_network_bursting_experimental
        #new_ax = plot_network_bursting_experimental(new_ax, bursting_ax, mega_ax=mega_ax)
        new_ax[0] = plot_network_bursting_experimental(new_ax[0], bursting_ax) #TODO: rename this func. It works for both experimental and simulated bursting
        new_ax[1] = plot_network_bursting_experimental(new_ax[1], mega_ax, mode='mega')
        
        # from RBS_network_models.network_analysis import plot_network_summary
        # new_ax = plot_network_summary(new_ax, bursting_ax, mega_ax=mega_ax)            
        
        # # if trim_start, trim first x seconds from the start of the simulation
        # if trim_start is not None and trim_start > 0 and trim_start < new_ax.get_xlim()[1]:
        #     new_ax.set_xlim(trim_start, new_ax.get_xlim()[1])
        # elif trim_start is not None and trim_start > 0 and trim_start > new_ax.get_xlim()[1]:
        #     modified_trim = new_ax.get_xlim()[1]*0.1
        #     new_ax.set_xlim(modified_trim, new_ax.get_xlim()[1])    
        
        # new_ax.set_title('Bursting summary')
        # new_ax.set_xlabel('Time (s)')
        # new_ax.set_ylabel('Fire rate (Hz)')
        new_ax[0].set_title('Bursting summary')
        #new_ax[0].set_xlabel('Time (s)')
        new_ax[0].set_ylabel('Fire rate (Hz)')
        new_ax[1].set_title('Mega bursting summary')
        new_ax[1].set_xlabel('Time (s)')
        new_ax[1].set_ylabel('Fire rate (Hz)')
        
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
        #bursting_plot_path = bursting_plot_path.replace('.json', '.png')
        if '.json' in bursting_plot_path:
            bursting_plot_path = bursting_plot_path.replace('.json', '.png')
        elif '.pkl' in bursting_plot_path:
            bursting_plot_path = bursting_plot_path.replace('.pkl', '.png')
        plt.savefig(bursting_plot_path, dpi=300)
        print(f"Bursting plot saved to {bursting_plot_path}")
        
        #save as pdf
        bursting_plot_path = sim_data_path.replace('_data', '_bursting_plot')
        #bursting_plot_path = bursting_plot_path.replace('.json', '.pdf')
        if '.json' in bursting_plot_path:
            bursting_plot_path = bursting_plot_path.replace('.json', '.pdf')
        elif '.pkl' in bursting_plot_path:
            bursting_plot_path = bursting_plot_path.replace('.pkl', '.pdf')
        plt.savefig(bursting_plot_path)
        print(f"Bursting plot saved to {bursting_plot_path}")
        plt.close()
        
        bursting_plot_paths = [
            bursting_plot_path,
            bursting_plot_path.replace('.pdf', '.png'),
        ]
        
        return bursting_plot_paths    
    bursting_plot_paths = plot_simulated_bursting_wrapper(trim_start = trim_start, **kwargs)
    # privileged_print("\tIndividual bursting plots saved.")
    # privileged_print(f'\tTime taken: {time()-start_bursting} seconds')
    print("\tIndividual bursting plots saved.")
    print(f'\tTime taken: {time()-start_bursting} seconds')

    # combine plots into a single summary plot
    start_summary = time()
    def plot_simulation_summary(trim_start = None, **kwargs):
        #fig, ax = plt.subplots(2, 1, figsize=(16, 9))
        # fig, ax = plt.subplots(2, 1, figsize=(16, 9))
        # fig2, ax[1] = plt.subplot(2, 1, 2)
        fig, ax = plt.subplots(3, 1, figsize=(16, 9))
        raster_plot_ax = ax[0]
        bursting_plot_ax = ax[1:]

        subplot = True
        raster_plot_ax = plot_simulated_raster_wrapper(ax=raster_plot_ax, subplot=subplot, 
                                                       #trim_start = trim_start, 
                                                       **kwargs)
        bursting_plot_ax = plot_simulated_bursting_wrapper(ax=bursting_plot_ax, subplot=subplot, 
                                                           #trim_start = trim_start, 
                                                           **kwargs)
        
        #make both plots share the same x-axis
        #raster_plot_ax.get_shared_x_axes().join(raster_plot_ax, bursting_plot_ax[0], bursting_plot_ax[1])
        bursting_plot_ax[0].set_xlim(raster_plot_ax.get_xlim())
        bursting_plot_ax[1].set_xlim(raster_plot_ax.get_xlim())
        print(f"simulation summary plot xlims set to {raster_plot_ax.get_xlim()[0]} to {raster_plot_ax.get_xlim()[1]}")
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
        #summary_plot_path = summary_plot_path.replace('.json', '.png')
        if '.json' in summary_plot_path:
            summary_plot_path = summary_plot_path.replace('.json', '.png')
        elif '.pkl' in summary_plot_path:
            summary_plot_path = summary_plot_path.replace('.pkl', '.png')
        plt.savefig(summary_plot_path, dpi=300)
        print(f"Summary plot saved to {summary_plot_path}")
        
        #save as pdf
        summary_plot_path = sim_data_path.replace('_data', '_summary_plot')
        #summary_plot_path = summary_plot_path.replace('.json', '.pdf')
        if '.json' in summary_plot_path:
            summary_plot_path = summary_plot_path.replace('.json', '.pdf')
        elif '.pkl' in summary_plot_path:
            summary_plot_path = summary_plot_path.replace('.pkl', '.pdf')
        plt.savefig(summary_plot_path)  
        print(f"Summary plot saved to {summary_plot_path}")
        plt.close()
        
        summary_plot_paths = [
            summary_plot_path,
            summary_plot_path.replace('.pdf', '.png'),
        ]    
        return summary_plot_paths
    plot_simulation_summary(trim_start=trim_start, **kwargs)
    # privileged_print("\tSimulation summary plot saved.")
    # privileged_print(f'\tTime taken: {time()-start_summary} seconds')
    print("\tSimulation summary plot saved.")
    print(f'\tTime taken: {time()-start_summary} seconds')

    # comparison plot against reference data snippet
    start_comparison = time()
    #activate_print()
    def plot_comparision_plot(ax_list=None, subplot=False, trim_start = None, **kwargs):
        #activate_print() #for debugging
        #fig, ax = plt.subplots(4, 1, figsize=(16, 9))
        if ax_list is None:
            #fig, ax_list = plt.subplots(4, 1, figsize=(16, 9))
            fig, ax_list = plt.subplots(6, 1, figsize=(16, 9))
            #sim_raster_ax, sim_bursting_ax, ref_raster_ax, ref_bursting_ax = ax_list
            sim_raster_ax, sim_bursting_ax, sim_mega_bursting_ax, ref_raster_ax, ref_bursting_ax, ref_mega_bursting_ax = ax_list
        else:
            #assert that there be 4 axes in the ax list
            #assert len(ax_list) == 4, "There must be 4 axes in the ax_list."
            assert len(ax_list) == 6, "There must be 6 axes in the ax_list."
            #sim_raster_ax, sim_bursting_ax, ref_raster_ax, ref_bursting_ax = ax_list
            sim_raster_ax, sim_bursting_ax, sim_mega_bursting_ax, ref_raster_ax, ref_bursting_ax, ref_mega_bursting_ax = ax_list
            subplot = True
        
        #plot simulated raster
        sim_raster_ax = plot_simulated_raster_wrapper(ax=sim_raster_ax, subplot=True, trim_start=trim_start, **kwargs)
        
        #plot simulated bursting
        sim_bursting_axes = ax_list[1:3]
        #sim_bursting_ax = plot_simulated_bursting_wrapper(ax=sim_bursting_ax, subplot=True, trim_start=trim_start, **kwargs)
        sim_bursting_axes = plot_simulated_bursting_wrapper(ax=sim_bursting_axes, subplot=True, trim_start=trim_start, **kwargs)
        
        # plot reference raster
        def plot_reference_raster_wrapper(ax=None, subplot=False, sim_data_length=None, **kwargs):
            #load npy ref data
            print(ax)
            print(subplot)
            #print(kwargs)
            
            #load npy ref data
            ref_data = np.load(
                #REFERENCE_DATA_NPY,
                reference_data_npy, 
                allow_pickle=True
                ).item()
            network_metrics = ref_data
            
            #HACK patch for different ref data format            
            try: spiking_data_by_unit = network_metrics['spiking_data']['spiking_data_by_unit'].copy()
            except: spiking_data_by_unit = network_metrics['spiking_data']['spiking_metrics_by_unit'].copy()
            
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
        def plot_reference_bursting_wrapper(ax=None, subplot=False, sim_data_length=None, trim_start = None, **kwargs):
            #ref_data = np.load(REFERENCE_DATA_NPY, allow_pickle=True).item()
            ref_data = np.load(
                #REFERENCE_DATA_NPY,
                reference_data_npy, 
                allow_pickle=True).item()
            network_metrics = ref_data
            conv_params = kwargs['conv_params']
            SpikeTimes = network_metrics['spiking_data']['spiking_times_by_unit']
            if ax is None:
                #fig, new_ax = plt.subplots(1, 1)
                fig, new_ax = plt.subplots(2, 1)
                fig.set_size_inches(16, 4.5)
            else:
                new_ax = ax
                subplot = True
                
            if sim_data_length is not None:
                for unit_id in SpikeTimes.keys():
                    spike_times = SpikeTimes[unit_id]
                    spike_times = spike_times[spike_times < sim_data_length]
                    SpikeTimes[unit_id] = spike_times
            
            #from DIV21.utils.fitness_helper import plot_network_activity_aw
            #from RBS_network_models.developing.utils.analysis_helper import plot_network_activity_aw
            #from RBS_network_models.network_analysis import plot_network_activity_aw
            #new_ax[0], _ = plot_network_activity_aw(new_ax, SpikeTimes, **conv_params) #TODO need to make sure this function agrees with mandar. would be best if we shared a function here.
            #new_ax[1], _ = plot_network_activity_aw(new_ax, SpikeTimes, **conv_params) #TODO need to make sure this function agrees with mandar. would be best if we shared a function here.
            # new_ax.set_title('Bursting summary')
            # new_ax.set_xlabel('Time (s)')
            # new_ax.set_ylabel('Fire rate (Hz)')
            
            
            bursting_ax = network_metrics['bursting_data']['bursting_summary_data']['ax']
            mega_ax = network_metrics['mega_bursting_data']['bursting_summary_data']['ax']
            
            #plot
            from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_network_bursting_experimental
            new_ax[0] = plot_network_bursting_experimental(new_ax[0], bursting_ax) #TODO: rename this func. It works for both experimental and simulated bursting
            new_ax[1] = plot_network_bursting_experimental(new_ax[1], mega_ax, mode='mega')
            
            #adjust axes titles and labels
            new_ax[0].set_title('Bursting summary')
            #new_ax[0].set_xlabel('Time (s)')
            new_ax[0].set_ylabel('Fire rate (Hz)')
            new_ax[1].set_title('Mega bursting summary')
            new_ax[1].set_xlabel('Time (s)')
            new_ax[1].set_ylabel('Fire rate (Hz)')           
            
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
        ref_bursting_axes = ax_list[4:6]
        ref_bursting_axes = plot_reference_bursting_wrapper(ax=ref_bursting_axes, subplot=True, sim_data_length=sim_data_length, trim_start = trim_start, **kwargs)
        
        # apply trim_start if trim_start is not None and trim_start > 0:
        # only apply to sim raster plot - allow x axis alignmnet code below to handle the rest
        if trim_start is not None and trim_start > 0:
            print("Trimming start of simulated data")
            # ensure trim_start is less than or equal to 10% of the x-axis range, else just trim first 10% of the x-axis range
            sim_duration = sim_raster_ax.get_xlim()[1] - sim_raster_ax.get_xlim()[0]
            if trim_start > sim_duration * 0.1:
                trim_start = sim_duration * 0.1
            sim_raster_ax.set_xlim(trim_start, sim_raster_ax.get_xlim()[1])
            print(f'Trimmed simulated raster plot range: {sim_raster_ax.get_xlim()[0]} to {sim_raster_ax.get_xlim()[1]}')
            
        
        # ensure xaxis of refernce plots matches simulated raster
        #print("Setting xlim of reference plots to match simulated raster")
        ref_raster_ax.set_xlim(sim_raster_ax.get_xlim())
        ref_bursting_ax.set_xlim(sim_raster_ax.get_xlim())
        ref_mega_bursting_ax.set_xlim(sim_raster_ax.get_xlim())
        sim_bursting_ax.set_xlim(sim_raster_ax.get_xlim())
        sim_mega_bursting_ax.set_xlim(sim_raster_ax.get_xlim())
        
        
        # Ensure y-axis is the same for bursting plots
        #print("Adjusting y-axis limits based on x-axis limits")
        def adjust_ylim_based_on_xlim(ax):
            x_data = ax.lines[0].get_xdata()
            y_data = ax.lines[0].get_ydata()
            xlim = ax.get_xlim()
            filtered_y_data = [y for x, y in zip(x_data, y_data) if xlim[0] <= x <= xlim[1]]
            if filtered_y_data:
                ax.set_ylim(min(filtered_y_data) * 0.8, max(filtered_y_data) * 1.2)

        adjust_ylim_based_on_xlim(sim_bursting_ax)
        adjust_ylim_based_on_xlim(ref_bursting_ax)  
        
        
        # Calculate the combined y-axis limits
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
        sim_mega_bursting_ax.set_ylim(combined_ylim)
        sim_mega_bursting_ax.set_ylim(combined_ylim)
        
        # # Make all four plots share the same x-axis as simulated raster
        # sim_raster_ax.get_shared_x_axes().join(sim_raster_ax, sim_bursting_ax)
        # sim_raster_ax.get_shared_x_axes().join(sim_raster_ax, ref_raster_ax)
        # sim_raster_ax.get_shared_x_axes().join(sim_raster_ax, ref_bursting_ax)
        
        # remove x-axis labels from all plots except the bottom one
        sim_raster_ax.set_xlabel('')
        sim_bursting_ax.set_xlabel('')
        sim_mega_bursting_ax.set_xlabel('')
        ref_raster_ax.set_xlabel('')
        ref_bursting_ax.set_xlabel('')
        
        # remove all titles
        sim_mega_bursting_ax.set_title('')
        ref_mega_bursting_ax.set_title('')
        sim_raster_ax.set_title('')
        sim_bursting_ax.set_title('')
        ref_bursting_ax.set_title('')
        ref_raster_ax.set_title('')
        
        #re-write all y-axis labels
        sim_raster_ax.set_ylabel('Neuron GID')
        sim_bursting_ax.set_ylabel('Fire rate (Hz)')
        sim_mega_bursting_ax.set_ylabel('Fire rate (Hz)')
        ref_raster_ax.set_ylabel('Neuron ID')
        ref_bursting_ax.set_ylabel('Fire rate (Hz)')  
        sim_mega_bursting_ax.set_ylabel('Fire rate (Hz)')
        
        #align all y-axis labels to the left margin
        x_coord = -0.045
        sim_raster_ax.yaxis.set_label_coords(x_coord, 0.5)
        sim_bursting_ax.yaxis.set_label_coords(x_coord, 0.5)
        sim_mega_bursting_ax.yaxis.set_label_coords(x_coord, 0.5)
        ref_raster_ax.yaxis.set_label_coords(x_coord, 0.5)
        ref_bursting_ax.yaxis.set_label_coords(x_coord, 0.5)
        ref_mega_bursting_ax.yaxis.set_label_coords(x_coord, 0.5)
        
        # remove x-axis ticks from all plots except the bottom one
        sim_raster_ax.set_xticks([])
        sim_bursting_ax.set_xticks([])
        sim_mega_bursting_ax.set_xticks([])
        ref_raster_ax.set_xticks([])
        ref_bursting_ax.set_xticks([])
        
        #
        if subplot:
            #plt.tight_layout()
            #plt.close()
            #return [sim_raster_ax, sim_bursting_ax, ref_raster_ax, ref_bursting_ax]
            return ax_list
        
        #plt.tight_layout()
        fig.tight_layout()
        
        if DEBUG_MODE:
            print("Saving comparison plot for debugging")
            # Save local for debugging
            dev_dir = os.path.dirname(os.path.realpath(__file__))
            fig.savefig(os.path.join(dev_dir, '_comparison_plot.png'), dpi=300)
            # Save local for debugging
        
        # save wherever data is saved
        #sim_data_path = SIMULATION_RUN_PATH
        comparison_plot_path = sim_data_path.replace('_data', '_comparison_plot')
        #remove file type and replace with png
        #comparison_plot_path = comparison_plot_path.replace('.json', '.png')
        if '.json' in comparison_plot_path:
            comparison_plot_path = comparison_plot_path.replace('.json', '.png')
        elif '.pkl' in comparison_plot_path:
            comparison_plot_path = comparison_plot_path.replace('.pkl', '.png')
        fig.savefig(comparison_plot_path, dpi=300)
        print(f"Comparison plot saved to {comparison_plot_path}")
        
        #save as pdf
        comparison_plot_path = sim_data_path.replace('_data', '_comparison_plot')
        #comparison_plot_path = comparison_plot_path.replace('.json', '.pdf')
        if '.json' in comparison_plot_path:
            comparison_plot_path = comparison_plot_path.replace('.json', '.pdf')
        elif '.pkl' in comparison_plot_path:
            comparison_plot_path = comparison_plot_path.replace('.pkl', '.pdf')
        fig.savefig(comparison_plot_path)
        print(f"Comparison plot saved to {comparison_plot_path}")
        plt.close()
    plot_comparision_plot(trim_start=trim_start, **kwargs)
    # privileged_print("\tComparison plot saved.")
    # privileged_print(f'\tTime taken: {time()-start_comparison} seconds')
    print("\tComparison plot saved.")
    print(f'\tTime taken: {time()-start_comparison} seconds')

    # build comparison summary slide
    start_summary_slide = time()
    def build_comparision_summary_slide(sim_data_path, trim_start = None,):
        fig, ax_list = plt.subplots(6, 1, figsize=(16, 18))
        
        #plot comparison plot
        plot_comparision_plot(ax_list=ax_list, subplot=True, trim_start=trim_start, **kwargs)
        
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
        # ax_list[0].text(0.5, 1.1, 'Simulated', ha='center', va='center', transform=ax_list[0].transAxes, fontsize=12)
        # ax_list[3].text(0.5, 1.1, 'Reference', ha='center', va='center', transform=ax_list[2].transAxes, fontsize=12)
        ax_list[0].title.set_text('Simulated')
        ax_list[3].title.set_text('Reference')
        
        # add time(s) label to the bottom plot
        ax_list[-1].set_xlabel('Time (s)')
                
        #create space to the right of plots for text
        fig.subplots_adjust(right=0.6)
        
        #create some space at the bottom for one line of text
        fig.subplots_adjust(bottom=0.1)
        
        #add text
        spiking_summary = kwargs['network_metrics']['spiking_data']['spiking_summary_data']
        bursting_summary = kwargs['network_metrics']['bursting_data']['bursting_summary_data']
        mega_bursting_summary = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']
        simulated_spiking_data = kwargs['network_metrics']['simulated_data']
        
        #TODO: if meanBurstrate is not in the spiking summary, then the following will fail, add nan in that case for now
        if 'mean_Burst_Rate' not in bursting_summary:
            bursting_summary['mean_Burst_Rate'] = np.nan                
        
        text = (
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
            f"\n"
            
            f"Simulated bursting summary:\n"
            f"  Bursting summary:\n"        
            f"  - Number of units: {bursting_summary['NumUnits']}\n"
            f"  - Baseline: {bursting_summary['baseline']} Hz\n"
            f"  - Fanofactor: {bursting_summary['fano_factor']}\n"
            f"  - Number of bursts: {bursting_summary['Number_Bursts']}\n"
            f"  - Mean burst rate: {bursting_summary['mean_Burst_Rate']} bursts/second\n"     
            f"  - Mean burst peak: {bursting_summary['mean_Burst_Peak']} Hz\n"
            f"  - CoV burst peak: {bursting_summary['cov_Burst_Peak']}\n"
            f"  - Mean IBI: {bursting_summary['mean_IBI']} s\n"
            f"  - CoV IBI: {bursting_summary['cov_IBI']}\n"
            f"  - Mean within burst ISI: {bursting_summary['MeanWithinBurstISI']} s\n"
            f"  - CoV within burst ISI: {bursting_summary['CoVWithinBurstISI']}\n"
            f"  - Mean outside burst ISI: {bursting_summary['MeanOutsideBurstISI']} s\n"
            f"  - CoV outside burst ISI: {bursting_summary['CoVOutsideBurstISI']}\n"
            f"  - Mean network ISI: {bursting_summary['MeanNetworkISI']} s\n"
            f"  - CoV network ISI: {bursting_summary['CoVNetworkISI']}\n"
            f"\n"
            
            f"  Mega Bursting summary:\n"
            f"  - Number of units: {mega_bursting_summary['NumUnits']}\n"
            f"  - Baseline: {mega_bursting_summary['baseline']} Hz\n"
            f"  - Fanofactor: {mega_bursting_summary['fano_factor']}\n"
            f"  - Number of bursts: {mega_bursting_summary['Number_Bursts']}\n"
            f"  - Mean burst rate: {mega_bursting_summary['mean_Burst_Rate']} bursts/second\n"
            f"  - Mean burst peak: {mega_bursting_summary['mean_Burst_Peak']} Hz\n"
            f"  - CoV burst peak: {mega_bursting_summary['cov_Burst_Peak']}\n"
            f"  - Mean IBI: {mega_bursting_summary['mean_IBI']} s\n"
            f"  - CoV IBI: {mega_bursting_summary['cov_IBI']}\n"
            f"  - Mean within burst ISI: {mega_bursting_summary['MeanWithinBurstISI']} s\n"
            f"  - CoV within burst ISI: {mega_bursting_summary['CoVWithinBurstISI']}\n"
            f"  - Mean outside burst ISI: {mega_bursting_summary['MeanOutsideBurstISI']} s\n"
            f"  - CoV outside burst ISI: {mega_bursting_summary['CoVOutsideBurstISI']}\n"
            f"  - Mean network ISI: {mega_bursting_summary['MeanNetworkISI']} s\n"
            f"  - CoV network ISI: {mega_bursting_summary['CoVNetworkISI']}\n"
            f"\n"
        )
        
        #append reference metrics to text
        ref_data = np.load(
            reference_data_npy, 
            allow_pickle=True).item()
        ref_spiking_summary = ref_data['spiking_data']['spiking_summary_data']
        ref_bursting_summary = ref_data['bursting_data']['bursting_summary_data']
        ref_mega_bursting_summary = ref_data['mega_bursting_data']['bursting_summary_data']    
        text += (
            f"Reference spiking summary:\n"
            f"  - Mean firing rate: {ref_spiking_summary['MeanFireRate']} Hz\n"
            f"  - Coefficient of variation: {ref_spiking_summary['CoVFireRate']}\n"
            f"  - Mean ISI: {ref_spiking_summary['MeanISI']} s\n"
            f"  - Coefficient of variation of ISI: {ref_spiking_summary['CoV_ISI']}\n"
            f"\n"
            
            f"Reference bursting summary:\n"
            f"  - Number of units: {ref_bursting_summary['NumUnits']}\n"
            f"  - Baseline: {ref_bursting_summary['baseline']} Hz\n"
            f"  - Fanofactor: {ref_bursting_summary['fano_factor']}\n"
            f"  - Number of bursts: {ref_bursting_summary['Number_Bursts']}\n"
            f"  - Mean burst rate: {ref_bursting_summary['mean_Burst_Rate']} bursts/second\n"
            f"  - Mean burst peak: {ref_bursting_summary['mean_Burst_Peak']} Hz\n"
            f"  - CoV burst peak: {ref_bursting_summary['cov_Burst_Peak']}\n"
            f"  - Mean IBI: {ref_bursting_summary['mean_IBI']} s\n"
            f"  - CoV IBI: {ref_bursting_summary['cov_IBI']}\n"
            f"  - Mean within burst ISI: {ref_bursting_summary['MeanWithinBurstISI']} s\n"
            f"  - CoV within burst ISI: {ref_bursting_summary['CoVWithinBurstISI']}\n"
            f"  - Mean outside burst ISI: {ref_bursting_summary['MeanOutsideBurstISI']} s\n"
            f"  - CoV outside burst ISI: {ref_bursting_summary['CoVOutsideBurstISI']}\n"
            f"  - Mean network ISI: {ref_bursting_summary['MeanNetworkISI']} s\n"
            f"  - CoV network ISI: {ref_bursting_summary['CoVNetworkISI']}\n"
            f"\n"
            
            f"Reference mega bursting summary:\n"
            f"  - Number of units: {ref_mega_bursting_summary['NumUnits']}\n"
            f"  - Baseline: {ref_mega_bursting_summary['baseline']} Hz\n"
            f"  - Fanofactor: {ref_mega_bursting_summary['fano_factor']}\n"
            f"  - Number of bursts: {ref_mega_bursting_summary['Number_Bursts']}\n"
            f"  - Mean burst rate: {ref_mega_bursting_summary['mean_Burst_Rate']} bursts/second\n"
            f"  - Mean burst peak: {ref_mega_bursting_summary['mean_Burst_Peak']} Hz\n"
            f"  - CoV burst peak: {ref_mega_bursting_summary['cov_Burst_Peak']}\n"
            f"  - Mean IBI: {ref_mega_bursting_summary['mean_IBI']} s\n"
            f"  - CoV IBI: {ref_mega_bursting_summary['cov_IBI']}\n"
            f"  - Mean within burst ISI: {ref_mega_bursting_summary['MeanWithinBurstISI']} s\n"
            f"  - CoV within burst ISI: {ref_mega_bursting_summary['CoVWithinBurstISI']}\n"
            f"  - Mean outside burst ISI: {ref_mega_bursting_summary['MeanOutsideBurstISI']} s\n"
            f"  - CoV outside burst ISI: {ref_mega_bursting_summary['CoVOutsideBurstISI']}\n"
            f"  - Mean network ISI: {ref_mega_bursting_summary['MeanNetworkISI']} s\n"
            f"  - CoV network ISI: {ref_mega_bursting_summary['CoVNetworkISI']}\n"
            f"\n"
        )
        #add average fitness to text
        # text += (
        #     f"**Average fitness: {average_fitness}**"
        #     ) 
        
        #add text to the right of the plots
        fig.text(0.65, 0.5, text, ha='left', va='center', fontsize=11.5)
        
        # print average fitness in bold just above data path
        fig.text(0.5, 0.04, f"Average fitness: {average_fitness}", ha='center', va='center', fontsize=11.5, fontweight='bold')
        
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
        #comparison_summary_slide_path = comparison_summary_slide_path.replace('.json', '.png')
        if '.json' in comparison_summary_slide_path:
            comparison_summary_slide_path = comparison_summary_slide_path.replace('.json', '.png')
        elif '.pkl' in comparison_summary_slide_path:
            comparison_summary_slide_path = comparison_summary_slide_path.replace('.pkl', '.png')
        fig.savefig(comparison_summary_slide_path, dpi=300)
        #privileged_print(f"Comparison summary slide saved to {comparison_summary_slide_path}")
        print(f"Comparison summary slide saved to {comparison_summary_slide_path}")
        
        #save as pdf
        comparison_summary_slide_path = sim_data_path.replace('_data', '_comparison_summary_slide')
        #comparison_summary_slide_path = comparison_summary_slide_path.replace('.json', '.pdf')
        if '.json' in comparison_summary_slide_path:
            comparison_summary_slide_path = comparison_summary_slide_path.replace('.json', '.pdf')
        elif '.pkl' in comparison_summary_slide_path:
            comparison_summary_slide_path = comparison_summary_slide_path.replace('.pkl', '.pdf')
        fig.savefig(comparison_summary_slide_path)
        #privileged_print(f"Comparison summary slide saved to {comparison_summary_slide_path}")
        print(f"Comparison summary slide saved to {comparison_summary_slide_path}")
        plt.close()
        
        # return comparison_summary_slide_paths
        comparison_summary_slide_paths = [
            comparison_summary_slide_path.replace('.pdf', '.png'),
            comparison_summary_slide_path.replace('.png', '.pdf'),
        ]
        return comparison_summary_slide_paths
    comparison_summary_slide_paths = build_comparision_summary_slide(sim_data_path, trim_start = trim_start)
    # privileged_print("\tComparison summary slide saved.")
    # privileged_print(f'\tTime taken: {time()-start_summary_slide} seconds')
    print("\tComparison summary slide saved.")
    print(f'\tTime taken: {time()-start_summary_slide} seconds')
    
    return comparison_summary_slide_paths

''' idk '''
def calculate_simulation_fit(
    sim_data_path, 
    #target_script_path, 
    #saved_files=None
    ):   
    
    #sim.clearAll()
    #sim.load(sim_data_path)
    try:
        simData = sim.allSimData
    except:
        print("sim.allSimData failed. Trying sim.load(sim_data_path)...")
        sim.load(sim_data_path)
        simData = sim.allSimData
        # print e and i rates for debugging
        #print(simData['popRates'])
        # privileged_print(f'path: {sim_data_path}')
        print(f'path: {sim_data_path}')
        # privileged_print(f"Excitatory firing rate: {simData['popRates']['E']}")
        # privileged_print(f"Inhibitory firing rate: {simData['popRates']['I']}")
        print(f"Excitatory firing rate: {simData['popRates']['E']}")
        print(f"Inhibitory firing rate: {simData['popRates']['I']}")
    #from DIV21.src.conv_params import conv_params
    #convolution_params = import_module_from_path(CONVOLUTION_PARAMS)
    #script_dir = os.path.dirname(os.path.realpath(__file__))
    #target_script_path = os.path.join(script_dir, target_script_path)
    #fitnessFuncArgs = import_module_from_path(target_script_path).fitnessFuncArgs
    #from DIV21.src.conv_params import conv_params
    #from RBS_network_models.developing.CDKL5.DIV21.src.conv_params import conv_params
    from RBS_network_models.CDKL5.DIV21.src.conv_params import conv_params
    
    # def get_fitness_file_path(sim_data_path, saved_files):
    #     # if saved_files is not None:
    #     #     #find the file with data and json in the name
    #     #     data_file = [f for f in saved_files if '_data' in f and 'json' in f][0]
    #     #     #replace _data with _fitness
    #     #     fitness_file = data_file.replace('_data', '_fitness')
    #     # else:
    #     fitness_file = sim_data_path.replace('_data', '_fitness')
    #     return fitness_file
    # fitness_save_path = get_fitness_file_path(sim_data_path, saved_files)        
    
    #from DIV21.src.fitness_targets import fitnessFuncArgs
    #from RBS_network_models.developing.CDKL5.DIV21.src.fitness_targets import fitnessFuncArgs
    from RBS_network_models.CDKL5.DIV21.src.fitness_targets import fitnessFuncArgs
    fitness_save_path = sim_data_path.replace('_data', '_fitness')
    kwargs = {
        'simConfig': sim.cfg,
        #'conv_params': convolution_params.conv_params,
        'conv_params': conv_params,
        'popData': sim.net.allPops,
        'cellData': sim.net.allCells, #not actually used in the fitness calculation, but whatever
        'targets': fitnessFuncArgs['targets'],
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'features': fitnessFuncArgs['features'],
        'fitness_save_path': fitness_save_path,
        'break_deals': False #no need to do deal_breakers when doing refitting or sensitivity analysis
    }
    average_fitness = fitnessFunc(simData, mode='simulated data', **kwargs)
    return average_fitness
