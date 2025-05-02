import os
import numpy as np
from netpyne import specs
from scipy.stats import truncnorm
from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import import_module_from_path
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
        
# Initialize NetParams object
netParams = specs.NetParams()

# Function for dynamic truncation to ensure symmetry
def dynamic_positive_normal(mean, std, size):    
    a, b = (0 - mean) / std, float("inf")
    initial_values = truncnorm.rvs(a, b, loc=mean, scale=std, size=size * 2)
    left_truncated_count = np.sum(initial_values < 0)
    upper_truncation_value = np.percentile(initial_values, 100 - (left_truncated_count / len(initial_values)) * 100)
    b_dynamic = (upper_truncation_value - mean) / std
    values = truncnorm.rvs(a, b_dynamic, loc=mean, scale=std, size=size)
    return values


# def create_network(cfg):
#     # Generate properties for all neurons
#     num_excite = cfg.num_excite
#     num_inhib = cfg.num_inhib

#     excitatory_props = {
#         'L': dynamic_positive_normal(cfg.E_L_mean, cfg.E_L_stdev, num_excite),
#         'diam': dynamic_positive_normal(cfg.E_diam_mean, cfg.E_diam_stdev, num_excite),
#         'Ra': dynamic_positive_normal(cfg.E_Ra_mean, cfg.E_Ra_stdev, num_excite),
#         'gnabar': dynamic_positive_normal(cfg.gnabar_E, cfg.gnabar_E_std, num_excite),
#         'gkbar': dynamic_positive_normal(cfg.gkbar_E, cfg.gkbar_E_std, num_excite),
#     }
#     inhibitory_props = {
#         'L': dynamic_positive_normal(cfg.I_L_mean, cfg.I_L_stdev, num_inhib),
#         'diam': dynamic_positive_normal(cfg.I_diam_mean, cfg.I_diam_stdev, num_inhib),
#         'Ra': dynamic_positive_normal(cfg.I_Ra_mean, cfg.I_Ra_stdev, num_inhib),
#         'gnabar': dynamic_positive_normal(cfg.gnabar_I, cfg.gnabar_I_std, num_inhib),
#         'gkbar': dynamic_positive_normal(cfg.gkbar_I, cfg.gkbar_I_std, num_inhib),
#     }

#     # Create cell rules
#     create_cell_params('E', excitatory_props, num_excite)
#     create_cell_params('I', inhibitory_props, num_inhib)

#     # Define populations
#     define_populations('E', cfg, excitatory_props, num_excite)
#     define_populations('I', cfg, inhibitory_props, num_inhib)
    
#     # Define connectivity
#     define_connectivity(cfg)

# Main execution
def main(cfg):
    
    #extract num_excite and num_inhib from cfg
    num_excite = cfg.num_excite
    num_inhib = cfg.num_inhib
    
    # initialize normally distributed properties
    def init_normally_distributed_properties(cfg, num_excite, num_inhib):

        excitatory_props = {
            'L': dynamic_positive_normal(cfg.E_L_mean, cfg.E_L_stdev, num_excite),
            'diam': dynamic_positive_normal(cfg.E_diam_mean, cfg.E_diam_stdev, num_excite),
            'Ra': dynamic_positive_normal(cfg.E_Ra_mean, cfg.E_Ra_stdev, num_excite),
            'gnabar': dynamic_positive_normal(cfg.gnabar_E, cfg.gnabar_E_std, num_excite),
            'gkbar': dynamic_positive_normal(cfg.gkbar_E, cfg.gkbar_E_std, num_excite),
        }
        inhibitory_props = {
            'L': dynamic_positive_normal(cfg.I_L_mean, cfg.I_L_stdev, num_inhib),
            'diam': dynamic_positive_normal(cfg.I_diam_mean, cfg.I_diam_stdev, num_inhib),
            'Ra': dynamic_positive_normal(cfg.I_Ra_mean, cfg.I_Ra_stdev, num_inhib),
            'gnabar': dynamic_positive_normal(cfg.gnabar_I, cfg.gnabar_I_std, num_inhib),
            'gkbar': dynamic_positive_normal(cfg.gkbar_I, cfg.gkbar_I_std, num_inhib),
        }
        return excitatory_props, inhibitory_props
    excitatory_props, inhibitory_props = init_normally_distributed_properties(cfg, num_excite, num_inhib)
    
    #set size of 3D space
    #for new method
    #these are position params taken from old KCNT1 data I think...these are a place holder.
    #TODO: get position data from CDKL5_DIV21 data
    DEFAULT_Y_RANGE = [6.72, 2065.09]
    DEFAULT_X_RANGE = [173.66, 3549.91]
    DEFAULT_Z_RANGE = [1, 11.61]
    netParams.sizeX = 4000
    netParams.sizeY = 2100
    netParams.sizeZ = 12
    
    #Generate clustered positions of neurons in a rectangular culture
    num_cells = num_excite + num_inhib
    def generate_clustered_neuronal_positions(y_range, x_range, z_range, num_cells=500, n_clusters=5):
        """
        Generate realistically clustered positions of neurons in a rectangular culture.
        
        Parameters:
            y_range (list): Range of Y coordinates [min, max].
            x_range (list): Range of X coordinates [min, max].
            z_range (list): Range of Z coordinates [min, max].
            num_cells (int): Total number of neurons to generate.
            n_clusters (int): Number of clusters to simulate.
        
        Returns:
            np.ndarray: Array of clustered neuron positions (x, y, z).
        """
        # Generate cluster centers
        cluster_centers = np.column_stack((
            np.random.uniform(x_range[0], x_range[1], n_clusters),  # X cluster centers
            np.random.uniform(y_range[0], y_range[1], n_clusters),  # Y cluster centers
            np.random.uniform(z_range[0], z_range[1], n_clusters)   # Z cluster centers
        ))
        
        # Distribute cells among clusters
        cells_per_cluster = [num_cells // n_clusters] * n_clusters
        remainder = num_cells % n_clusters
        for i in range(remainder):
            cells_per_cluster[i] += 1  # Distribute remainder randomly
        
        # Generate points for each cluster
        positions = []
        for i, center in enumerate(cluster_centers):
            cluster_points = np.random.multivariate_normal(
                mean=center, 
                cov=np.diag([100, 100, 1]),  # Covariance matrix for spread
                size=cells_per_cluster[i]
            )
            positions.append(cluster_points)
        
        # Combine all cluster points
        positions = np.vstack(positions)
        
        return positions
    positions = generate_clustered_neuronal_positions(DEFAULT_Y_RANGE, DEFAULT_X_RANGE, DEFAULT_Z_RANGE, 
                                                      num_cells=num_cells, 
                                                      n_clusters=5
                                                      )
    
    #of the generated positions, randomly select subsets for excitatory and inhibitory cells
    def select_cells_from_positions(positions, num_excite, num_inhib):
        # Shuffle indices to ensure random selection
        indices = np.arange(len(positions))
        np.random.shuffle(indices)
        
        # Select cells for excitatory and inhibitory populations
        E_cells = indices[:num_excite]
        I_cells = indices[num_excite:num_excite + num_inhib]
        
        posE = positions[E_cells]
        posI = positions[I_cells]
        
        return posE, posI
    
    posE, posI = select_cells_from_positions(positions, num_excite, num_inhib) 
    
    #config cell params
    def config_cell_params(cell_type, props, num_cells, positions):
        # Default 3D space ranges for random placement
        
        old_method = False
        new_method = True #I might completely be off about how to do this...
        third_method = False
        if old_method:
            cellRule = {
                'conds': {'cellType': cell_type},
                'secs': {
                    'soma': {
                        'geom': {
                            'L': props['L'].tolist(),
                            'diam': props['diam'].tolist(),
                            'Ra': props['Ra'].tolist(),
                        },
                        'mechs': {
                            'hh': {
                                'gnabar': props['gnabar'].tolist(),
                                'gkbar': props['gkbar'].tolist(),
                                'gl': [0.003] * num_cells, 'el': [-70] * num_cells,
                            }
                        }
                    }
                }
            }
            netParams.cellParams[cell_type] = cellRule
        elif new_method:            
            
            #build individual cell rules for each cell
            if hasattr(netParams, 'cellsLists') is False:
                netParams.cellsLists = {}
            netParams.cellsLists[cell_type] = []
            secsList = []
            cellLabelList = []
            for i in range(num_cells):
                pos = positions[i]
                x, y, z = pos
                #make sure all values are floats, not np.float64
                #diam = float(props['diam'][i])
                #x, y, z = float(x), float(y), float(z)
                cellsLabel = f'{cell_type}{i}'
                cellRule = {
                    #'pop': cell_type,
                    #'cellLabel': cellsLabel,
                    #'cellModel': 'HH',                   
                    'conds': {                          # I finally understgand how conds works. This sets the conditions required for population
                                                        # attributes to be applied to the cell.
                                                        # In this case, the cellType, cellModel, cellLabel, and pop must be equal to the values
                                                        # specified later in the popParams dictionary.
                                                        # I do cell params before pop params so I can pass cellsList 
                                                        # to popParams - but now I'm not sure if I need to do that at all.
                                                        # everything is based on cellTags I guess.
                        'cellType': cell_type,
                        'cellModel': 'HH',
                        'pop': cell_type,
                        'cellLabel': cellsLabel,
                        #'x': x, 'y': y, 'z': z,
                    },
                    'secs': {
                        'soma': {
                            'geom': {
                                'L': props['L'][i],
                                'diam': props['diam'][i],
                                'Ra': props['Ra'][i],
                                ## apprently I can also pass position this way
                                #'pt3d': [x, y, z, diam], # x, y, z, diameter #nvm idk how to get this to validate
                                
                                #REVISED COMMENT:
                                # I could do this if I wwanted. what this is useful for is building morphologies
                                # of specific compartments.
                                #the correct way to implement:
                                # 'pt3d': [
                                #    (0, 0, 0, 1),
                                #    (0, 0, 1, 1),
                                #    (0, 0, 2, 1),
                                # etc... a list of tuples of x, y, z, diameter
                                # ],
                                # TODO: I will need to do this eventually.
                                # TODO: extract morphologies from CDKL5_DIV21 and get them into this format
                            },
                            'mechs': {
                                'hh': {
                                    'gnabar': props['gnabar'][i],
                                    'gkbar': props['gkbar'][i],
                                    'gl': 0.003, 'el': -70,
                                }
                            },
                        }
                    },
                    #'x': x, 'y': y, 'z': z,
                },
                cellRule = cellRule[0]
                netParams.cellsLists[cell_type].append(
                    {
                        'cellLabel': cellsLabel,
                        'x': x, 'y': y, 'z': z,     #yup, the only thing cellsList is good for is to pass exact positions to the popParams
                        #'cellModel': 'HH',         #Let me ammend that, cellsList is meant to pass positions and popParams to specific cells
                        #                           #cellLabel lets me apply specific cell rules to specifc cells
                        #                           #any other conds that can be applied globably to all cells in a population
                        #                           #should be applied in popParams config
                        #maybe over here?
                        #'pt3d': (x, y, z, props['diam'][i]), # x, y, z, diameter NOPE SEE ABOVE FOR REVISED COMMENT
                    }
                )
                netParams.cellParams[cellsLabel] = cellRule         
            print(f'Created {num_cells} cells of type {cell_type}')
            print(f'netParams.cellParams configured for {cell_type}')
        elif third_method:
            cellRule = {
                'conds': {
                    'cellType': cell_type,
                    'cellModel': 'HH',
                    },
                #'secs': {}
                #'cellModel': 'HH',
                # 'secs': {
                #     'soma': {
                #         'geom': {
                #             'L': props['L'][0],
                #             'diam': props['diam'][0],
                #             'Ra': props['Ra'][0],
                #         },
                #         'mechs': {
                #             'hh': {
                #                 'gnabar': props['gnabar'][0],
                #                 'gkbar': props['gkbar'][0],
                #                 'gl': 0.003, 'el': -70,
                #             }
                #         }
                #     }
                # }
                'diversityFraction': 0.1,
            }
            
            #for debug print sec params and vals
            import pprint
            #pprint.pprint(cellRule['secs']['soma'])
            netParams.cellParams[cell_type] = cellRule
    config_cell_params('E', excitatory_props, num_excite, posE)
    config_cell_params('I', inhibitory_props, num_inhib, posI)
    
    #config pop params
    def config_pop_params(cell_type, cfg, props, num_cells):
        old_method = False
        new_method = True
        third_method = False
        if old_method:
            netParams.popParams['E'] = {
                'cellType': 'E',
                'numCells': cfg.num_excite,
                'cellModel': 'HH'
            }
            netParams.popParams['I'] = {
                'cellType': 'I',
                'numCells': cfg.num_inhib,
                'cellModel': 'HH'
            }
        elif new_method:
            #assert cellsList[cell_type] is a list
            assert isinstance(netParams.cellsLists[cell_type], list), f'netParams.cellsLists[{cell_type}] is not a list'
            netParams.popParams[cell_type] = {
                'cellType': cell_type,
                'numCells': num_cells,
                'cellModel': 'HH',
                'cellsList': netParams.cellsLists[cell_type],
                #'cellModel': 'HH',
                #'diversity': True,
            }
        elif third_method:
            #cellsList = {}
            # cellsList = []
            # for i in range(num_cells):
            #     cellsList.append({
            #         'cellLabel': f'{cell_type}{i}',
            #         'conds': {
            #             'cellType': cell_type,
            #             #'cellModel': 'HH',
            #             },
            #         #'cellModel': 'HH',
            #         'secs': {'soma': {
            #             'geom': {
            #                 'L': props['L'][i],
            #                 'diam': props['diam'][i],
            #                 'Ra': props['Ra'][i],
            #             },
            #             'mechs': {
            #                 'hh': {
            #                     'gnabar': props['gnabar'][i],
            #                     'gkbar': props['gkbar'][i],
            #                     'gl': 0.003, 'el': -70,
            #                 }
            #             }
            #         }}
            #     })
            netParams.popParams[cell_type] = {
                'cellType': cell_type,
                'cellsList': cellsList,
                'cellModel': 'HH',
            }
        print(f'Created {num_cells} populations of type {cell_type}')
        print(f'netParams.popParams configured for {cell_type}')
    config_pop_params('E', cfg, excitatory_props, num_excite)
    config_pop_params('I', cfg, inhibitory_props, num_inhib)
    
    #config connectivity params
    def config_connectivity(cfg):
        from time import time
        start = time()
        
        old_method = False
        if old_method:
        
            dynamic_normal_applied = False
            #dynamic_normal_applied = True - too complex, not implemented, not needed for now
            
            if not dynamic_normal_applied:
                print('Static normal applied')
                print('Generating static normal values for connectivity properties...')
                conn_specs = {
                    'E->I': {'pre': 'E', 'post': 'I', 'synMech': 'exc', 
                            'prob': cfg.probEI, 
                            'weight': cfg.weightEI},
                    'I->E': {'pre': 'I', 'post': 'E', 'synMech': 'inh', 
                            'prob': cfg.probIE, 
                            'weight': cfg.weightIE},
                    'E->E': {'pre': 'E', 'post': 'E', 'synMech': 'exc', 
                            'prob': cfg.probEE, 
                            'weight': cfg.weightEE},
                    'I->I': {'pre': 'I', 'post': 'I', 'synMech': 'inh', 
                            'prob': cfg.probII, 
                            'weight': cfg.weightII},
                }
                for key, spec in conn_specs.items():
                    netParams.connParams[key] = {
                        'preConds': {'cellType': spec['pre']},
                        'postConds': {'cellType': spec['post']},
                        'probability': spec['prob'],
                        'weight': spec['weight'],
                        'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                        'synMech': spec['synMech']
                    }

                netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': cfg.tau1_exc, 'tau2': cfg.tau2_exc, 'e': 0}
                netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': cfg.tau1_inh, 'tau2': cfg.tau2_inh, 'e': -75}
                
                print('Connectivity properties generated')
                time_elapsed = time() - start
                print(f'Connectivity properties generated in {time_elapsed:.2f} seconds')
            else:
                print('Dynamic normal applied')
                print('Generating dynamic normal values for connectivity properties...')
                
                EI_props = {
                    'prob': dynamic_positive_normal(cfg.probEI, cfg.probEI_std, cfg.num_excite * cfg.num_inhib),
                    'weight': dynamic_positive_normal(cfg.weightEI, cfg.weightEI_std, cfg.num_excite * cfg.num_inhib),
                }
                IE_props = {
                    'prob': dynamic_positive_normal(cfg.probIE, cfg.probIE_std, cfg.num_inhib * cfg.num_excite),
                    'weight': dynamic_positive_normal(cfg.weightIE, cfg.weightIE_std, cfg.num_inhib * cfg.num_excite),
                }
                EE_props = {
                    'prob': dynamic_positive_normal(cfg.probEE, cfg.probEE_std, cfg.num_excite * cfg.num_excite),
                    'weight': dynamic_positive_normal(cfg.weightEE, cfg.weightEE_std, cfg.num_excite * cfg.num_excite),
                }
                II_props = {
                    'prob': dynamic_positive_normal(cfg.probII, cfg.probII_std, cfg.num_inhib * cfg.num_inhib),
                    'weight': dynamic_positive_normal(cfg.weightII, cfg.weightII_std, cfg.num_inhib * cfg.num_inhib),
                }
                
                for i in range(cfg.num_excite):
                    for j in range(cfg.num_inhib):
                        netParams.connParams[f'E->I_{i}_{j}'] = {
                            'preConds': {'cellType': 'E', 'cellIndex': i},
                            'postConds': {'cellType': 'I', 'cellIndex': j},
                            'probability': EI_props['prob'][i * cfg.num_inhib + j],
                            'weight': EI_props['weight'][i * cfg.num_inhib + j],
                            'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                            'synMech': 'exc'
                        }
                        
                for i in range(cfg.num_inhib):
                    for j in range(cfg.num_excite):
                        netParams.connParams[f'I->E_{i}_{j}'] = {
                            'preConds': {'cellType': 'I', 'cellIndex': i},
                            'postConds': {'cellType': 'E', 'cellIndex': j},
                            'probability': IE_props['prob'][i * cfg.num_excite + j],
                            'weight': IE_props['weight'][i * cfg.num_excite + j],
                            'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                            'synMech': 'inh'
                        }

                for i in range(cfg.num_excite):
                    for j in range(cfg.num_excite):
                        netParams.connParams[f'E->E_{i}_{j}'] = {
                            'preConds': {'cellType': 'E', 'cellIndex': i},
                            'postConds': {'cellType': 'E', 'cellIndex': j},
                            'probability': EE_props['prob'][i * cfg.num_excite + j],
                            'weight': EE_props['weight'][i * cfg.num_excite + j],
                            'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                            'synMech': 'exc'
                        }
                        
                for i in range(cfg.num_inhib):
                    for j in range(cfg.num_inhib):
                        netParams.connParams[f'I->I_{i}_{j}'] = {
                            'preConds': {'cellType': 'I', 'cellIndex': i},
                            'postConds': {'cellType': 'I', 'cellIndex': j},
                            'probability': II_props['prob'][i * cfg.num_inhib + j],
                            'weight': II_props['weight'][i * cfg.num_inhib + j],
                            'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                            'synMech': 'inh'
                        }
                
                netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': cfg.tau1_exc, 'tau2': cfg.tau2_exc, 'e': 0}
                netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': cfg.tau1_inh, 'tau2': cfg.tau2_inh, 'e': -75}
                
                print('Connectivity properties generated')
                time_elapsed = time() - start
                print(f'Connectivity properties generated in {time_elapsed:.2f} seconds')
        else:
            print('Static normal applied')
            print('Generating static normal values for connectivity properties...')
            conn_specs = {
                'E->I': {'pre': 'E', 'post': 'I', 'synMech': 'exc', 
                        'prob': cfg.probEI, 
                        'weight': cfg.weightEI},
                'I->E': {'pre': 'I', 'post': 'E', 'synMech': 'inh', 
                        'prob': cfg.probIE, 
                        'weight': cfg.weightIE},
                'E->E': {'pre': 'E', 'post': 'E', 'synMech': 'exc', 
                        'prob': cfg.probEE, 
                        'weight': cfg.weightEE},
                'I->I': {'pre': 'I', 'post': 'I', 'synMech': 'inh', 
                        'prob': cfg.probII, 
                        'weight': cfg.weightII},
            }
            for key, spec in conn_specs.items():
                netParams.connParams[key] = {
                    'preConds': {'cellType': spec['pre']},
                    'postConds': {'cellType': spec['post']},
                    'probability': spec['prob'],
                    'weight': spec['weight'],
                    'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                    'synMech': spec['synMech']
                }

            netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': cfg.tau1_exc, 'tau2': cfg.tau2_exc, 'e': 0}
            netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': cfg.tau1_inh, 'tau2': cfg.tau2_inh, 'e': -75}
            
            print('Connectivity properties generated')
            time_elapsed = time() - start
            print(f'Connectivity properties generated in {time_elapsed:.2f} seconds')
    config_connectivity(cfg)
    
    #save network
    def save_network(cfg, filename = None):
        #assert that cfg.saveFolder and cfg.simLabel are not '' or None
        try:
            assert cfg.saveFolder, 'cfg.saveFolder not found'
            assert cfg.simLabel, 'cfg.simLabel not found'
        except:
            import warnings
            warnings.simplefilter('always')
            
            # warning, if running a batch process but unable to find cfg.saveFolder and cfg.simLabel,
            # something is wrong.
            # however if running a single simulation, this is normal
            warning_message = ('\n'
                                'WARNING: cfg.saveFolder and cfg.simLabel not found.\n'
                                'This is normal if running a single simulation.\n'
                                'Otherwise, something is wrong.\n')
            warnings.warn(warning_message)
            
            print('generating temporary saveFolder and simLabel...')
            cfg.saveFolder = './testing'
            cfg.saveFolder = os.path.abspath(cfg.saveFolder)
            if not os.path.exists(cfg.saveFolder):
                os.makedirs(cfg.saveFolder)
            cfg.simLabel = 'test'
        output_path = os.path.join(cfg.saveFolder, f"{cfg.simLabel}_netParams.json")
        filename = output_path # if filename is None else filename maybe implement this later
        netParams.save(filename)
        print(f'Network parameters saved to {filename}')
    save_network(cfg)

# Import cfg and run
try:
    from __main__ import cfg
    main(cfg)
except ImportError:
    raise Exception("cfg not found")
