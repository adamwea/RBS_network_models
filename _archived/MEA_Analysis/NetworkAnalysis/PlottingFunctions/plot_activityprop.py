import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import pdb
import json
from math import sqrt
# This script reads the compiled csv made by compileNetworkFiles_JL.m and a reference note
# to plot the wt vs. het burst properties overdays

# setting starts here
###############################################################################################################################################
# Read the JSON file
with open('act_plt_settings.json', 'r') as file:
    settings_data = json.load(file)
## set plot saving dir
opDir = settings_data['opDir']
# set data and reference note dir
data_f = opDir + 'ActivityScan_outputs/Compiled_ActivityScan.csv'
#data_f = '/home/jonathan/Documents/Scripts/Matlab/scripts_output/CDKL5/ActivityScan_outputs/Compiled_ActivityScan.csv'
reference_f = settings_data['refDir']
#reference_f = '/home/jonathan/Documents/Scripts/Python/CDKL5_Notes.xlsx'

#opDir = '/home/jonathan/Documents/Scripts/Matlab/scripts_output/CDKL5/'

input_string1 = input("Enter the assay type that need to be plotted")
# set exclude lists
if not input_string1:
    print("no  assay value inputted")
    exit(0)
elif len(input_string1.split(','))> 1:
    assay_type_keywords = [x.lower().strip() for x in input_string1.split(',')]
else:
    assay_type_keywords=[input_string1.lower().strip()]
chip_exclude = []
well_exclude = []
while True:
    # Prompt for chip id input
    input_string2 = input("Enter a chip id to exclude (hit enter to finish): ")
    
    # Break the loop if the input is empty, indicating completion
    if not input_string2:
        break

    try:
        # Add the entered chip id to the exclusion list
        chip_exclude.append(input_string2)
        print(f"Chip being excluded: {input_string2}")

        # Request corresponding well id for the entered chip id
        well_input = input(f"Input well id for chip {input_string2}: ")
        
        # Validate and add the well id to the exclusion list
        well_exclude.append(int(well_input))
    except ValueError:
        print("Invalid input for well id. Please enter a numerical value.")
    except Exception as e:
        print(f"An error occurred: {e}")



input_string3 = input("Enter comma-separated run_ids to exclude (hit enter if none )")
# set exclude lists
if not input_string3:
    run_exclude =[]
else :
    try:
        run_exclude =[int(x) for x in input_string3.split(',')]
        print(f"chips being excluded are {run_exclude}")
    except Exception:
        print("Invalid input")

input_string4 = input("Enter  chip id+wellid track them (hit enter if none )")
# set exclude lists
if not input_string4:
    track_chips =[]
else :
    try:
        track_chips =[x for x in input_string4.split(',')]
        print(f"chips being tracked are {track_chips}")
    except Exception:
        print("Invalid input")# setting ends here
###############################################################################################################################################

# make output dir if not existed
opt_dir = opDir + 'ActivityScan_outputs/meanActivityProperty_graphs/'
if not os.path.exists(opt_dir):
    os.makedirs(opt_dir)

# read files
data_df = pd.read_csv(data_f)
ref_df = pd.read_excel(reference_f)

# sort df based on Run IDs and reindex
data_df = data_df.sort_values(by=['Run_ID'],ascending=True,ignore_index=True)
ref_df = ref_df.sort_values(by=['Run #'],ascending=True,ignore_index=True)

#To do : need to check why NAN
#data_df = data_df['DataFrame Column'].fillna(0)
data_df = data_df.replace(np.NaN,0.0)
print(data_df)
#combine data df with reference note
assay_l = []
genotype_l = []
for i in data_df.index:
    if data_df.loc[i]['Run_ID'] in list(ref_df['Run #']):
        temp_df = ref_df[ref_df['Run #'] == data_df.loc[i]['Run_ID']]
        assay = str(temp_df['Assay'].unique()[0])
        assay_l.append(assay)
        genotype = str(temp_df['Neuron Source'].unique()[0])
        genotype_l.append(genotype)
assay_l = [x.lower() for x in assay_l]
genotype_l = [x.lower() for x in genotype_l]

df = data_df.assign(Assay=assay_l)
#pdb.set_trace()
print(df.columns)
def plot_network_graph(working_df,output_type, assay_type):
    #extract data based on assay_type
    df = working_df[working_df['Assay'].str.lower().str.contains(assay_type.lower())]
    #create assay title
    assay_title = assay_type.title()
    #define input based on output
    if output_type == 'Mean_FiringRate':
        title = 'Firing Rate'
    if output_type == 'Mean_SpikeAmplitude':
        title = 'Amplitude'

    #div array
    div = df['DIV'].unique()
    total_div = len(div)
    # Find unique values in 'Genotype' column and count them
    unique_genotypes = df['NeuronType'].str.strip().unique()
    total_genotypes = len(unique_genotypes)

    # Print the number of unique genotypes
    print(f"Number of unique Genotypes: {total_genotypes}")

    # Initialize output arrays for each unique genotype
    output_arrays = {genotype: [] for genotype in unique_genotypes}
    chip_arrays = {genotype: [] for genotype in unique_genotypes}
    well_arrays = {genotype: [] for genotype in unique_genotypes}

    print(unique_genotypes)
    # Fill data from data frame
    for i in div:
        for genotype in unique_genotypes:
            temp_df = df.loc[(df['DIV'] == i) & (df['NeuronType'].str.strip() == genotype)]
            output_arrays[genotype].append(np.array(temp_df[output_type]))
            chip_arrays[genotype].append(np.array(temp_df['Chip_ID']))
            well_arrays[genotype].append(np.array(temp_df['Well']))
    ##plot
    # bar width
    w = total_div/32
    gaplength = 1/(len(unique_genotypes)+2)
    #create x-coordinates of bars
   # Create x-coordinates of bars
    x_day = [int(d) for d in div]
    x_genotype = {genotype: [] for genotype in unique_genotypes}
    x_d = list(range(0, total_div))
    
    # Assign x-coordinates for each genotype
    for i, genotype in enumerate(unique_genotypes):
        for x in x_d:
            x_genotype[genotype].append(x + (gaplength*i+1))
    #plotting
    fig, ax = plt.subplots()
    # Generate a list of distinct colors based on the number of genotypes
    colors = [plt.colormaps['Set1'](i) for i in np.linspace(0, 1, len(unique_genotypes))]# Using a colormap to generate colors
    colors2 = [plt.colormaps['Set2'](i) for i in np.linspace(0, 1, len(unique_genotypes))]#
    marker_shapes = ['^', 's', 'v', 'D', '+', 'x', '*', 'H', '8']
    marker_chips={chip:marker_shapes[idx] for idx, chip in enumerate(track_chips)}
    #    # Plot data for each genotype
    mean_data_all ={}
    yerr_data_all = {}
    n_data_all={}
    #plot WT bar
    for i,genotype in enumerate(unique_genotypes):
        y_data = output_arrays[genotype]

        chipy_data = chip_arrays[genotype]
        welly_data = well_arrays[genotype]
        #print("type: ",type(genotype))
        # Calculate statistics
        mean_data = [np.mean([n for n in yi if np.isfinite(n)]) for yi in y_data]
        yerr_data = [np.std([n for n in yi if np.isfinite(n)], ddof=1)/np.sqrt(np.size(yi)) for yi in y_data]
        n_data = [len(yi) for yi in y_data]
        # Store statistics in dictionaries
        mean_data_all[genotype] = mean_data
        yerr_data_all[genotype] = yerr_data
        n_data_all[genotype] = n_data
        # Save statistics to file
        output_file = f"intermediate_files/{title}_{genotype}_statistics.txt"
        with open(output_file, 'w') as file:
            file.write(f"{genotype} Statistics\n")
            file.write("Mean: " + ", ".join([str(m) for m in mean_data]) + "\n")
            file.write("SEM: " + ", ".join([str(sem) for sem in yerr_data]) + "\n")
            file.write("Sample Size (n): " + ", ".join([str(n_data)]) + "\n")

        # Plot bars

        alpha_value = 0.7
        ax.bar(x_genotype[genotype],
            height=mean_data,
            yerr=yerr_data,
            capsize=3,
            width=gaplength,
            color=colors[i],
            edgecolor='black',
            ecolor='black',alpha=alpha_value,
            label=genotype)
            # Plot scatter points
        for j in range(len(x_genotype[genotype])):
            #ax.scatter(x_genotype[genotype][j] + np.zeros(y_data[j].size), y_data[j], s=20,color=colors2[i])
            combined_data = [chip + str(well) for chip, well in zip(chipy_data[j], welly_data[j])]

            # Check if the concatenated string is in track_chips, and set the marker accordingly
            markers = [marker_chips.get(chipwell, 'o') for chipwell in combined_data]
            #marker_chips[combined_data[0]] if combined_data[0] in track_chips else 'o'

            # Use the marker in the scatter plot
            jitter_amount=0.08
            for k in range(len(y_data[j])):
                #pdb.set_trace()
                ax.scatter(
                    x_genotype[genotype][j] +np.random.uniform(-jitter_amount, jitter_amount, 1),
                    y_data[j][k],
                    s=10,
                    color=colors[i] if markers[k]=='o' else 'black',
                    marker=markers[k]
                        )
   
    #perform ttest
    for i in range(len(x_d)):
        #maxim = max([max( output_arrays[genotype][i] )for genotype in unique_genotypes])
        maxim = max(max(array) for genotype_arrays in output_arrays.values() for array in genotype_arrays)
        count = 1
        p_values = []
        for j, genotype1 in enumerate(unique_genotypes):
            for k, genotype2 in enumerate(unique_genotypes):
                if j < k:
                    #pdb.set_trace()
                    #print("mean_data_all",mean_data_all[genotype1])
                    #print("type:",type(genotype1))
                    mean1, sem1, n1 = mean_data_all[genotype1][i], yerr_data_all[genotype1][i], n_data_all[genotype1][i]
                    mean2, sem2, n2 = mean_data_all[genotype2][i], yerr_data_all[genotype2][i], n_data_all[genotype2][i]
                    #t_stat, p_value = stats.ttest_ind_from_stats(mean1, sem1, n1, mean2, sem2, n2)
                    sed = sqrt(sem1**2.0 + sem2**2.0)
                    t_stat = (mean1 - mean2) / sed
                    # degrees of freedom
                    degreef = n1+n2 - 2
                    alpha=0.05
                    # calculate the critical value
                    cv = stats.t.ppf(1.0 - alpha, degreef)
                    # calculate the p-value
                    p_value = (1.0 - stats.t.cdf(abs(t_stat), degreef)) * 2.0
                    p_values.append([mean1,sem1,mean2,sem2,p_value])

                    # Plot significance
                    #maxim = max(np.max(output_arrays[genotype1][i]), np.max(output_arrays[genotype2][i]))
                    x1, x2 = x_genotype[genotype1][i], x_genotype[genotype2][i]
                    ax.plot([x1, x2], [maxim + 0.05*maxim*(count)] * 2, 'k', linewidth=1.5)
                    sign = "***" if p_value <= 0.001 else "**" if p_value <= 0.01 else "*" if p_value <= 0.05 else "ns"
                    
                    ax.text((x1 + x2) / 2, maxim +0.05*maxim*(count), sign, ha='center', va='bottom', fontsize=7)
                    count = count +1
    
                    with open(output_file, 'a') as file:
                                file.write(f"P values:{p_values} \n")

    # Axis scaling and labeling
    xmin = 0
    xmax = (max(df['DIV']) - xmin)*1.25
    ymin = 0
    ymax = (max(df[output_type]) - ymin)*1.4

    plt.title(assay_title + ' ' + title)
    plt.xlabel('DIV')
    plt.ylabel(title)
    plt.xticks(list(map(lambda x: x + 1+(2*gaplength), x_d)), x_day)
    plt.axis([xmin, total_div + 1, ymin, ymax])
    plt.legend(title='type',loc='upper right', bbox_to_anchor=(1.1, 1.05))


    svg_dir = os.path.join(opt_dir, 'svg')
    jpg_dir = os.path.join(opt_dir, 'jpg')

    # Create the directories if they do not exist
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)

    # Now, save the figures to the specified format and directory
    plt.savefig(os.path.join(svg_dir, f"{assay_title} {title}.pdf"), dpi=300, format='svg')
    plt.savefig(os.path.join(jpg_dir, f"{assay_title} {title}.jpg"), dpi=300, format='jpg')
    return fig

#exclude chip ids and runs that are in the exclude list
exclude_l = []
# Filter rows based on chip_exclude and well_exclude
#pdb.set_trace()
# Creating a dictionary for easier exclusion lookup
exclude_dict = dict(zip(chip_exclude, well_exclude))


if exclude_dict:
    # Using list comprehension for filtering
    mask = [(row.Chip_ID in exclude_dict) and (row.Well == exclude_dict[row.Chip_ID]) for index, row in df.iterrows()]

    df = df[~pd.Series(mask)]


    # Filter rows based on run_exclude
df = df[~df['Run_ID'].isin(run_exclude)]


#Run grapher
#assay_types = list(df['Assay'].unique())
for i in assay_type_keywords:
    #working_df = df[df['Assay'].str.lower().str.contains(i.lower())]
    plot_network_graph(df, 'Mean_FiringRate',i)
    plot_network_graph(df, 'Mean_SpikeAmplitude',i) 
