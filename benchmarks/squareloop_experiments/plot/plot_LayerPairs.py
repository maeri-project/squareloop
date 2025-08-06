import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

model = 'mobv3'
#model = 'resnet18'



def dependent_layer_pairs(dependency_file):
    with open(dependency_file, 'r') as f:
        data = yaml.safe_load(f)

    result = []
    result_real = []
    for key, value in data.items():
        key_layer_id = value["layer_id_for_timeloop"]
        for dep in value.get("dependent_next_layer", []):
            dep_layer_id = data[dep]["layer_id_for_timeloop"]
            result.append((key_layer_id, dep_layer_id))
            result_real.append((key, dep))

    #print(result)
    #print()
    #print(result_real)
    return result, result_real



df = pd.read_csv('../results/LayerPairs/stats.csv', skipinitialspace=True)
#df = pd.read_csv('LayerPairs.csv', skipinitialspace=True)

rehash_raw = df[(df['Type'] == 'Rehash') & (df['Model'] == model)]['Latency'].tolist()
rehash = [x for y in rehash_raw for x in [y,0]]

rehash_in = df[(df['Type'] == 'MapRehashIn') & (df['Model'] == model)]['Latency'].tolist()
in2out_in = df[(df['Type'] == 'In2OutIn') & (df['Model'] == model)]['Latency'].tolist()
out2in_in = df[(df['Type'] == 'Out2InIn') & (df['Model'] == model)]['Latency'].tolist()

rehash_out = df[(df['Type'] == 'MapRehashOut') & (df['Model'] == model)]['Latency'].tolist()
in2out_out = df[(df['Type'] == 'In2OutOut') & (df['Model'] == model)]['Latency'].tolist()
out2in_out = df[(df['Type'] == 'Out2InOut') & (df['Model'] == model)]['Latency'].tolist()

besti = [i2oi if i2oi+i2oo < o2ii+o2io else o2ii for (i2oi,i2oo,o2ii,o2io) in zip(in2out_in,in2out_out,out2in_in,out2in_out)]
besto = [i2oo if i2oi+i2oo < o2ii+o2io else o2io for (i2oi,i2oo,o2ii,o2io) in zip(in2out_in,in2out_out,out2in_in,out2in_out)]

layer_in = [x for pair in zip(rehash_in,besti) for x in pair]
layer_out = [x for pair in zip(rehash_out,besto) for x in pair]


#layer_pairs_in = df[(df['Type'] == 'MapRehashIn') & (df['Model'] == model)]['Layer'].tolist()
#layer_pairs_out = df[(df['Type'] == 'MapRehashOut') & (df['Model'] == model)]['Layer'].tolist()
#layer_pairs = [pair for pair in zip(layer_pairs_in, layer_pairs_out)]
_, layer_pairs = dependent_layer_pairs('dependencies/'+model+'.yaml')


# Raw latency values
#layer_in = [
#    921504, 921504,
#    663552, 663552
#]
#layer_out = [
#    921504, 2151576,
#    903168, 903168,
#]
#rehash = [
#    2151576, 0,
#    473976, 0,
#]

# Normalize to global min
#global_min = 1
#latencies_norm = [v / global_min for v in latencies_raw]

#layer_pairs = [(1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (20, 21), (21, 22), (22, 23), (23, 24), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (31, 32), (32, 33), (34, 35), (35, 36), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (57, 58), (58, 59), (59, 60), (60, 61)]






sum_bars = [lin+lout+r for lin, lout, r in zip(layer_in, layer_out, rehash)]




# Define bar layout
bar_width = 0.6
color_list = ["#D0D0D0", "#808080", "#990000"]
method_labels = ["First layer", "Second layer", "Rehash"]
#group_labels = ["Layers\n2,3", "Layers\n6,7", "Layers\n9,10", "Layers\n11,12", "Layers\n14,15", "Layers\n16,17", "Layers\n19,20"]
group_labels = ["Layers\n"+str(l1)+","+str(l2) for l1,l2 in layer_pairs]

# Font sizes (reduced)
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

# X positions
x_positions = []
group_centers = []

# Remaining crypto groups: 2 bars each
center = bar_width/2
for _ in range(len(layer_in)//2):
    x_positions.append(center - bar_width/2 - 0.025)
    x_positions.append(center + bar_width/2 + 0.025)
    group_centers.append(center)
    center += 1.5

# Plot
#fig, ax = plt.subplots(figsize=(12, 7))
fig, ax = plt.subplots(figsize=(24, 7))
bars1 = ax.bar(x_positions, layer_in, width=bar_width,
              color=color_list[0],
              edgecolor='k', bottom=0)
bars2 = ax.bar(x_positions, layer_out, width=bar_width,
              color=color_list[1],
              edgecolor='k', bottom=layer_in)
bars3 = ax.bar(x_positions, rehash, width=bar_width,
              color=color_list[2],
              edgecolor='k', bottom=[lin+lout for lin, lout in zip(layer_in, layer_out)])

# X ticks
ax.set_xticks(group_centers)
#ax.set_xticklabels(group_labels, fontsize=28, rotation=0)
ax.set_xticklabels(group_labels, fontsize=5, rotation=0)

# Labels and limits
ax.set_ylabel("Latency", fontsize=26)
ax.set_ylim([0, max(sum_bars) * 1.2])
ax.tick_params(axis='y', labelsize=26)

## Annotate bars
#for bar in bars3:
#    height = bar.get_height()
#    ax.annotate(f'{height:.2f}',
#                xy=(bar.get_x() + bar.get_width() / 2, height),
#                xytext=(0, 3),
#                textcoords="offset points",
#                ha='center', va='bottom', fontsize=20, fontweight='bold')

for i in range(0, len(sum_bars), 2):
    #x_end = bars1[i+1].get_x() - bars1[i+1].get_width()/2
    #x_start = bars1[i].get_x() + bars1[i].get_width()/2
    x_end = bars1[i+1].get_x()
    x_start = bars1[i].get_x() + bars1[i].get_width()
    y_end = sum_bars[i+1]
    y_start = sum_bars[i]
    ax.annotate(
        #'', xy=(x_end, y_end-0.05), xytext=(x_start, y_start-0.05),
        '', xy=(x_end+bar_width*1/10, y_end), xytext=(x_start-bar_width*1/10, y_start),
        #arrowprops=dict(arrowstyle="simple", color='black', lw=2, mutation_scale=15)
        arrowprops=dict(arrowstyle="simple", color='black', lw=1, mutation_scale=5)
    )
    diff = 100*(y_end-y_start)/y_start
    diff = ('-' if round(diff)==0 and diff<0 else '' if diff<0 else '+')+str(round(diff))+'%'
    ax.text(
        #x_end+bar_width*5/8, y_end, diff,
        x_end+bar_width*5/8, y_end+1000, diff,
        ha='center', va='bottom',
        #fontsize=16, color='black',
        fontsize=5, color='black',
        fontweight='bold'
    )

# Legend
handles = [plt.Rectangle((0, 0), 1, 1, color=c, edgecolor='k') for c in color_list]
ax.legend(handles, method_labels, fontsize=24,
          loc='best', ncol=3, handletextpad=0.4, columnspacing=0.3, borderpad=0.2)

# Horizontal reference line at y=1
ax.axhline(y=1, color='black', linestyle='dotted', linewidth=1)

# Save and show
plt.tight_layout()
plt.savefig("rehash.pdf", bbox_inches="tight", transparent=True)
plt.show()
