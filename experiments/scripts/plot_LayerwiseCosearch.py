import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd  # unused
# import yaml  # unused
import sys
import os
import utils as util_script  # noqa: E402
from utils import squareloop_dir
sys.path.append(os.path.dirname(__file__))

arch = 'eyeriss'
model = 'resnet18'

result_folder = squareloop_dir+'experiments/results/LayerwiseCosearch/' + model + '/'


########################
# Compute rehash latencies for s2loop from search_script
########################

# Paths for dependency graph and layer layouts
dependency_file = util_script.squareloop_dir + 'benchmarks/script/crosslayer_search/test/resnet18/resnet18_dependent.yaml'
base_path = util_script.squareloop_dir + 'experiments/results/LayerwiseCosearch/resnet18'
problem_path = util_script.squareloop_dir + 'benchmarks/layer_shapes/resnet18'

# Build dependency groups and read layouts
dataspace_deps, _ = util_script.parse_dataspace_dependencies(dependency_file)
layer_dataspace_layouts = util_script.read_all_layer_dataspace_layouts(base_path, problem_path)

# Calculate per-group rehash latencies and map to 21 layers (default 0 if missing)
group_rehash_latencies = util_script.calculate_rehash_latency(dataspace_deps, layer_dataspace_layouts, util_script.crypto_file)
rehash_latencies_s2loop = [group_rehash_latencies.get(i, 0) for i in range(1, 22)]

########################
# Plot the results
########################

latencies_s2loop = [
    1204280,    # 1
    688264,     # 2
    688264,     # 3
    688264,     # 4
    688264,     # 5
    344222,     # 6
    688336,     # 7
    57398,      # 8
    688336,     # 9
    688336,     # 10
    344220,     # 11
    688336,     # 12
    57398,      # 13
    688336,     # 14
    688448,     # 15
    344360,     # 16
    688177,     # 17
    57382,      # 18
    688324,     # 19
    688324,     # 20
    769536      # 21
]

latencies_sloop = [
    1204416,  # Layer 1
    713856,   # Layer 2 
    688128,   # Layer 3
    713856,   # Layer 4
    688128,   # Layer 5
    380928,   # Layer 6
    688128,   # Layer 7
    285696,   # Layer 8
    688128,   # Layer 9
    688128,   # Layer 10
    448512,   # Layer 11
    897024,   # Layer 12
    140160,   # Layer 13
    897024,   # Layer 14
    897024,   # Layer 15
    1772544,  # Layer 16
    3551232,  # Layer 17
    198144,   # Layer 18
    3551232,  # Layer 19
    3551232,  # Layer 20
    769536    # Layer 21
]
rehash_latencies_sloop = [
    0,          # 0
    861792,     # 1-2 and 1-4
    0,          # 2
    150552,     # 3-4
    259584,     # 4
    317208,     # 5-6 and 5-8
    0,          # 6
    163116,     # 7-9
    75276,      # 8-9
    0,          # 9
    145356,     # 10-11 and 10-13
    0,          # 11
    75462,      # 12-14
    37638,      # 13-14
    0,          # 14
    70278,      # 15-16 and 15-18
    0,          # 16
    38022,      # 17-19
    18822,      # 18-19
    0,          # 19
    19212       # 20-21
]
# Calculate total latencies
sloop_total = sum(latencies_sloop) + sum(rehash_latencies_sloop)
s2loop_total = sum(latencies_s2loop) + sum(rehash_latencies_s2loop)
print(f"Total Sloop latency: {sloop_total}")
print(f"Total S2loop latency: {s2loop_total}")
# Calculate geometric mean ratio between sloop and s2loop totals
# Calculate individual ratios for each layer
ratios = []
for i in range(len(latencies_sloop)):
    sloop_layer_total = latencies_sloop[i] + rehash_latencies_sloop[i]
    s2loop_layer_total = latencies_s2loop[i] + rehash_latencies_s2loop[i]
    if sloop_layer_total > 0:  # Avoid division by zero
        ratios.append(sloop_layer_total/ s2loop_layer_total)

# Calculate geometric mean of ratios
geomean_ratio = np.exp(np.mean(np.log(ratios)))
print(f"Geometric mean ratio (S2loop/Sloop): {geomean_ratio:.3f}")

# Create data for stacked bar chart
layers = range(1, 22)  # 21 layers
width = 0.35  # Reduced width to accommodate side-by-side bars

# Create the figure and axis
fig, ax = plt.subplots(figsize=(15, 6.5))

# Create stacked bars for sloop
sloop_bars = ax.bar(np.array(layers) - width/2, latencies_sloop, width, label='Layer Latency (Sloop)', color='grey', edgecolor='black')
sloop_rehash = ax.bar(np.array(layers) - width/2, rehash_latencies_sloop, width, bottom=latencies_sloop, label='Rehash Latency (Sloop)', color='#990000', edgecolor='black')

# Create stacked bars for s2loop
s2loop_bars = ax.bar(np.array(layers) + width/2, latencies_s2loop, width, label='Layer Latency (S2loop)', color='grey', edgecolor='black')
s2loop_rehash = ax.bar(np.array(layers) + width/2, rehash_latencies_s2loop, width, bottom=latencies_s2loop, label='Rehash Latency (S2loop)', color='#990000', edgecolor='black')


# Add black arrows where s2loop total is less than sloop total
for i in range(len(layers)):
    sloop_total = latencies_sloop[i] + rehash_latencies_sloop[i]
    s2loop_total = latencies_s2loop[i] + rehash_latencies_s2loop[i]
    if s2loop_total < sloop_total:
        ax.annotate('',
                   xy=(layers[i], s2loop_total), 
                   xytext=(layers[i]+0.5, s2loop_total+200000),
                   arrowprops=dict(facecolor='black', shrink=0.05),
                   fontsize=12)

# Add arrows for locations where sloop total is less than s2loop total
for i in range(len(layers)):
    sloop_total = latencies_sloop[i] + rehash_latencies_sloop[i]
    s2loop_total = latencies_s2loop[i] + rehash_latencies_s2loop[i]
    if sloop_total < s2loop_total:
        ax.annotate('',
                   xy=(layers[i], sloop_total),
                   xytext=(layers[i]-0.5, sloop_total+200000),
                   arrowprops=dict(facecolor='#990000',  edgecolor='#990000',shrink=0.05),
                   fontsize=12)

# Customize the plot
ax.set_xlabel('Layer ID', fontsize=26)
ax.set_ylabel('Latency (cycles)', fontsize=26)
# Create legend
legend = ax.legend(['Compute Latency', 'Rehash Latency'], fontsize=26)

# Add arrow annotation next to legend
ax.annotate('',
           xy=(8.9, 3.4*1e6),
           xytext=(8.4, 3.4*1e6+200000),
           arrowprops=dict(facecolor='#990000', edgecolor='#990000', shrink=0.05),
           fontsize=20)
# Add text annotation
ax.text(15, 3.5*1e6, r'caused by detailed', fontsize=22, ha='right', va='center')
ax.text(15, 3.2*1e6, r'memory modeling in $S^2Loop$', fontsize=22, ha='right', va='center')

ax.annotate('',
           xy=(8.9, 2.7*1e6+200000),
           xytext=(8.4, 2.7*1e6),
           arrowprops=dict(facecolor='black', shrink=0.05),
           fontsize=20)
ax.text(15, 2.8*1e6, r'$S^2Loop$ finds better', fontsize=22, ha='right', va='center')
ax.text(15, 2.5*1e6, 'Mapping using memory-aware search', fontsize=22, ha='right', va='center')

# Set x-axis ticks
ax.set_xticks(layers)
ax.tick_params(axis='both', which='major', labelsize=26)

# Add grid for better readability
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()
plt.savefig(result_folder+"LayerwiseCosearch_"+arch+"_"+model+".pdf", bbox_inches="tight", transparent=True)
