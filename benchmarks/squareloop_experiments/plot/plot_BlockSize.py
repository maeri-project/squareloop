import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

block_sizes = [8, 16, 32, 64, 128, 256]

df = pd.read_csv('results/BlockSize/stats.csv', skipinitialspace=True)
#df = pd.read_csv('BlockSize.csv', skipinitialspace=True)

crypto_lat = df.loc[(df['Type'] == 'Map') & (df['Block size'].isin(block_sizes)), 'Latency'].tolist()
nocrypto_lat = df.loc[(df['Type'] == 'NoCrypto') & (df['Block size'].isin(block_sizes)), 'Latency'].tolist()

latencies_raw = [x for pair in zip(nocrypto_lat, crypto_lat) for x in pair]


# Normalize to global min
global_min = min(latencies_raw)
latencies_norm = [v / global_min for v in latencies_raw]

# Define bar layout
bar_width = 0.6
color_list = ["#808080", "#990000"]
method_labels = ["Without encrypted authentication", "With encrypted authentication"]
group_labels = [str(i) for i in block_sizes]

# Font sizes (reduced)
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

# X positions
x_positions = []
group_centers = []

# Remaining crypto groups: 2 bars each
center = 1
for _ in range(len(latencies_raw)//2):
    x_positions.append(center - bar_width/2)
    x_positions.append(center + bar_width/2)
    group_centers.append(center)
    center += 1.5

# Plot
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(x_positions, latencies_norm, width=bar_width,
              color=color_list * len(group_centers),
              edgecolor='k')

# X ticks
ax.set_xlabel("Block Size", fontsize=26)
ax.set_xticks(group_centers)
ax.set_xticklabels(group_labels, fontsize=28, rotation=0)

# Labels and limits
ax.set_ylabel("Normalized latency", fontsize=26)
ax.set_ylim([0, max(latencies_norm) * 1.2])
ax.tick_params(axis='y', labelsize=26)

# Annotate bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=20, fontweight='bold')

# Legend
handles = [plt.Rectangle((0, 0), 1, 1, color=c, edgecolor='k') for c in color_list]
ax.legend(handles, method_labels, fontsize=19,
          loc='best', ncol=3, handletextpad=0.4, columnspacing=0.3, borderpad=0.2)

# Horizontal reference line at y=1
ax.axhline(y=1, color='black', linestyle='dotted', linewidth=1)

# Save and show
plt.tight_layout()
plt.savefig("block_size_scaling.pdf", bbox_inches="tight", transparent=True)
plt.show()
