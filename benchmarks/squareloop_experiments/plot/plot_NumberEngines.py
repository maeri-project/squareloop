import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

num_engines = [2, 4, 8, 16, 32, 64]

df = pd.read_csv('../results/NumberEngines/stats.csv', skipinitialspace=True)
#df = pd.read_csv('NumberEngines.csv', skipinitialspace=True)

baseline_lat = df.loc[df['Type'] == 'NoCrypto', 'Latency'].tolist()
searched_lat = df.loc[(df['Type'] == 'Map') & (df['Shared crypto engine'] == False) & (df['Number of crypto engines'].isin(num_engines)), 'Latency'].tolist()
restricted_lat = df.loc[(df['Type'] == 'RestrictLayout') & (df['Shared crypto engine'] == False) & (df['Number of crypto engines'].isin(num_engines)), 'Latency'].tolist()


latencies_raw = baseline_lat + [x for pair in zip(searched_lat, restricted_lat) for x in pair]

# Normalize to global min
global_min = latencies_raw[0]
latencies_norm = [v / global_min for v in latencies_raw]

# Define bar layout
bar_width = 0.6
color_list = ["#990000", "#808080"]
color_list_ext = ["#D0D0D0"] + color_list
method_labels = ["Baseline", "Searched AuthBlock", "Fixed AuthBlock"]
group_labels = ["Baseline"] + ["N$_{AES}$="+str(i) for i in num_engines]

# Font sizes (reduced)
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

# X positions
x_positions = []
group_centers = []

# First two standalone bars
x_positions.append(0)
group_centers.append(0)

# Remaining crypto groups: 2 bars each
center = 1.5
for _ in range(len(num_engines)):
    x_positions.append(center - bar_width/2)
    x_positions.append(center + bar_width/2)
    group_centers.append(center)
    center += 1.5

# Plot
fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(x_positions, latencies_norm, width=bar_width,
              color=(["#D0D0D0"]) + color_list * len(num_engines),
              edgecolor='k')

# X ticks
ax.set_xticks(group_centers)
ax.set_xticklabels(group_labels, fontsize=28, rotation=0)

# Labels and limits
ax.set_ylabel("Normalized latency", fontsize=26)
ax.set_ylim([0, max(latencies_norm) * 1.2])
ax.tick_params(axis='y', labelsize=26)

# Annotate bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}' if height < 10 else f'{height:.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=20, fontweight='bold')

# Legend
handles = [plt.Rectangle((0, 0), 1, 1, color=c, edgecolor='k') for c in color_list_ext]
ax.legend(handles, method_labels, fontsize=24,
          loc='best', ncol=3, handletextpad=0.4, columnspacing=0.3, borderpad=0.2)

# Horizontal reference line at y=1
ax.axhline(y=1, color='black', linestyle='dotted', linewidth=1)

# Save and show
plt.tight_layout()
plt.savefig("crypto_engine_scaling.pdf", bbox_inches="tight", transparent=True)
plt.show()
