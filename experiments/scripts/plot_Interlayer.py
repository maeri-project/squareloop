import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import sys
import os

sys.path.append(os.path.dirname(__file__))
from utils import squareloop_dir

archs = ['eyeriss', 'systolic', 'vector256']
models = ['resnet18', 'mobv3']


wall_times = []


for arch in archs:
    for model in models:
        result_folder = squareloop_dir+'experiments/results/Interlayer/'+arch+'_'+model+'/'

        df = pd.read_csv(result_folder+'stats.csv', skipinitialspace=True)


        baseline = df[df['Type'] == 'Baseline']['Latency'].tolist()
        baseline_rehash = df[df['Type'] == 'BaselineRehash']['Latency'].tolist() + [0]
        baseline_total = df[df['Type'] == 'BaselineTotal']['Latency'].item()

        constrained = df[df['Type'] == 'Constrained']['Latency'].tolist()
        constrained_idx = df[df['Type'] == 'Constrained']['Layer'].tolist()
        constrained_idx = [int(i) for i in constrained_idx]
        constrained_idx, constrained = zip(*sorted(zip(constrained_idx, constrained)))
        constrained = list(constrained)

        constrained_rehash_raw = df[df['Type'] == 'ConstrainedRehash']['Latency'].tolist()
        constrained_rehash_raw += df[df['Type'] == 'ConstrainedRehashForced']['Latency'].tolist()
        constrained_rehash_raw_idx = df[df['Type'] == 'ConstrainedRehash']['Layer'].tolist()
        constrained_rehash_raw_idx += df[df['Type'] == 'ConstrainedRehashForced']['Layer'].tolist()
        constrained_rehash_raw_idx = [int(i.split('_')[0]) for i in constrained_rehash_raw_idx]
        constrained_rehash_dict = {idx : val for idx, val in zip(constrained_rehash_raw_idx, constrained_rehash_raw)}
        constrained_rehash = [constrained_rehash_dict[i] if i in constrained_rehash_dict else 0 for i in range(1, len(constrained))] + [0]

        constrained_total = df[df['Type'] == 'ConstrainedTotal']['Latency'].item()

        forced_rehash_layers = df[df['Type'] == 'ConstrainedRehashForced']['Layer'].tolist()
        forced_rehash_layers = [int(i.split('_')[0]) for i in forced_rehash_layers]

        better_rehash_layers = df[df['Type'] == 'ConstrainedRehash']['Layer'].tolist()
        better_rehash_layers = [int(i.split('_')[0]) for i in better_rehash_layers]

        total_time = df[df['Type'] == 'TotalTime']['Wall time'].item()
        wall_times.append(arch + "," + model + "," + str(total_time))

        #print(baseline)
        #print(baseline_rehash)
        #print(baseline_total)
        #print(constrained)
        #print(constrained_rehash)
        #print(constrained_total)










        sum_bars_baseline = [lat+rehash for lat, rehash in zip(baseline, baseline_rehash)]
        sum_bars_constrained = [lat+rehash for lat, rehash in zip(constrained, constrained_rehash)]



        all_layer_bars = [x for pair in zip(baseline, constrained) for x in pair]
        all_rehash_bars = [x for pair in zip(baseline_rehash, constrained_rehash) for x in pair]
        all_sum_bars = [x for pair in zip(sum_bars_baseline, sum_bars_constrained) for x in pair]

        num_groups = len(all_sum_bars) // 2


        # Define bar layout
        bar_width = 0.6
        color_list = ["#808080", "#990000"]
        method_labels = ["Layer latency", "Rehash"]
        group_labels = [str(i) for i in range(1, num_groups+1)]

        # X positions
        x_positions = []
        group_centers = []

        # Remaining crypto groups: 2 bars each
        center = 0
        for _ in range(num_groups):
            x_positions.append(center - bar_width/2)
            x_positions.append(center + bar_width/2)
            group_centers.append(center)
            center += 1.5

        # Plot
        #fig, ax = plt.subplots(figsize=(12, 7))
        fig, ax = plt.subplots(figsize=(num_groups*0.6, 10))
        bars1 = ax.bar(x_positions, all_layer_bars, width=bar_width,
                    color=color_list[0],
                    edgecolor='k', bottom=0)
        bars2 = ax.bar(x_positions, all_rehash_bars, width=bar_width,
                    color=color_list[1],
                    edgecolor='k', bottom=all_layer_bars)

        center += 1.5
        ax2 = ax.twinx()
        bar3 = ax2.bar(center - bar_width/2, baseline_total, width=bar_width,
            color="#D0D0D0",
            edgecolor='k')
        bar4 = ax2.bar(center + bar_width/2, constrained_total, width=bar_width,
            color="#D0D0D0",
            edgecolor='k')

        # X ticks
        ax.set_xlabel("Layer number", fontsize=34)
        ax.set_xlim([group_centers[0]-1.5, center+1.5])
        ax.set_xticks(group_centers+[center])
        #ax.set_xticklabels(group_labels, fontsize=28, rotation=0)
        ax.set_xticklabels(group_labels+['Total'], fontsize=34, rotation=0)

        for i, label in enumerate(ax.get_xticklabels()[:-1]):
            if i % 2 != 1:  
                label.set_visible(False)

        # Labels and limits
        ft_size = 40

        ax.set_ylabel("Latency (cycles)", fontsize=ft_size)
        ax.set_ylim([0, max(all_sum_bars) * 1.2])
        ax.tick_params(axis='y', labelsize=ft_size)
        ax.yaxis.get_offset_text().set_fontsize(30)

        ax2.set_ylabel('Total latency (cycles)', fontsize=ft_size)
        ax2.set_ylim([0, max(baseline_total, constrained_total) * 1.1])
        ax2.tick_params(axis='y', labelsize=ft_size)
        ax2.yaxis.get_offset_text().set_fontsize(30)

        # Legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, edgecolor='k') for c in color_list]
        #ax.legend(handles, method_labels, fontsize=24,
        #        loc='best', ncol=3, handletextpad=0.4, columnspacing=0.3, borderpad=0.2)
        legend_font = 36 if model == 'mobv3' else 34
        legend = ax.legend(handles, method_labels, fontsize=legend_font, loc='upper center',
                  handletextpad=0.4, columnspacing=0.3, borderpad=0.2, bbox_to_anchor=(0.37, 1.0))

        # Get legend x-coordinate
        bbox = legend.get_window_extent()
        inv = ax.transData.inverted()
        data_bbox = bbox.transformed(inv)
        legend_x = data_bbox.x0
        legend_y = data_bbox.y0

        # Draw horizontal lines using ax2
        ax2.axhline(y=baseline_total, xmin=(bar3[0].get_x()+bar3[0].get_width())/ax2.get_xlim()[1], color='black', linestyle='--')
        ax2.axhline(y=constrained_total, xmin=(bar4[0].get_x()+bar4[0].get_width())/ax2.get_xlim()[1], color='black', linestyle=':')

        improvement = 100 * (constrained_total - baseline_total ) / baseline_total
        improvement = ('-' if round(improvement)==0 and improvement<0 else '' if improvement<0 else '+')+str(round(improvement))+'%'
        ax2.text(
            bar4[0].get_x()-0.5, baseline_total, improvement,
            ha='center', va='bottom',
            fontsize=34, color='black',
            fontweight='bold'
        )

        arrow_h = 400000 * max(all_sum_bars)/5e6
        arrow_w = 0.70
        arrow_font = 30
        for i in range(num_groups):
            if (i+1) in forced_rehash_layers:
                ax.annotate('',
                        xy=(group_centers[i], sum_bars_constrained[i]), 
                        xytext=(group_centers[i]+arrow_w, sum_bars_constrained[i]+arrow_h),
                        arrowprops=dict(arrowstyle='simple', color='black'),
                        fontsize=arrow_font)

        for i in range(num_groups):
            if (i+1) in better_rehash_layers:
                ax.annotate('',
                        xy=(group_centers[i], sum_bars_baseline[i]), 
                        xytext=(group_centers[i]-arrow_w, sum_bars_baseline[i]+arrow_h),
                        arrowprops=dict(arrowstyle='simple', color='#990000'),
                        fontsize=arrow_font)

        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        #x_pos = x_lim[0]+1.5
        #x_pos = x_lim[1]-25
        if model == 'mobv3':
            x_pos = legend_x-17
            y_pos = (y_lim[0]+y_lim[1])*0.95 - arrow_h/2
        else:
            x_pos = legend_x
            y_pos = legend_y - arrow_h*1.5
        ax.annotate('',
           xy=(x_pos-arrow_w, y_pos),
           xytext=(x_pos, y_pos+arrow_h),
           arrowprops=dict(arrowstyle='simple', color='black'),
           fontsize=arrow_font)
        ax.text(x_pos+arrow_w/4, y_pos+arrow_h/3, r'Forced rehash due to', fontsize=legend_font, ha='left', va='center')
        y_pos -= arrow_h
        ax.text(x_pos-arrow_w, y_pos+arrow_h/3, r'dependency breaker', fontsize=legend_font, ha='left', va='center')
        y_pos -= arrow_h
        ax.text(x_pos-arrow_w, y_pos+arrow_h/3, r'(MaxPooling)', fontsize=legend_font, ha='left', va='center')
        #y_pos -= 2*arrow_h
        #ax.annotate('',
        #   xy=(x_pos, y_pos),
        #   xytext=(x_pos-arrow_w, y_pos+arrow_h),
        #   arrowprops=dict(arrowstyle='simple', color='#990000'),
        #   fontsize=12)
        #ax.text(x_pos, y_pos+arrow_h/3, r'Breaking dependency with rehash is optimal', fontsize=legend_font, ha='left', va='center')

        # Save and show
        plt.tight_layout()
        plt.savefig(result_folder+"Interlayer_"+arch+"_"+model+".pdf", bbox_inches="tight", transparent=True)
        #plt.show()


for s in wall_times:
    print(s)