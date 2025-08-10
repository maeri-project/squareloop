import sys
import os

sys.path.append(os.path.dirname(__file__))
from utils import *

def run_Squareloop1Layer_exp():
    print("InterlayerInitialSearch")

    result_dir = exp_dir + 'results/InterlayerInitialSearch/'

    csv_file = result_dir + 'stats.csv'
    if not os.path.exists(csv_file):
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        with open(csv_file, 'w') as f:
            csv_header = "Type, Architecture, Model, Layer, Energy, Latency, Wall time\n"
            f.write(csv_header)

    archs = ['eyeriss', 'systolic', 'vector256']
    for arch in archs:
        for model in model_path:
            for layer in unique_layers[model]:
                run_squareloop(arch, model, layer, 'Squareloop1Layer', result_dir, csv_file)


run_Squareloop1Layer_exp()