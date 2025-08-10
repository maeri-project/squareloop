import sys
import os

sys.path.append(os.path.dirname(__file__))
from utils import *

def run_NumberEngines_exp():
    print("NumberEngines")

    result_dir = exp_dir + 'results/NumberEngines/'

    csv_file = result_dir + 'stats.csv'
    with open(csv_file, 'w') as f:
        csv_header = "Type, Architecture, Model, Layer, Shared crypto engine, Number of crypto engines, Energy, Latency, Wall time\n"
        f.write(csv_header)


    arch = 'eyeriss'
    model = 'mobv3'
    layer = 4
    shared_options = ['false']
    number_engines_options = [1, 2, 4, 8, 16, 32, 64]
    run_squareloop(arch, model, layer, 'NoCrypto', result_dir, csv_file, shared='', number_engines='', no_crypto=True)
    for shared in shared_options:
        for number_engines in number_engines_options:
            run_squareloop(arch, model, layer, 'Map', result_dir, csv_file, shared=shared, number_engines=number_engines)
    for shared in shared_options:
        for number_engines in number_engines_options:
            layout_file = result_dir + 'layout_' + 'Map' + '_' + arch + '_' + model + '_' + str(layer) + '_' + 'false' + '_' + str(1) + '.yaml'
            run_squareloop(arch, model, layer, 'RestrictLayout', result_dir, csv_file, shared=shared, number_engines=number_engines, layout_file=layout_file)


run_NumberEngines_exp()