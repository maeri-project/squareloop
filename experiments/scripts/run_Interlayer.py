import time
import os
import subprocess
import yaml
import copy
import shutil
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from utils import squareloop_dir

squareloop_ld_path = 'LD_LIBRARY_PATH=' + squareloop_dir + 'build:$LD_LIBRARY_PATH'
squareloop_path = squareloop_dir + 'build/'
squareloop_mapper = squareloop_path + 'timeloop-mapper'
squareloop_model = squareloop_path + 'timeloop-model'

exp_dir = squareloop_dir + 'experiments/'

benchmarks_dir = squareloop_dir + 'benchmarks/'

mapper_file = benchmarks_dir + 'mapper/mapper_squareloop.yaml'

crypto_file = benchmarks_dir + 'crypto/AES-GCM-parallel.yaml'

arch_path_no_constraint = {
    'eyeriss' : benchmarks_dir + 'arch_designs/eyeriss_like/arch/eyeriss_like.yaml ' + benchmarks_dir + 'arch_designs/eyeriss_like/arch/components/*',
    'systolic' : benchmarks_dir + 'arch_designs/vector_256.yaml',
    'vector256' : benchmarks_dir + 'arch_designs/vector_256.yaml',
}

arch_path_components = {
    'eyeriss' : benchmarks_dir + 'arch_designs/eyeriss_like/arch/components/*',
    'systolic' : '',
    'vector256' : '',
}

arch_path_constraints = {
    'eyeriss' : benchmarks_dir + 'arch_designs/eyeriss_like/constraints/*',
    'systolic' : benchmarks_dir + 'arch_designs/systolic_constraint/mapspace_XY_OS.yaml',
    'vector256' : '',
}

arch_path_constraints_depthwise = {
    'eyeriss' : benchmarks_dir + 'arch_designs/eyeriss_like/constraints_depthwise/*',
    'systolic' : benchmarks_dir + 'arch_designs/systolic_constraint_depthwise/mapspace_XY_OS.yaml',
    'vector256' : '',
}

arch_path_single = {
    'eyeriss' : benchmarks_dir + 'arch_designs/eyeriss_like/arch/eyeriss_like.yaml',
    'systolic' : benchmarks_dir + 'arch_designs/vector_256.yaml',
    'vector256' : benchmarks_dir + 'arch_designs/vector_256.yaml',
}

model_path = {
    'resnet18' : benchmarks_dir + 'layer_shapes/resnet18/resnet18_',
    'mobv3' : benchmarks_dir + 'layer_shapes/mobv3/mobilenet_v3_large_',
    'bert_conv' : benchmarks_dir + 'layer_shapes/bert_conv/bert_conv_layer',
}

model_num_layers = {
    'resnet18' : 21,
    'mobv3' : 62,
    'bert_conv' : 3,
}

unique_layers = {
    'resnet18' : [1, 2, 6, 7, 8, 11, 12, 13, 16, 17, 18, 21],
    'mobv3' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31, 32, 33, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 51, 52, 53, 54, 55, 56],
    'bert_conv' : [1, 2, 3],
}

depthwise_layers = {
    'resnet18' : [],
    'mobv3' : [2, 5, 8, 11, 16, 21, 26, 29, 32, 35, 38, 43, 48, 53, 58],
    'bert_conv' : [],
}


dependency_pairs_dict = {
    'resnet18' : [(2, 3), (4, 5), (6, 7), (9, 10), (11, 12), (14, 15), (16, 17), (19, 20)],
    'mobv3' : [(1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (20, 21), (21, 22), (22, 23), (23, 24), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (31, 32), (32, 33), (34, 35), (35, 36), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (57, 58), (58, 59), (59, 60), (60, 61)],
}

unique_layer_mapping_dict = {
    'resnet18' : [0, 1, 2, 2, 2, 2, 6, 7, 8, 7, 7, 11, 12, 13, 12, 12, 16, 17, 18, 17, 17, 21],
    'mobv3' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31, 32, 33, 31, 32, 33, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 42, 48, 44, 45, 51, 52, 53, 54, 55, 56, 52, 53, 54, 55, 56, 52],
}








def read_layout(filename):
    with open(filename, "r") as file:
        layout = yaml.safe_load(file)
    return layout['layout']

def read_mapping(filename):
    with open(filename, "r") as file:
        mapping = yaml.safe_load(file)
    return mapping['mapping']


def write_layout(filename, layout):
    with open(filename, 'w') as file:
        yaml.dump({'layout': layout}, file)


def factors2dict(s):
    return {k: int(v) for k, v in (pair.split('=') for pair in s.split())}

def dict2factors(d):
    return ' '.join(f"{k}={v}" for k, v in d.items())


def find_in_layout(layout, ftarget, ftype):
    for d in layout:
        if d['target'] == ftarget and d['type'] == ftype:
            return factors2dict(d['factors'])
    return {}

def assign_in_layout(layout, ftarget, ftype, res):
    for d in layout:
        if d['target'] == ftarget and d['type'] == ftype:
            d['factors'] = dict2factors(res)
            break

def is_layout_depthwise(layout):
    intra = find_in_layout(layout, 'MainMemory', 'intraline')
    return ('K' not in intra)







def layout_assign_perm(layout_in_raw, layout_out_raw, perm):
    layout_in = copy.deepcopy(layout_in_raw)
    layout_out = copy.deepcopy(layout_out_raw)
    intra_in = find_in_layout(layout_in, 'MainMemory', 'intraline')
    auth_in = find_in_layout(layout_in, 'MainMemory', 'authblock_lines')
    if len(auth_in) == 0:
        factors_default = 'C=1 R=1 S=1 N=1 V=1 H=1 W=1 L=1 Q=1 P=1' if is_layout_depthwise(layout_in) \
            else 'C=1 K=1 R=1 S=1 N=1 V=1 H=1 W=1 L=1 Q=1 P=1'
        permutation_default = 'CRSNVHWLQP'if is_layout_depthwise(layout_in) \
            else 'CKRSNVHWLQP'
        layout_in.append({'target':'MainMemory', 'type':'authblock_lines', 
                          'factors':factors_default,
                          'permutation':permutation_default})
        auth_in = find_in_layout(layout_in, 'MainMemory', 'authblock_lines')
    intra_out = find_in_layout(layout_out, 'MainMemory', 'intraline')
    auth_out = find_in_layout(layout_out, 'MainMemory', 'authblock_lines')
    if len(auth_out) == 0:
        factors_default = 'C=1 R=1 S=1 N=1 V=1 H=1 W=1 L=1 Q=1 P=1' if is_layout_depthwise(layout_out) \
            else 'C=1 K=1 R=1 S=1 N=1 V=1 H=1 W=1 L=1 Q=1 P=1'
        permutation_default = 'CRSNVHWLQP'if is_layout_depthwise(layout_out) \
            else 'CKRSNVHWLQP'
        layout_out.append({'target':'MainMemory', 'type':'authblock_lines', 
                          'factors':factors_default,
                          'permutation':permutation_default})
        auth_out = find_in_layout(layout_out, 'MainMemory', 'authblock_lines')

    for direction, a, b in perm:
        if direction == 'in':
            intra_in[a] = intra_out[b]
            auth_in[a] = auth_out[b]
        elif direction == 'out':
            intra_out[a] = intra_in[b]
            auth_out[a] = auth_in[b]

    assign_in_layout(layout_in, 'MainMemory', 'intraline', intra_in)
    assign_in_layout(layout_in, 'MainMemory', 'authblock_lines', auth_in)
    assign_in_layout(layout_out, 'MainMemory', 'intraline', intra_out)
    assign_in_layout(layout_out, 'MainMemory', 'authblock_lines', auth_out)

    return layout_in, layout_out




def proposed_layouts(layout_in_file, layout_out_file):
    layout_in = read_layout(layout_in_file)
    layout_out = read_layout(layout_out_file)
    layouts = [
        layout_assign_perm(layout_in, layout_out, [('out','N','N'), ('out','V','L'), ('out','H','P'), ('out','W','Q')]),
        layout_assign_perm(layout_in, layout_out, [('in','N','N'), ('in','L','V'), ('in','P','H'), ('in','Q','W')]),
    ]
    csv_names = [
        'In2Out',
        'Out2In',
    ]

    return layouts, csv_names







def read_crypto_config() -> int:
    crypto_config_path = crypto_file
    """
    Read crypto configuration file and extract auth-additional-cycle-per-block value.

    Args:
        crypto_config_path: Path to the crypto configuration YAML file

    Returns:
        int: auth-additional-cycle-per-block value, defaults to 1 if not found
    """
    auth_cycle_per_datapath = 0
    enc_cycle_per_datapath = 0
    auth_additional_cycle_per_block = 0
    datapath = 0
    try:
        with open(crypto_config_path, 'r') as f:
            crypto_data = yaml.safe_load(f)

        if 'crypto' in crypto_data and 'auth-additional-cycle-per-block' in crypto_data['crypto']:
            auth_additional_cycle_per_block = crypto_data['crypto']['auth-additional-cycle-per-block']
            auth_cycle_per_datapath = crypto_data['crypto']['auth-cycle-per-datapath']
            enc_cycle_per_datapath = crypto_data['crypto']['enc-cycle-per-datapath']
            datapath = crypto_data['crypto']['datapath']
        else:
            print(f"Warning: auth-additional-cycle-per-block not found in {crypto_config_path}, using default value 1")
        return auth_additional_cycle_per_block, auth_cycle_per_datapath, enc_cycle_per_datapath, datapath

    except FileNotFoundError:
        print(f"Warning: Crypto config file not found: {crypto_config_path}, using default auth_cycle_per_datapath=1")
        return 1
    except Exception as e:
        print(f"Error reading crypto config file {crypto_config_path}: {e}, using default auth_cycle_per_datapath=1")
        return 1


def rehash_latency(layout_in_file, layout_out_file):
    word_bits = 16
    auth_additional_cycle_per_block, auth_cycle_per_datapath, enc_cycle_per_datapath, datapath = read_crypto_config()
    layout_in = read_layout(layout_in_file)
    layout_out = read_layout(layout_out_file)

    total_rehash_latency = 0

    for layout, ranks in [(layout_in, ['N', 'L', 'P', 'Q']), (layout_out, ['N', 'V', 'H', 'W'])]:
        intra = find_in_layout(layout, 'MainMemory', 'intraline')
        inter = find_in_layout(layout, 'MainMemory', 'interline')
        auth = find_in_layout(layout, 'MainMemory', 'authblock_lines')

        authblock_lines_size = 1
        num_authblock_lines = 1
        for r in ranks:
            if r in auth:
                authblock_lines_size *= auth[r] * intra[r]
                num_authblock_lines *= np.ceil(inter[r] / auth[r])
            else:
                authblock_lines_size *= intra[r]
                num_authblock_lines *= inter[r]

        latency_per_authblock = np.ceil(authblock_lines_size * word_bits / datapath) * np.max(enc_cycle_per_datapath, auth_cycle_per_datapath) +  auth_additional_cycle_per_block
        total_rehash_latency += num_authblock_lines * latency_per_authblock
    
    return total_rehash_latency





def is_layer_depthwise(model, layer):
    return (layer in depthwise_layers[model])

















def extract_energy_latency(model=False):
    energy = None
    cycles = None
    filename = 'timeloop-model.stats.txt' if model else 'timeloop-mapper.stats.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == "Summary Stats":
                for j in range(i+1, len(lines)):
                    if lines[j].strip().startswith("Cycles:"):
                        parts = lines[j].strip().split()
                        cycles = int(parts[1])
                        break
                for j in range(i+1, len(lines)):
                    if lines[j].strip().startswith("Energy:"):
                        parts = lines[j].strip().split()
                        energy = float(parts[1])
                        break
                break
    if energy == None or cycles == None:
        print('ERROR: Could not extract energy or cycles')
        exit(1)
    return energy, cycles




def run_squareloop(arch, model, layer, csv_type, result_dir, csv_file='', layout_file='', mapping_file='', save=True):
    print_str = csv_type + ' ' + arch + ' ' + model + ' ' + 'layer ' + str(layer)
    print(print_str)

    arch_file = arch_path_no_constraint[arch]
    if not mapping_file:
        arch_file += ' '
        arch_file += arch_path_constraints_depthwise[arch] if is_layer_depthwise(model, layer) else arch_path_constraints[arch]

    workload_file = model_path[model] + str(layer) + '.yaml'

    mapper_file_tmp = mapper_file if not mapping_file else ''

    squareloop_exe = squareloop_model if mapping_file else squareloop_mapper
    
    mapper_command = squareloop_ld_path + ' ' + squareloop_exe + ' ' + mapper_file_tmp + ' ' + arch_file + ' ' + workload_file + ' ' + crypto_file + ' ' + layout_file + ' ' + mapping_file
    #print(mapper_command)

    start = time.time()
    result = subprocess.run(mapper_command, capture_output=True, text=True, shell=True)
    end = time.time()

    if save:
        layout_file_write = result_dir + 'layout_' + csv_type + '_' + arch + '_' + model + '_' + str(layer)
        layout_file_write += '.yaml'
        subprocess.run('cp ' + layout_file + ' ' + layout_file_write, capture_output=False, shell=True)

        mapping_file_write = result_dir + 'mapping_' + csv_type + '_' + arch + '_' + model + '_' + str(layer)
        mapping_file_write += '.yaml'
        if mapping_file:
            subprocess.run('cp ' + mapping_file + ' ' + mapping_file_write, capture_output=False, shell=True)
        else:
            subprocess.run('cp timeloop-mapper.map.yaml ' + mapping_file_write, capture_output=False, shell=True)

    energy, latency = extract_energy_latency()
    wall_time = end - start

    if save:
        csv_str = csv_type + ', ' + arch + ', ' + model + ', ' + str(layer) + ', ' 
        csv_str += str(energy) + ', ' + str(latency) + ', ' + f"{wall_time:.3f}" + '\n'
        with open(csv_file, 'a') as f:
            f.write(csv_str)
    
    return latency









def run_Interlayer_exp(arch, model):
    startInterlayer = time.time()
    print("LayerPairs")

    result_dir = exp_dir + 'results/Interlayer/'+arch+'_'+model+'/'
    os.makedirs(result_dir, exist_ok=True)

    baseline_layouts_dir = exp_dir + 'results/InterlayerInitialSearch/'

    csv_file = result_dir + 'stats.csv'
    with open(csv_file, 'w') as f:
        csv_header = "Type, Architecture, Model, Layer, Energy, Latency, Wall time\n"
        f.write(csv_header)

    dependency_pairs = dependency_pairs_dict[model]


    # Baseline
    total_baseline_latency = 0

    for i in range(1, model_num_layers[model]+1):
        unique_layer_i = unique_layer_mapping_dict[model][i]
        baseline_layout = baseline_layouts_dir + 'layout_' + 'Squareloop1Layer' + '_' + arch + '_' + model + '_' + str(unique_layer_i) + '.yaml'
        total_baseline_latency += run_squareloop(arch, model, i, 'Baseline', result_dir, csv_file=csv_file, layout_file=baseline_layout)

    def get_baseline_layout_file(layer):
        return result_dir + 'layout_' + 'Baseline' + '_' + arch + '_' + model + '_' + str(layer) + '.yaml'

    #for i, j in dependency_pairs:
    for i in range(1, model_num_layers[model]):
        j = i+1
        print("BaselineRehash", arch, model, "layer", str(i)+'_'+str(j))
        rehash_cost = rehash_latency(get_baseline_layout_file(i), get_baseline_layout_file(j)) 
        total_baseline_latency += rehash_cost
        csv_str = 'BaselineRehash' + ', ' + arch + ', ' + model + ', ' + str(i)+'_'+str(j) + ', ' + 'N/A' + ', ' + str(rehash_cost) + ', ' + 'N/A' + '\n'
        with open(csv_file, 'a') as f:
            f.write(csv_str)

    csv_str = 'BaselineTotal' + ', ' + arch + ', ' + model + ', ' + 'N/A' + ', ' + 'N/A' + ', ' + str(total_baseline_latency) + ', ' + 'N/A' + '\n'
    with open(csv_file, 'a') as f:
        f.write(csv_str)


    # Constrained
    def best_chain_latency(ordering, save=False):
        ordering = [i for i in ordering]
        print("Ordering", ordering, "write =", save)

        pairs = [(i,i+1) for i in ordering if i+1 in ordering]
        
        tmp_dir = './tmp_layouts/'
        os.makedirs(tmp_dir, exist_ok=True)
        layout_tmp = tmp_dir + 'layout_tmp.yaml'
        def tmp_layout(layer):
            return tmp_dir + 'layout' + str(layer) + '.yaml'

        for i in ordering:
            shutil.copy(get_baseline_layout_file(i), tmp_layout(i))
        
        rehash_bitmask = []
        for i, j in pairs:
            baseline = 0
            baseline += run_squareloop(arch, model, i, 'TestChainBaseline', result_dir, layout_file=tmp_layout(i), save=False)
            baseline += run_squareloop(arch, model, j, 'TestChainBaseline', result_dir, layout_file=tmp_layout(j), save=False)
            print("TestChainBaselineRehash", arch, model, "layer", str(i)+'_'+str(j))
            baseline += rehash_latency(tmp_layout(i), tmp_layout(j)) 

            prop_layouts, csv_names = proposed_layouts(tmp_layout(i), tmp_layout(j))
            prop_latencies = []
            for (layout_new_in, layout_new_out), csv_name in zip(prop_layouts, csv_names):
                lat = 0
                write_layout(layout_tmp, layout_new_in)
                lat += run_squareloop(arch, model, i, 'TestChain'+csv_name+'In', result_dir, layout_file=layout_tmp, save=False)
                write_layout(layout_tmp, layout_new_out)
                lat += run_squareloop(arch, model, j, 'TestChain'+csv_name+'Out', result_dir, layout_file=layout_tmp, save=False)
                prop_latencies.append(lat)
            
            if baseline < min(prop_latencies):
                rehash_bitmask.append(True)
            else:
                rehash_bitmask.append(False)
                best_prop_idx = prop_latencies.index(min(prop_latencies))
                write_layout(tmp_layout(i), prop_layouts[best_prop_idx][0])
                write_layout(tmp_layout(j), prop_layouts[best_prop_idx][1])
        
        total_latency = 0
        for i in ordering:
            total_latency += run_squareloop(arch, model, i, 'Constrained', result_dir, csv_file=csv_file, layout_file=tmp_layout(i), save=save)
        for (i, j), do_rehash in zip(pairs, rehash_bitmask):
            if do_rehash:
                print("ConstrainedRehash", arch, model, "layer", str(i)+'_'+str(j))
                rehash_cost = rehash_latency(tmp_layout(i), tmp_layout(j))
                total_latency += rehash_cost
                if save:
                    csv_str = 'ConstrainedRehash' + ', ' + arch + ', ' + model + ', ' + str(i)+'_'+str(j) + ', ' + 'N/A' + ', ' + str(rehash_cost) + ', ' + 'N/A' + '\n'
                    with open(csv_file, 'a') as f:
                        f.write(csv_str)
        
        return total_latency

    best_total_latency = 0
    b = 1
    while b <= model_num_layers[model]:
        e = b
        while (e, e+1) in dependency_pairs:
            e += 1

        if b == e:   # layer without dependency
            baseline_layout = get_baseline_layout_file(b)
            best_total_latency += run_squareloop(arch, model, b, 'Constrained', result_dir, csv_file=csv_file, layout_file=baseline_layout)

        else:   # dependency chain from b to e
            print("Chain [" + str(b) + "," + str(e) + "]")
            if e-b > 1:
                orderings = [range(b, e+1), range(e, b-1, -1)]
                ordering_latencies = [best_chain_latency(o, save=False) for o in orderings]
                best_ordering_idx = ordering_latencies.index(min(ordering_latencies))
                best_ordering = orderings[best_ordering_idx]
            else:
                best_ordering = range(b, e+1)
            best_total_latency += best_chain_latency(best_ordering, save=True)

        if e+1 <= model_num_layers[model]:   # rehash for last element in chain (regardless if chain has 1 layer or more)
            print("ConstrainedRehashForced", arch, model, "layer", str(e)+'_'+str(e+1))
            rehash_cost = rehash_latency(get_baseline_layout_file(e), get_baseline_layout_file(e+1)) 
            best_total_latency += rehash_cost
            csv_str = 'ConstrainedRehashForced' + ', ' + arch + ', ' + model + ', ' + str(e)+'_'+str(e+1) + ', ' + 'N/A' + ', ' + str(rehash_cost) + ', ' + 'N/A' + '\n'
            with open(csv_file, 'a') as f:
                f.write(csv_str)

        b = e + 1

    csv_str = 'ConstrainedTotal' + ', ' + arch + ', ' + model + ', ' + 'N/A' + ', ' + 'N/A' + ', ' + str(best_total_latency) + ', ' + 'N/A' + '\n'
    with open(csv_file, 'a') as f:
        f.write(csv_str)

    endInterlayer = time.time()
    wall_time = endInterlayer - startInterlayer
    csv_str = 'TotalTime' + ', ' + arch + ', ' + model + ', ' + 'N/A' + ', ' + 'N/A' + ', ' + 'N/A' + ', ' + f"{wall_time:.3f}" + '\n'
    with open(csv_file, 'a') as f:
        f.write(csv_str)
















archs = ['eyeriss', 'systolic', 'vector256']
models = ['resnet18', 'mobv3']
for arch in archs:
    for model in models:
        run_Interlayer_exp(arch, model)