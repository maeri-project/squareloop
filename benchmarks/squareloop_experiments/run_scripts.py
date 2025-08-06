import time
import subprocess
import yaml
import copy
import numpy as np

timeloop_dir = '/home/workspace/timeloop/'
squareloop_dir = '/home/workspace/squareloop/'

timeloop_ld_path = 'LD_LIBRARY_PATH=' + timeloop_dir + 'build:$LD_LIBRARY_PATH'
timeloop_path = timeloop_dir + 'build/'
timeloop_mapper = timeloop_path + 'timeloop-mapper'
timeloop_model = timeloop_path + 'timeloop-model'

squareloop_ld_path = 'LD_LIBRARY_PATH=' + squareloop_dir + 'build:$LD_LIBRARY_PATH'
squareloop_path = squareloop_dir + 'build/'
squareloop_mapper = squareloop_path + 'timeloop-mapper'
squareloop_model = squareloop_path + 'timeloop-model'

exp_dir = squareloop_dir + 'benchmarks/squareloop_experiments/'

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
    'resnet18' : benchmarks_dir + 'layer_shapes/resnet18/resnet18_batch1_layer',
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
        layout_assign_perm(layout_in, layout_out, [('out','N','N'), ('out','V','L'), ('out','H','Q'), ('out','W','P')]),
        layout_assign_perm(layout_in, layout_out, [('in','N','N'), ('in','L','V'), ('in','Q','H'), ('in','P','W')]),
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







def generate_crypto(crypto_file_original, crypto_file_tmp, shared, number_engines):
    new_lines = ["\n  shared: " + shared,
                "\n  number-engines: " + str(number_engines)]

    with open(crypto_file_original, 'r') as f:
        lines = f.readlines()

    while lines and lines[-1].strip() == '':
        lines.pop()
    lines = lines[:-2]
    lines.extend(new_lines)

    with open(crypto_file_tmp, 'w') as f:
        f.writelines(lines)
        

def generate_block_size_arch(arch_file, arch_file_tmp, block_size):
    with open(arch_file, 'r') as f:
        lines = f.readlines()

    with open(arch_file_tmp, 'w') as f:
        for line in lines:
            if line.strip() == "block_size: 64":
                f.write(line.replace("64", str(block_size)))
            elif line.strip() == "width: 1024":
                f.write(line.replace("1024", str(block_size*16)))
            elif line.strip() == "memory_width: 1024":
                f.write(line.replace("1024", str(block_size*16)))
            elif line.strip() == "read_bandwidth: 64":
                f.write(line.replace("64", str(block_size)))
            elif line.strip() == "write_bandwidth: 64":
                f.write(line.replace("64", str(block_size)))
            elif line.strip() == "read_bandwidth: 128":
                f.write(line.replace("128", str(block_size*2)))
            elif line.strip() == "write_bandwidth: 128":
                f.write(line.replace("128", str(block_size*2)))
            else:
                f.write(line)










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




def run_squareloop(arch, model, layer, csv_type, result_dir, csv_file, layout_file='', mapping_file='', use_timeloop=False, shared=None, number_engines=None, block_size=None, no_crypto=False):
    no_crypto = no_crypto or use_timeloop

    print_str = csv_type + ' ' + arch + ' ' + model + ' ' + 'layer ' + str(layer)
    if shared != None and shared != '':
        print_str += ' ' + shared + ' ' + str(number_engines)
    if block_size != None and block_size != '':
        print_str += ' ' + str(block_size)
    print(print_str)

    crypto_file_tmp = crypto_file
    if shared != None and shared != '':
        crypto_file_tmp = 'crypto.yaml'
        generate_crypto(crypto_file, crypto_file_tmp, shared, number_engines)
    elif no_crypto:
        crypto_file_tmp = ''

    arch_file = arch_path_no_constraint[arch]
    if block_size != None and block_size != '':
        arch_file = 'arch.yaml'
        generate_block_size_arch(arch_path_single[arch], arch_file, block_size)
        arch_file += ' ' + arch_path_components[arch]
    if not mapping_file:
        arch_file += ' '
        arch_file += arch_path_constraints_depthwise[arch] if is_layer_depthwise(model, layer) else arch_path_constraints[arch]

    workload_file = model_path[model] + str(layer) + '.yaml'

    mapper_file_tmp = mapper_file if not mapping_file else ''

    squareloop_exe = squareloop_model if mapping_file else squareloop_mapper
    timeloop_exe = timeloop_model if mapping_file else timeloop_mapper
    ld_path_tmp = timeloop_ld_path if use_timeloop else squareloop_ld_path
    exe_tmp = timeloop_exe if use_timeloop else squareloop_exe
    
    mapper_command = ld_path_tmp + ' ' + exe_tmp + ' ' + mapper_file_tmp + ' ' + arch_file + ' ' + workload_file + ' ' + crypto_file_tmp + ' ' + layout_file + ' ' + mapping_file
    #print(mapper_command)

    start = time.time()
    result = subprocess.run(mapper_command, capture_output=True, text=True, shell=True)
    end = time.time()

    layout_file_write = result_dir + 'layout_' + csv_type + '_' + arch + '_' + model + '_' + str(layer)
    if shared != None and shared != '':
        layout_file_write += '_' + str(shared) + '_' + str(number_engines)
    if block_size != None and block_size != '':
        layout_file_write += '_' + str(block_size)
    layout_file_write += '.yaml'
    if layout_file:
        subprocess.run('cp ' + layout_file + ' ' + layout_file_write, capture_output=False, shell=True)
    elif not use_timeloop:
        subprocess.run('cp timeloop-mapper.layout.yaml ' + layout_file_write, capture_output=False, shell=True)

    mapping_file_write = result_dir + 'mapping_' + csv_type + '_' + arch + '_' + model + '_' + str(layer)
    if shared != None and shared != '':
        mapping_file_write += '_' + str(shared) + '_' + str(number_engines)
    if block_size != None and block_size != '':
        mapping_file_write += '_' + str(block_size)
    mapping_file_write += '.yaml'
    if mapping_file:
        subprocess.run('cp ' + mapping_file + ' ' + mapping_file_write, capture_output=False, shell=True)
    else:
        subprocess.run('cp timeloop-mapper.map.yaml ' + mapping_file_write, capture_output=False, shell=True)

    energy, latency = extract_energy_latency()
    wall_time = end - start

    csv_str = csv_type + ', ' + arch + ', ' + model + ', ' + str(layer) + ', ' 
    if shared != None:
        csv_str += str(shared) + ', ' + str(number_engines) + ', '
    if block_size != None:
        csv_str += str(block_size) + ', '
    csv_str += str(energy) + ', ' + str(latency) + ', ' + f"{wall_time:.3f}" + '\n'
    with open(csv_file, 'a') as f:
        f.write(csv_str)










def run_Timeloop1Layer_exp():
    print("Timeloop1Layer")

    result_dir = exp_dir + 'results/Timeloop1Layer/'

    csv_file = result_dir + 'stats.csv'
    with open(csv_file, 'w') as f:
        csv_header = "Type, Architecture, Model, Layer, Energy, Latency, Wall time\n"
        f.write(csv_header)

    for arch in arch_path_single:
        for model in model_path:
            num_layers = model_num_layers[model]
            for layer in range(1, num_layers+1):
                run_squareloop(arch, model, layer, 'Timeloop1Layer', result_dir, csv_file, use_timeloop=True)


def run_Squareloop1Layer_exp():
    print("Squareloop1Layer")

    result_dir = exp_dir + 'results/Squareloop1Layer/'

    csv_file = result_dir + 'stats.csv'
    with open(csv_file, 'w') as f:
        csv_header = "Type, Architecture, Model, Layer, Energy, Latency, Wall time\n"
        f.write(csv_header)

    for arch in arch_path_single:
        for model in model_path:
            for layer in unique_layers[model]:
            #num_layers = model_num_layers[model]
            #for layer in range(1, num_layers+1):
                run_squareloop(arch, model, layer, 'Squareloop1Layer', result_dir, csv_file)


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
    shared_options = ['false', 'true']
    number_engines_options = [1, 2, 4, 8, 16, 32, 64, 128]
    run_squareloop(arch, model, layer, 'TimeLoop', result_dir, csv_file, shared='', number_engines='', use_timeloop=True)
    run_squareloop(arch, model, layer, 'NoCrypto', result_dir, csv_file, shared='', number_engines='', no_crypto=True)
    for shared in shared_options:
        for number_engines in number_engines_options:
            run_squareloop(arch, model, layer, 'Map', result_dir, csv_file, shared=shared, number_engines=number_engines)
    for shared in shared_options:
        for number_engines in number_engines_options:
            layout_file = result_dir + 'layout_' + 'Map' + '_' + arch + '_' + model + '_' + str(layer) + '_' + 'false' + '_' + str(1) + '.yaml'
            run_squareloop(arch, model, layer, 'RestrictLayout', result_dir, csv_file, shared=shared, number_engines=number_engines, layout_file=layout_file)
    for shared in shared_options:
        for number_engines in number_engines_options:
            layout_file = result_dir + 'layout_' + 'Map' + '_' + arch + '_' + model + '_' + str(layer) + '_' + 'false' + '_' + str(1) + '.yaml'
            mapping_file = result_dir + 'mapping_' + 'Map' + '_' + arch + '_' + model + '_' + str(layer) + '_' + 'false' + '_' + str(1) + '.yaml'
            run_squareloop(arch, model, layer, 'RestrictLayoutAndMapping', result_dir, csv_file, shared=shared, number_engines=number_engines, layout_file=layout_file, mapping_file=mapping_file)


def run_BlockSize_exp():
    print("BlockSize")

    result_dir = exp_dir + 'results/BlockSize/'

    csv_file = result_dir + 'stats.csv'
    with open(csv_file, 'w') as f:
        csv_header = "Type, Architecture, Model, Layer, Block size, Energy, Latency, Wall time\n"
        f.write(csv_header)


    arch = 'eyeriss'
    model = 'mobv3'
    layer = 4
    block_size_options = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    for block_size in block_size_options:
        run_squareloop(arch, model, layer, 'Map', result_dir, csv_file, block_size=block_size)
    for block_size in block_size_options:
        run_squareloop(arch, model, layer, 'NoCrypto', result_dir, csv_file, block_size=block_size, no_crypto=True)


def run_LayerPairs_exp():
    print("LayerPairs")

    result_dir = exp_dir + 'results/LayerPairs/'

    layouts_dir = exp_dir + 'results/Squareloop1Layer/'

    csv_file = result_dir + 'stats.csv'
    with open(csv_file, 'w') as f:
        csv_header = "Type, Architecture, Model, Layer, Energy, Latency, Wall time\n"
        f.write(csv_header)


    archs = ['eyeriss']
    models = ['resnet18', 'mobv3']
    layers = {
        'resnet18' : [(2,2), (2,2), (6,7), (7,7), (11,12), (12,12), (16,17), (17,17)],
        'mobv3' : [(1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (15, 16), (16, 17), (17, 18), (18, 19), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (31, 32), (32, 33), (31, 32), (32, 33), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (42, 48), (48, 44), (44, 45), (45, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (52, 53), (53, 54), (54, 55), (55, 56)],
    }

    for arch in archs:
        for model in models:
            for layer_in, layer_out in layers[model]:

                layout_tmp = 'layout.yaml'

                layout_in = layouts_dir + 'layout_' + 'Squareloop1Layer' + '_' + arch + '_' + model + '_' + str(layer_in) + '.yaml'
                run_squareloop(arch, model, layer_in, 'MapRehashIn', result_dir, csv_file, layout_file=layout_in)

                layout_out = layouts_dir + 'layout_' + 'Squareloop1Layer' + '_' + arch + '_' + model + '_' + str(layer_out) + '.yaml'
                run_squareloop(arch, model, layer_out, 'MapRehashOut', result_dir, csv_file, layout_file=layout_out)

                print("Rehash", arch, model, "layer", str(layer_in)+'_'+str(layer_out))
                rehash_cost = rehash_latency(layout_in, layout_out) 
                csv_str = 'Rehash' + ', ' + arch + ', ' + model + ', ' + str(layer_in)+'_'+str(layer_out) + ', ' + 'N/A' + ', ' + str(rehash_cost) + ', ' + 'N/A' + '\n'
                with open(csv_file, 'a') as f:
                    f.write(csv_str)

                prop_layouts, csv_names = proposed_layouts(layout_in, layout_out)
                for (layout_new_in, layout_new_out), csv_name in zip(prop_layouts, csv_names):
                    write_layout(layout_tmp, layout_new_in)
                    run_squareloop(arch, model, layer_in, csv_name+'In', result_dir, csv_file, layout_file=layout_tmp)

                    write_layout(layout_tmp, layout_new_out)
                    run_squareloop(arch, model, layer_out, csv_name+'Out', result_dir, csv_file, layout_file=layout_tmp)











#run_Timeloop1Layer_exp()

run_Squareloop1Layer_exp()

#run_NumberEngines_exp()

#run_BlockSize_exp()

#run_LayerPairs_exp()