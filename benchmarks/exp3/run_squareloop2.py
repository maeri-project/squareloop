import time
import subprocess

timeloop_regular_ld_path = 'LD_LIBRARY_PATH=/home/ubuntu/repo/timeloop/build:$LD_LIBRARY_PATH'
timeloop_regular_path = '/home/ubuntu/repo/timeloop/build/'
timeloop_regular_mapper = timeloop_regular_path + 'timeloop-mapper'
timeloop_regular_model = timeloop_regular_path + 'timeloop-model'

timeloop_ld_path = 'LD_LIBRARY_PATH=/home/ubuntu/repo/squareloop/build:$LD_LIBRARY_PATH'
timeloop_path = '/home/ubuntu/repo/squareloop/build/'
timeloop_mapper = timeloop_path + 'timeloop-mapper'
timeloop_model = timeloop_path + 'timeloop-model'

exp1_dir = '/home/ubuntu/repo/squareloop/benchmarks/exp3/'

csv_file = exp1_dir + 'result/stats.csv'

mapper_file = exp1_dir + 'mapper.yaml'

crypto_file = exp1_dir + 'AES-GCM-parallel.yaml'

new_arch_file = exp1_dir + 'arch.yaml'

arch_path = {
    'eyeriss' : exp1_dir + 'arch/eyeriss_like/arch/eyeriss_like.yaml ' + exp1_dir + 'arch/eyeriss_like/arch/components/* ' + exp1_dir + 'arch/eyeriss_like/constraints/*',
    'systolic' : exp1_dir + 'arch/systolic_array/*',
    'vector256' : exp1_dir + 'arch/vector_256.yaml',
}

arch_other_path = {
    'eyeriss' : exp1_dir + 'arch/eyeriss_like/arch/components/* ' + exp1_dir + 'arch/eyeriss_like/constraints/*',
    'systolic' : exp1_dir + 'arch/systolic_array/mapspace_XY_OS.yaml',
    'vector256' : '',
}

arch_path_no_constraint = {
    'eyeriss' : exp1_dir + 'arch/eyeriss_like/arch/eyeriss_like.yaml ' + exp1_dir + 'arch/eyeriss_like/arch/components/* ',
    'systolic' : exp1_dir + 'arch/systolic_array/*',
    'vector256' : exp1_dir + 'arch/vector_256.yaml',
}

arch_path_single = {
    'eyeriss' : exp1_dir + 'arch/eyeriss_like/arch/eyeriss_like.yaml',
    'systolic' : exp1_dir + 'arch/systolic_array/vector_256.yaml',
    'vector256' : exp1_dir + 'arch/vector_256.yaml',
}

model_path = {
    'resnet18' : exp1_dir + 'workload/resnet18/resnet18_batch1_layer',
    'mobv3' : exp1_dir + 'workload/mobv3/mobilenet_v3_large_',
}

model_num_layers = {
    'resnet18' : 21,
    'mobv3' : 62,
}


#def extract_energy_latency(output, line=-3):
#    last_line = output.stdout.strip().split('\n')[line]
#    parts = last_line.split('|')
#
#    pj_compute_str = parts[1].strip().split('=')[1].strip()
#    cycles_str = parts[2].strip().split('=')[1].strip()
#
#    pj_per_compute = float(pj_compute_str)
#    cycles = int(cycles_str)
#    return pj_per_compute, cycles
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
        print('ERROR: none in energy&cycle extract')
        exit(1)
    return energy, cycles


def generate_block_size_arch(arch_file, block_size):
    with open(arch_file, 'r') as f:
        lines = f.readlines()

    with open(new_arch_file, 'w') as f:
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


def run_mapper(arch, model, layer, block_size):
    generate_block_size_arch(arch_path_single[arch], block_size);

    arch_file = new_arch_file + ' ' + arch_other_path[arch]
    workload_file = model_path[model] + str(layer) + '.yaml'
    
    mapper_command = timeloop_ld_path + ' ' + timeloop_mapper + ' ' + mapper_file + ' ' + arch_file + ' ' + workload_file + ' ' + crypto_file

    start = time.time()
    result = subprocess.run(mapper_command, capture_output=True, text=True, shell=True)
    end = time.time()

    mapping_file = exp1_dir + 'result/mapping_' + arch + '_' + model + '_' + str(layer) + '_' + str(block_size) + '.yaml'
    #mapping_file = exp1_dir + 'result/mapping_' + arch + '_' + model + '_' + str(layer) + '.yaml'
    subprocess.run('cp timeloop-mapper.map.yaml ' + mapping_file, capture_output=False, shell=True)

    layout_file = exp1_dir + 'result/layout_' + arch + '_' + model + '_' + str(layer) + '_' + str(block_size) + '.yaml'
    #layout_file = exp1_dir + 'result/layout_' + arch + '_' + model + '_' + str(layer) + '.yaml'
    subprocess.run('cp timeloop-mapper.layout.yaml ' + layout_file, capture_output=False, shell=True)

    energy, latency = extract_energy_latency()
    wall_time = end - start

    csv_str = 'mapper, ' + arch + ', ' + model + ', ' + str(layer) + ', ' + str(block_size) + ', ' + str(energy) + ', ' + str(latency) + ', ' + f"{wall_time:.3f}" + '\n'
    with open(csv_file, 'a') as f:
        f.write(csv_str)


def run_mapper_no_crypto(arch, model, layer, block_size):
    generate_block_size_arch(arch_path_single[arch], block_size);

    arch_file = new_arch_file + ' ' + arch_other_path[arch]
    workload_file = model_path[model] + str(layer) + '.yaml'
    
    mapper_command = timeloop_ld_path + ' ' + timeloop_mapper + ' ' + mapper_file + ' ' + arch_file + ' ' + workload_file 

    start = time.time()
    result = subprocess.run(mapper_command, capture_output=True, text=True, shell=True)
    end = time.time()

    mapping_file = exp1_dir + 'result/mapping_nocrypto_' + arch + '_' + model + '_' + str(layer) + '_' + str(block_size) + '.yaml'
    #mapping_file = exp1_dir + 'result/mapping_' + arch + '_' + model + '_' + str(layer) + '.yaml'
    subprocess.run('cp timeloop-mapper.map.yaml ' + mapping_file, capture_output=False, shell=True)

    layout_file = exp1_dir + 'result/layout_nocrypto_' + arch + '_' + model + '_' + str(layer) + '_' + str(block_size) + '.yaml'
    #layout_file = exp1_dir + 'result/layout_' + arch + '_' + model + '_' + str(layer) + '.yaml'
    subprocess.run('cp timeloop-mapper.layout.yaml ' + layout_file, capture_output=False, shell=True)

    energy, latency = extract_energy_latency()
    wall_time = end - start

    csv_str = 'no_crypto, ' + arch + ', ' + model + ', ' + str(layer) + ', ' + str(block_size) + ', ' + str(energy) + ', ' + str(latency) + ', ' + f"{wall_time:.3f}" + '\n'
    with open(csv_file, 'a') as f:
        f.write(csv_str)


#def run_mapper_regular(arch, model, layer):
#    arch_file = arch_path[arch]
#    workload_file = model_path[model] + str(layer) + '.yaml'
#    
#    mapper_command = timeloop_regular_ld_path + ' ' + timeloop_regular_mapper + ' ' + mapper_file + ' ' + arch_file + ' ' + workload_file 
#
#    start = time.time()
#    result = subprocess.run(mapper_command, capture_output=True, text=True, shell=True)
#    end = time.time()
#
#    mapping_file = exp1_dir + 'result/mapping_timeloop_' + arch + '_' + model + '_' + str(layer) + '.yaml'
#    #mapping_file = exp1_dir + 'result/mapping_' + arch + '_' + model + '_' + str(layer) + '.yaml'
#    subprocess.run('cp timeloop-mapper.map.yaml ' + mapping_file, capture_output=False, shell=True)
#
#    layout_file = exp1_dir + 'result/layout_timeloop_' + arch + '_' + model + '_' + str(layer) + '.yaml'
#    #layout_file = exp1_dir + 'result/layout_' + arch + '_' + model + '_' + str(layer) + '.yaml'
#    subprocess.run('cp timeloop-mapper.layout.yaml ' + layout_file, capture_output=False, shell=True)
#
#    energy, latency = extract_energy_latency()
#    wall_time = end - start
#
#    csv_str = 'timeloop, ' + arch + ', ' + model + ', ' + str(layer) + ', ' + 'N/A' + ', ' + 'N/A' + ', ' + str(energy) + ', ' + str(latency) + ', ' + f"{wall_time:.3f}" + '\n'
#    with open(csv_file, 'a') as f:
#        f.write(csv_str)
#
#
#def run_mapper_restrict_layout(arch, model, layer, shared, number_engines):
#    generate_crypto(shared, number_engines)
#
#    arch_file = arch_path[arch]
#    workload_file = model_path[model] + str(layer) + '.yaml'
#
#    layout_file = exp1_dir + 'result/layout_' + arch + '_' + model + '_' + str(layer) + '_' + 'false' + '_' + str(1) + '.yaml'
#    
#    mapper_command = timeloop_ld_path + ' ' + timeloop_mapper + ' ' + mapper_file + ' ' + arch_file + ' ' + workload_file + ' ' + crypto_file + ' ' + layout_file
#
#    start = time.time()
#    result = subprocess.run(mapper_command, capture_output=True, text=True, shell=True)
#    end = time.time()
#
#    mapping_file = exp1_dir + 'result/mappingrestricted_' + arch + '_' + model + '_' + str(layer) + '_' + shared + '_' + str(number_engines) + '.yaml'
#    #mapping_file = exp1_dir + 'result/mapping_' + arch + '_' + model + '_' + str(layer) + '.yaml'
#    subprocess.run('cp timeloop-mapper.map.yaml ' + mapping_file, capture_output=False, shell=True)
#
#    energy, latency = extract_energy_latency()
#    wall_time = end - start
#
#    csv_str = 'mapper_restricted_layout, ' + arch + ', ' + model + ', ' + str(layer) + ', ' + shared + ', ' + str(number_engines) + ', ' + str(energy) + ', ' + str(latency) + ', ' + f"{wall_time:.3f}" + '\n'
#    with open(csv_file, 'a') as f:
#        f.write(csv_str)
#
#
#def run_model(arch, model, layer, shared, number_engines):
#    generate_crypto(shared, number_engines)
#
#    arch_file = arch_path_no_constraint[arch]
#    workload_file = model_path[model] + str(layer) + '.yaml'
#
#    #mapping_file = exp1_dir + 'result/mapping_' + arch + '_' + model + '_' + str(layer) + '.yaml'
#    #layout_file = exp1_dir + 'result/layout_' + arch + '_' + model + '_' + str(layer) + '.yaml'
#    #mapping_file = exp1_dir + 'map_mob1.yaml'
#    #layout_file = exp1_dir + 'layout_mob1.yaml'
#    mapping_file = exp1_dir + 'result/mapping_' + arch + '_' + model + '_' + str(layer) + '_' + 'false' + '_' + str(1) + '.yaml'
#    layout_file = exp1_dir + 'result/layout_' + arch + '_' + model + '_' + str(layer) + '_' + 'false' + '_' + str(1) + '.yaml'
#    
#    model_command = timeloop_ld_path + ' ' + timeloop_model + ' ' + arch_file + ' ' + workload_file + ' ' + crypto_file + ' ' + mapping_file + ' ' + layout_file
#
#    start = time.time()
#    result = subprocess.run(model_command, capture_output=True, text=True, shell=True)
#    end = time.time()
#
#    energy, latency = extract_energy_latency(model=True)
#    wall_time = end - start
#
#    csv_str = 'model, ' + arch + ', ' + model + ', ' + str(layer) + ', ' + shared + ', ' + str(number_engines) + ', ' + str(energy) + ', ' + str(latency) + ', ' + f"{wall_time:.3f}" + '\n'
#    with open(csv_file, 'a') as f:
#        f.write(csv_str)


with open(csv_file, 'w') as f:
    csv_header = "Type, Architecture, Model, Layer, Block size, Energy, Latency, Wall time\n"
    f.write(csv_header)


arch = 'eyeriss'
model = 'mobv3'
layer = 2
block_size_options = [32, 64, 128, 256]
for block_size in block_size_options:
    print("MAP", arch, model, "layer", layer, "block_size", block_size)
    run_mapper(arch, model, layer, block_size)
for block_size in block_size_options:
    print("NOCRYPTO", arch, model, "layer", layer, "block_size", block_size)
    run_mapper_no_crypto(arch, model, layer, block_size)