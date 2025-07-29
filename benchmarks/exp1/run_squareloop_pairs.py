import time
import subprocess

timeloop_ld_path = 'LD_LIBRARY_PATH=/home/ubuntu/repo/squareloop/build:$LD_LIBRARY_PATH'
timeloop_path = '/home/ubuntu/repo/squareloop/build/'
timeloop_mapper = timeloop_path + 'timeloop-mapper'

exp1_dir = '/home/ubuntu/repo/squareloop/benchmarks/exp1/'

csv_file = exp1_dir + 'result/squareloop/stats.csv'

mapper_file = exp1_dir + 'mapper.yaml'

crypto_file = exp1_dir + 'crypto.yaml'

arch_path = {
    'eyeriss' : exp1_dir + 'arch/eyeriss_like/arch/eyeriss_like.yaml ' + exp1_dir + 'arch/eyeriss_like/arch/components/* ' + exp1_dir + 'arch/eyeriss_like/constraints/*',
    'systolic' : exp1_dir + 'arch/systolic_array/*',
    'vector256' : exp1_dir + 'arch/vector_256.yaml',
}

model_path = {
    'resnet18' : exp1_dir + 'workload/resnet18/resnet18_batch1_layer',
    'mobv3' : exp1_dir + 'workload/mobv3/mobilenet_v3_large_',
}

dependency_path = {
    'resnet18' : exp1_dir + 'workload/resnet18/resnet18_dependent.yaml',
    'mobv3' : exp1_dir + 'workload/mobv3/mobv3_dependent.yaml',
}

model_num_layers = {
    'resnet18' : 21,
    'mobv3' : 62,
}


def parse_dependencies(filename):
    result = []
    current_index = -1

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            if ':' in line:
                # New index line like "4:" or "5:"
                index_part, value_part = line.split(':', 1)
                current_index = int(index_part.strip()) - 1

                # Ensure result has enough space
                while len(result) <= current_index:
                    result.append([])

                if value_part.strip() == '[]':
                    result[current_index] = []
            elif line.startswith('-'):
                value = int(line[1:].strip())
                result[current_index].append(value)

    return result


def extract_energy_latency(output):
    last_line = output.stdout.strip().split('\n')[-3]
    parts = last_line.split('|')

    pj_compute_str = parts[1].strip().split('=')[1].strip()
    cycles_str = parts[2].strip().split('=')[1].strip()

    pj_per_compute = float(pj_compute_str)
    cycles = int(cycles_str)
    return pj_per_compute, cycles


def run_mapper(arch, model, layer):
    arch_file = arch_path[arch]
    workload_file = model_path[model] + str(layer) + '.yaml'
    
    mapper_command = timeloop_ld_path + ' ' + timeloop_mapper + ' ' + mapper_file + ' ' + arch_file + ' ' + workload_file + ' ' + crypto_file

    start = time.time()
    result = subprocess.run(mapper_command, capture_output=True, text=True, shell=True)
    end = time.time()

    mapping_file = exp1_dir + 'result/squareloop/mapping_' + arch + '_' + model + '_' + str(layer) + '.yaml'
    subprocess.run('cp timeloop-mapper.map.yaml ' + mapping_file, capture_output=False, shell=True)

    layout_file = exp1_dir + 'result/squareloop/layout_' + arch + '_' + model + '_' + str(layer) + '.yaml'
    subprocess.run('cp timeloop-mapper.layout.yaml ' + layout_file, capture_output=False, shell=True)

    energy, latency = extract_energy_latency(result)
    wall_time = end - start

    csv_str = arch + ', ' + model + ', ' + str(layer) + ', ' + str(energy) + ', ' + str(latency) + ', ' + f"{wall_time:.3f}" + '\n'
    with open(csv_file, 'a') as f:
        f.write(csv_str)


with open(csv_file, 'w') as f:
    pass  

for arch in arch_path:
    for model in model_path:
        dependencies = parse_dependencies(dependency_path[model])
        num_layers = model_num_layers[model]
        for layer in range(1, num_layers+1):
            print(arch, model, layer)
            run_mapper(arch, model, layer)