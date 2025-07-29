import time
import subprocess

timeloop_ld_path = 'LD_LIBRARY_PATH=/home/ubuntu/repo/squareloop/build:$LD_LIBRARY_PATH'
timeloop_path = '/home/ubuntu/repo/squareloop/build/'
timeloop_mapper = timeloop_path + 'timeloop-mapper'
timeloop_model = timeloop_path + 'timeloop-model'

exp1_dir = '/home/ubuntu/repo/squareloop/benchmarks/exp2/'

csv_file = exp1_dir + 'result/stats.csv'

mapper_file = exp1_dir + 'mapper.yaml'

crypto_template_file = exp1_dir + 'crypto_template.yaml'
crypto_file = exp1_dir + 'crypto.yaml'

arch_path = {
    'eyeriss' : exp1_dir + 'arch/eyeriss_like/arch/eyeriss_like2.yaml ' + exp1_dir + 'arch/eyeriss_like/arch/components/* ' + exp1_dir + 'arch/eyeriss_like/constraints/*',
    'systolic' : exp1_dir + 'arch/systolic_array/*',
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


def extract_energy_latency(output, line=-3):
    last_line = output.stdout.strip().split('\n')[line]
    parts = last_line.split('|')

    pj_compute_str = parts[1].strip().split('=')[1].strip()
    cycles_str = parts[2].strip().split('=')[1].strip()

    pj_per_compute = float(pj_compute_str)
    cycles = int(cycles_str)
    return pj_per_compute, cycles


def generate_crypto(shared, number_engines):
    subprocess.run('cp ' + crypto_template_file + ' ' + crypto_file, capture_output=False, shell=True)
    with open(crypto_file, 'a') as f:
        f.write("\n  shared: " + shared)
        f.write("\n  number_engines: " + str(number_engines))


def run_mapper(arch, model, layer, shared, number_engines):
    generate_crypto(shared, number_engines)

    arch_file = arch_path[arch]
    workload_file = model_path[model] + str(layer) + '.yaml'
    
    mapper_command = timeloop_ld_path + ' ' + timeloop_mapper + ' ' + mapper_file + ' ' + arch_file + ' ' + workload_file + ' ' + crypto_file

    start = time.time()
    result = subprocess.run(mapper_command, capture_output=True, text=True, shell=True)
    end = time.time()

    #mapping_file = exp1_dir + 'result/mapping_' + arch + '_' + model + '_' + str(layer) + '_' + shared + '_' + str(number_engines) + '.yaml'
    mapping_file = exp1_dir + 'result/mapping_' + arch + '_' + model + '_' + str(layer) + '.yaml'
    subprocess.run('cp timeloop-mapper.map.yaml ' + mapping_file, capture_output=False, shell=True)

    #layout_file = exp1_dir + 'result/layout_' + arch + '_' + model + '_' + str(layer) + '_' + shared + '_' + str(number_engines) + '.yaml'
    layout_file = exp1_dir + 'result/layout_' + arch + '_' + model + '_' + str(layer) + '.yaml'
    subprocess.run('cp timeloop-mapper.layout.yaml ' + layout_file, capture_output=False, shell=True)

    energy, latency = extract_energy_latency(result)
    wall_time = end - start

    csv_str = arch + ', ' + model + ', ' + str(layer) + ', ' + shared + ', ' + str(number_engines) + ', ' + str(energy) + ', ' + str(latency) + ', ' + f"{wall_time:.3f}" + '\n'
    with open(csv_file, 'a') as f:
        f.write(csv_str)


def run_model(arch, model, layer, shared, number_engines):
    generate_crypto(shared, number_engines)

    arch_file = arch_path[arch]
    workload_file = model_path[model] + str(layer) + '.yaml'
    
    model_command = timeloop_ld_path + ' ' + timeloop_model + ' ' + arch_file + ' ' + workload_file + ' ' + crypto_file

    start = time.time()
    result = subprocess.run(model_command, capture_output=True, text=True, shell=True)
    end = time.time()

    energy, latency = extract_energy_latency(result, line=-1)
    wall_time = end - start

    csv_str = arch + ', ' + model + ', ' + str(layer) + ', ' + shared + ', ' + str(number_engines) + ', ' + str(energy) + ', ' + str(latency) + ', ' + f"{wall_time:.3f}" + '\n'
    with open(csv_file, 'a') as f:
        f.write(csv_str)


with open(csv_file, 'w') as f:
    csv_header = "Architecture, Model, Layer, Shared crypto engine, Number of crypto engines, Energy, Latency, Wall time\n"
    f.write(csv_header)

#for arch in arch_path:
#    for model in model_path:
#        num_layers = model_num_layers[model]
#        for layer in range(1, num_layers+1):
#            print(arch, model, layer)
#            run_mapper(arch, model, layer)
arch = 'eyeriss'
model = 'mobv3'
num_layers = model_num_layers[model]
for layer in range(1, num_layers+1):
    #for shared in ['false', 'true']:
    for shared in ['false']:
        #for number_engines in [1, 2, 4, 8]:
        for number_engines in [1]:
            print(arch, model, "layer", layer, "shared", shared, "number_engines", number_engines)
            run_mapper(arch, model, layer, shared, number_engines)

            print(arch, model, "layer", layer, "shared", shared, "number_engines", 2)
            run_model(arch, model, layer, shared, number_engines)