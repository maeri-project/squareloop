import time
import subprocess

timeloop_ld_path = 'LD_LIBRARY_PATH=/home/ubuntu/repo/timeloop/build:$LD_LIBRARY_PATH'
timeloop_path = '/home/ubuntu/repo/timeloop/build/'
timeloop_mapper = timeloop_path + 'timeloop-mapper'

exp1_dir = '/home/ubuntu/repo/squareloop/benchmarks/exp1/'

csv_file = exp1_dir + 'result/timeloop/stats.csv'

mapper_file = exp1_dir + 'mapper.yaml'

arch_path = {
    'eyeriss' : exp1_dir + 'arch/eyeriss_like/arch/eyeriss_like.yaml ' + exp1_dir + 'arch/eyeriss_like/arch/components/* ' + exp1_dir + 'arch/eyeriss_like/constraints/*',
    'vector256' : exp1_dir + 'arch/vector_256.yaml',
    'systolic' : exp1_dir + 'arch/systolic_array/*',
}

model_path = {
    'resnet18' : exp1_dir + 'workload/resnet18/resnet18_batch1_layer',
    'mobv3' : exp1_dir + 'workload/mobv3/mobilenet_v3_large_',
    'bert_conv' : exp1_dir + 'workload/bert_conv/bert_conv_layer',
}

model_num_layers = {
    'resnet18' : 21,
    'mobv3' : 62,
    'bert_conv' : 3,
}


def extract_energy_latency(output):
    last_line = output.stdout.strip().split('\n')[-1]
    parts = last_line.split('|')

    pj_compute_str = parts[1].strip().split('=')[1].strip()
    cycles_str = parts[2].strip().split('=')[1].strip()

    pj_per_compute = float(pj_compute_str)
    cycles = int(cycles_str)
    return pj_per_compute, cycles


def run_mapper(arch, model, layer):
    arch_file = arch_path[arch]
    workload_file = model_path[model] + str(layer) + '.yaml'
    
    mapper_command = timeloop_ld_path + ' ' + timeloop_mapper + ' ' + mapper_file + ' ' + arch_file + ' ' + workload_file 

    start = time.time()
    result = subprocess.run(mapper_command, capture_output=True, text=True, shell=True)
    end = time.time()

    mapping_file = exp1_dir + 'result/timeloop/mapping_' + arch + '_' + model + '_' + str(layer) + '.yaml'
    subprocess.run('cp timeloop-mapper.map.yaml ' + mapping_file, capture_output=False, shell=True)

    energy, latency = extract_energy_latency(result)
    wall_time = end - start

    csv_str = arch + ', ' + model + ', ' + str(layer) + ', ' + str(energy) + ', ' + str(latency) + ', ' + f"{wall_time:.3f}" + '\n'
    with open(csv_file, 'a') as f:
        f.write(csv_str)


#with open(csv_file, 'w') as f:
#    pass  

for arch in arch_path:
    #for model in model_path:
    for model in ['bert_conv']:
        num_layers = model_num_layers[model]
        for layer in range(1, num_layers+1):
            print(arch, model, layer)
            run_mapper(arch, model, layer)