import os
import yaml
import sys
import random
import shutil

from layout_utils import check_layout_dependency

# Cross-layer search provided workloads (entire neural networks) and architecture
# Assume that layouts for the workload + architecture are already generated
def get_legal_layouts(layout_path, workload_path, workload_name_prefix, arch_name_prefix, layer_idx, this_dependent_layout_path, dependent_layer_idx):
    # Find all layouts in layout_path (e.g., benchmarks/layout/{workload_name}/{arch_name_prefix}_{.....}_{layer_idx}.yaml)
    all_files = [f for f in os.listdir(layout_path)]
    this_layer_files = []
    for f in all_files:
        if (arch_name_prefix in f) and ('{}.yaml'.format(layer_idx) in f):
            this_layer_files.append(f)

    if this_dependent_layout_path is None:
        return this_layer_files
    
    legal_layout_files = []
    for f in this_layer_files:
        if check_layout_dependency(this_dependent_layout_path, os.path.join(layout_path, f), os.path.join(workload_path, '{}_layer{}.yaml'.format(workload_name_prefix, dependent_layer_idx)), os.path.join(workload_path, '{}_layer{}.yaml'.format(workload_name_prefix, layer_idx))):
            legal_layout_files.append(f)
    
    return legal_layout_files

def timeloop_mapper_with_layout(workload_path, workload_name_prefix, layout_path, layout, arch_path, mapper_path, idx, workdir):

    # Original work directory
    original_dir = os.getcwd()

    shutil.copy(os.path.join(layout_path, layout), os.path.join(workdir, 'layout.yaml'))

    arch_path_abs = os.path.abspath(arch_path)
    workload_path_abs = os.path.abspath(workload_path)
    mapper_path_abs = os.path.abspath(mapper_path)

    # timeloop-mapper arch.yaml layer.yaml mapper.yaml layout.yaml
    print("Work directory: {}".format(workdir))
    os.chdir(workdir)

    if os.path.isfile(arch_path_abs):
        arch_file = arch_path_abs
    else:
        # directory case
        arch_files = []
        for root, dirs, files in os.walk(arch_path_abs):
            for name in files:
                if '.yaml' in name:
                    arch_file.append(os.path.join(root, name))
        arch_file = ' '.join(arch_files)

    layer_file = os.path.join(workload_path_abs, '{}_layer{}.yaml'.format(workload_name_prefix, idx))

    mapper_file = mapper_path_abs
    
    layout_file = 'layout.yaml'

    cmd = 'timeloop-mapper {} {} {} {}'.format(arch_file, layer_file, mapper_file, layout_file)
    print("Executing command: {}".format(cmd))

    os.system(cmd)

    # parse the result
    cycles = 0
    energy = 0
    edp = 0
    with open('timeloop-mapper.stats.txt', 'r') as f:
        for line in f:
            if 'Cycles:' in line:
                cycles = line.split('Cycles:')[1].strip()
            elif 'Energy:' in line:
                energy = line.split('Energy:')[1].strip().split('uJ')[0].strip()
            elif 'EDP(J*cycle):' in line:
                edp = float(line.split('EDP(J*cycle):')[1].strip())
    
    os.chdir(original_dir)

    return {'cycles': cycles, 'energy': energy, 'edp': edp}

def run_crosslayer_search(workload_path, workload_name_prefix, layout_path, arch_path, arch_name_prefix, mapper_path, \
                          crosslayer_policy, cost_policy, save_path):
    # Check if save_path exists - if not create
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    assert (crosslayer_policy in ['sequential', 'random'])

    # Find the dependent layer information
    with open(os.path.join(workload_path, '{}_dependent.yaml'.format(workload_name_prefix)), 'r') as f:
        layer_dependency = yaml.safe_load(f)
    
    layer_idx = layer_dependency.keys()
    
    if crosslayer_policy == 'sequential':
        layer_visit_order = layer_idx
    elif crosslayer_policy == 'random':
        layer_visit_order = layer_idx[:]
        random.shuffle(layer_visit_order)
    else:
        raise NotImplementedError()
    
    # Iterate layer_visit_order
    visited = []
    dependency_broken = []
    for idx in layer_visit_order:
        # Check subfolder
        subfolder = os.path.join(save_path, '{}_layer{}'.format(workload_name_prefix, idx))
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
        
        # Check previous dependent layer and if it was already visited
        # There should be only ONE previous dependent layer 
        # (TODO; for multiple previous dependent layer (e.g., residual nets), we assume the addition will incur 'rehash' anyway)
        if len(layer_dependency[idx]) > 0:
            this_dependent_layer = layer_dependency[idx][0]
            if this_dependent_layer in visited:
                this_dependent_layout_path = os.path.join(save_path, '{}_layer{}/layout.yaml'.format(workload_name_prefix, this_dependent_layer))
            else:
                this_dependent_layout_path = None
        else:
            this_dependent_layer = -1
            this_dependent_layout_path = None
        
        legal_layouts = get_legal_layouts(layout_path, workload_path, workload_name_prefix, arch_name_prefix, idx, this_dependent_layout_path, this_dependent_layer)
        # print(legal_layouts)
        # If legal_layouts is null, then it means that no pre-defined layout can satisfy the condition
        # --> Break the dependency here
        if (len(legal_layouts) == 0) and (this_dependent_layout_path is not None):
            print("No legal cross-layer layout is found between layer {} and layer {}.".format(this_dependent_layer, idx))
            print("Break the dependency here (rehashed).")

            this_dependent_layer_path = None
            legal_layouts = get_legal_layouts(layout_path, workload_path, workload_name_prefix, arch_name_prefix, idx, this_dependent_layout_path, this_dependent_layer)
            dependency_broken.append((this_dependent_layer, idx))

        # for each legal layout, run search and get its performance and energy values
        cost_summary = {}
        for layout in legal_layouts:
            if not os.path.exists(os.path.join(subfolder, layout)):
                os.mkdir(os.path.join(subfolder, layout))
            cost_summary[layout] = timeloop_mapper_with_layout(workload_path, workload_name_prefix, layout_path, layout, arch_path, mapper_path, idx, os.path.join(subfolder, layout))

        # find the best layout
        cost_for_choosing_layout = {key: value[cost_policy] for (key, value) in cost_summary.items()}
        best_layout = max(cost_for_choosing_layout, key=cost_for_choosing_layout.get)

        # copy mapping/layout/stats found to here
        shutil.copy(os.path.join(subfolder, best_layout, 'timeloop-mapper.map.yaml'), os.path.join(subfolder, 'timeloop-mapper.map.yaml'))
        shutil.copy(os.path.join(subfolder, best_layout, 'layout.yaml'), os.path.join(subfolder, 'layout.yaml'))
        shutil.copy(os.path.join(subfolder, best_layout, 'timeloop-mapper.stats.txt'), os.path.join(subfolder, 'timeloop-mapper.stats.txt'))

        # append this layer to visited
        visited.append(idx)

    # add any rehashing costs

    # report end-to-end performance/energy


if __name__ == '__main__':
    #  layout_path, workload_path, workload_name_prefix, arch_name_prefix, layer_idx, this_dependent_layout_path, dependent_layer_idx
    # print(get_legal_layouts('../../layout/alexnet/', 'alexnet', 'AlexNet', 'vector_256', 4, '../../layout/alexnet/vector_256_SRCQPMNHW_Wx8Hx4_3.yaml', 3))

    run_crosslayer_search('test/alexnet', 'AlexNet', 'test/alexnet_layout', '../../arch_designs/vector_256.yaml', 'vector_256', \
                          'test/mapper.yaml', 'sequential', 'cycles', 'test/alexnet_search')