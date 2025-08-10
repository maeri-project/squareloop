import time
import subprocess
import yaml
import copy
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
import re
import math

##########################
# Configurations
##########################


squareloop_dir = '/home/ubuntu/squareloop/'

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


##########################
# Helper Functions
##########################


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

##########################
# Rehash Latency Header
##########################

@dataclass
class Dataspace:
    layer_id: int
    dataspace_type: str  # "Inputs", "Weights", or "Outputs"

    def __str__(self) -> str:
        return f"Layer{self.layer_id}_{self.dataspace_type}"


@dataclass
class LayerDataspaces:
    layer_id: int
    inputs: Dataspace
    weights: Dataspace
    outputs: Dataspace


@dataclass
class DataspaceLayoutNest:
    dataspace_name: str
    target: str
    layout_type: str  # "interline", "intraline", or "authblock_lines"
    factors: Dict[str, int]
    permutation: str
    ranks: List[str]


@dataclass
class LayerDataspaceLayouts:
    layer_id: int
    inputs_layouts: List[DataspaceLayoutNest]
    weights_layouts: List[DataspaceLayoutNest]
    outputs_layouts: List[DataspaceLayoutNest]

    def get_layouts_by_dataspace(self, dataspace_name: str) -> List[DataspaceLayoutNest]:
        name = dataspace_name.lower()
        if name == "inputs":
            return self.inputs_layouts
        if name == "weights":
            return self.weights_layouts
        if name == "outputs":
            return self.outputs_layouts
        return []


@dataclass
class LayoutNest:
    target: str
    layout_type: str
    factors: Dict[str, int]
    permutation: str
    ranks: List[str]


# Helpers


def create_layer_dataspaces(layer_ids: List[int]) -> Dict[int, LayerDataspaces]:
    layer_dataspaces: Dict[int, LayerDataspaces] = {}
    for layer_id in layer_ids:
        layer_dataspaces[layer_id] = LayerDataspaces(
            layer_id=layer_id,
            inputs=Dataspace(layer_id, "Inputs"),
            weights=Dataspace(layer_id, "Weights"),
            outputs=Dataspace(layer_id, "Outputs"),
        )
    return layer_dataspaces


def parse_factors_string(factors_str: str) -> Dict[str, int]:
    factors: Dict[str, int] = {}
    if not factors_str:
        return factors
    for pair in factors_str.split():
        if '=' in pair:
            rank, value = pair.split('=', 1)
            try:
                factors[rank.strip()] = int(value.strip())
            except ValueError:
                pass
    return factors


def read_layout_file(layout_file_path: str) -> List[LayoutNest]:
    try:
        with open(layout_file_path, 'r') as f:
            layout_data = yaml.safe_load(f)

        layout_nests: List[LayoutNest] = []
        if layout_data and 'layout' in layout_data:
            for entry in layout_data['layout']:
                target = entry.get('target', '')
                layout_type = entry.get('type', '')
                factors_str = entry.get('factors', '')
                permutation = entry.get('permutation', '')

                factors = parse_factors_string(factors_str)
                ranks = list(permutation) if permutation else []

                layout_nests.append(
                    LayoutNest(
                        target=target,
                        layout_type=layout_type,
                        factors=factors,
                        permutation=permutation,
                        ranks=ranks,
                    )
                )

        return layout_nests

    except FileNotFoundError:
        print(f"Warning: Layout file not found: {layout_file_path}")
        return []


def read_layer_problem_definition(layer_file_path: str) -> Dict[str, List[str]]:
    try:
        with open(layer_file_path, 'r') as f:
            layer_data = yaml.safe_load(f)

        dataspace_to_ranks: Dict[str, List[str]] = {}
        if layer_data and 'problem' in layer_data and 'shape' in layer_data['problem']:
            for d in layer_data['problem']['shape'].get('data-spaces', []):
                name = d.get('name', '')
                ranks = d.get('ranks', [])
                if name and ranks:
                    dataspace_to_ranks[name] = ranks
        return dataspace_to_ranks

    except FileNotFoundError:
        print(f"Warning: Layer problem file not found: {layer_file_path}")
        return {}


def split_factors_by_dataspace(
    factors: Dict[str, int],
    dataspace_to_ranks: Dict[str, List[str]],
) -> Dict[str, Dict[str, int]]:
    dataspace_factors: Dict[str, Dict[str, int]] = {}
    for dataspace_name, ranks in dataspace_to_ranks.items():
        dataspace_factors[dataspace_name] = {}
        for rank in ranks:
            if rank in factors:
                dataspace_factors[dataspace_name][rank] = factors[rank]
    return dataspace_factors


def read_all_layer_dataspace_layouts(
    base_path: str = "/home/ubuntu/squareloop/benchmarks/squareloop_resent18",
    problem_path: str = "/home/ubuntu/squareloop/benchmarks/script/crosslayer_search/test/resnet18",
) -> Dict[int, LayerDataspaceLayouts]:
    layer_dataspace_layouts: Dict[int, LayerDataspaceLayouts] = {}

    try:
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if not (os.path.isdir(item_path) and item.startswith('resnet18_layer')):
                continue

            layer_match = re.search(r'resnet18_layer(\d+)', item)
            if not layer_match:
                continue
            layer_id = int(layer_match.group(1))

            problem_file_path = os.path.join(problem_path, f'resnet18_batch1_layer{layer_id}.yaml')
            dataspace_to_ranks = read_layer_problem_definition(problem_file_path)
            if not dataspace_to_ranks:
                continue

            layout_file_path = os.path.join(item_path, 'timeloop-mapper.layout.yaml')
            layout_nests = read_layout_file(layout_file_path)
            if not layout_nests:
                continue

            inputs_layouts: List[DataspaceLayoutNest] = []
            weights_layouts: List[DataspaceLayoutNest] = []
            outputs_layouts: List[DataspaceLayoutNest] = []

            for nest in layout_nests:
                dataspace_factors = split_factors_by_dataspace(nest.factors, dataspace_to_ranks)
                for dataspace_name, factors in dataspace_factors.items():
                    if not factors:
                        continue
                    dataspace_ranks = dataspace_to_ranks[dataspace_name]
                    dataspace_permutation = ''.join([r for r in nest.permutation if r in dataspace_ranks])

                    dln = DataspaceLayoutNest(
                        dataspace_name=dataspace_name,
                        target=nest.target,
                        layout_type=nest.layout_type,
                        factors=factors,
                        permutation=dataspace_permutation,
                        ranks=dataspace_ranks,
                    )

                    if dataspace_name == "Inputs":
                        inputs_layouts.append(dln)
                    elif dataspace_name == "Weights":
                        weights_layouts.append(dln)
                    elif dataspace_name == "Outputs":
                        outputs_layouts.append(dln)

            layer_dataspace_layouts[layer_id] = LayerDataspaceLayouts(
                layer_id=layer_id,
                inputs_layouts=inputs_layouts,
                weights_layouts=weights_layouts,
                outputs_layouts=outputs_layouts,
            )

        return layer_dataspace_layouts

    except Exception as e:
        print(f"Error reading layer dataspace layouts: {e}")
        return {}


def find_dataspace_layouts(
    dataspace: Dataspace,
    layer_dataspace_layouts: Dict[int, LayerDataspaceLayouts],
) -> List[DataspaceLayoutNest]:
    layer_id = dataspace.layer_id
    if layer_id not in layer_dataspace_layouts:
        return []
    return layer_dataspace_layouts[layer_id].get_layouts_by_dataspace(dataspace.dataspace_type)


def parse_dataspace_dependencies(
    dependency_file_path: str,
) -> Tuple[List[List[Dataspace]], Dict[int, LayerDataspaces]]:
    try:
        with open(dependency_file_path, 'r') as f:
            dependencies = yaml.safe_load(f)

        all_layers = set()
        for layer, deps in (dependencies or {}).items():
            all_layers.add(layer)
            for dep in (deps or []):
                all_layers.add(dep)

        layer_dataspaces = create_layer_dataspaces(sorted(all_layers))

        assigned: set = set()
        groups: List[List[Dataspace]] = []

        for layer in sorted(all_layers):
            if layer in dependencies and dependencies[layer]:
                current_input = layer_dataspaces[layer].inputs
                connection_group: List[Dataspace] = []

                if str(current_input) not in assigned:
                    connection_group.append(current_input)
                    assigned.add(str(current_input))

                for dep_layer in dependencies[layer]:
                    dep_output = layer_dataspaces[dep_layer].outputs
                    if str(dep_output) not in assigned:
                        connection_group.append(dep_output)
                        assigned.add(str(dep_output))

                if connection_group:
                    groups.append(connection_group)

        for layer_id in sorted(all_layers):
            layer_ds = layer_dataspaces[layer_id]
            if str(layer_ds.inputs) not in assigned:
                groups.append([layer_ds.inputs])
                assigned.add(str(layer_ds.inputs))
            if str(layer_ds.outputs) not in assigned:
                groups.append([layer_ds.outputs])
                assigned.add(str(layer_ds.outputs))

        return groups, layer_dataspaces

    except FileNotFoundError:
        print(f"Error: Dependency file not found: {dependency_file_path}")
        return [], {}
    except Exception as e:
        print(f"Error parsing dependency file {dependency_file_path}: {e}")
        return [], {}


##########################
# Rehash Latency Calculation
##########################

def read_crypto_config(
) -> Tuple[int, int, int, int]:
    defaults: Tuple[int, int, int, int] = (1, 1, 1, 1)
    try:
        with open(crypto_file, 'r') as f:
            crypto_data = yaml.safe_load(f) or {}
        c = crypto_data.get('crypto', {}) or {}
        return (
            int(c.get('auth-additional-cycle-per-block', defaults[0])),
            int(c.get('auth-cycle-per-datapath', defaults[1])),
            int(c.get('enc-cycle-per-datapath', defaults[2])),
            int(c.get('datapath', defaults[3])),
        )
    except Exception:
        print(f"Warning: Could not read crypto config at {crypto_file}; using defaults {defaults}")
        return defaults


def calculate_rehash_latency(
    dataspace_deps: List[List[Dataspace]],
    layer_dataspace_layouts: Dict[int, LayerDataspaceLayouts],
    crypto_config_path: crypto_file,
) -> Dict[int, int]:
    word_bits = 16
    (
        auth_additional_cycle_per_block,
        auth_cycle_per_datapath,
        enc_cycle_per_datapath,
        datapath,
    ) = read_crypto_config()

    datapath = max(1, int(datapath))
    per_block_additional = int(auth_additional_cycle_per_block)
    per_datapath_cycle = max(int(enc_cycle_per_datapath), int(auth_cycle_per_datapath))

    group_rehash_latencies: Dict[int, int] = {}

    for group_idx, dep_group in enumerate(dataspace_deps, 1):
        total_rehash_latency = 0.0

        for dataspace in dep_group:
            if dataspace.dataspace_type != "Outputs":
                continue

            dataspace_layouts = find_dataspace_layouts(dataspace, layer_dataspace_layouts)
            layout_by_type: Dict[str, DataspaceLayoutNest] = {}
            for layout in dataspace_layouts:
                if layout.target == "DRAM":
                    layout_by_type[layout.layout_type] = layout

            required_types = ["intraline", "interline", "authblock_lines"]
            if not all(t in layout_by_type for t in required_types):
                continue

            intraline_factors = layout_by_type["intraline"].factors
            interline_factors = layout_by_type["interline"].factors
            authblock_lines_factors = layout_by_type["authblock_lines"].factors

            all_ranks = set(intraline_factors.keys()) | set(interline_factors.keys()) | set(authblock_lines_factors.keys())

            num_authblock_lines = 1.0
            authblock_lines_size = 1.0

            for rank in all_ranks:
                intraline_factor = int(intraline_factors.get(rank, 1))
                interline_factor = int(interline_factors.get(rank, 1))
                authblock_lines_factor = int(authblock_lines_factors.get(rank, 1))

                denom = max(1, authblock_lines_factor)
                num_authblock_lines *= math.ceil(interline_factor / denom)
                authblock_lines_size *= max(1, authblock_lines_factor) * max(1, intraline_factor)

            latency_per_authblock = (
                authblock_lines_size * word_bits / datapath * per_datapath_cycle + per_block_additional
            )
            total_rehash_latency += num_authblock_lines * latency_per_authblock

        group_rehash_latencies[group_idx] = int(total_rehash_latency)

    return group_rehash_latencies


##########################
# Squareloop Functions
##########################


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


def run_squareloop(arch, model, layer, csv_type, result_dir, csv_file, layout_file='', mapping_file='', shared=None, number_engines=None, block_size=None, no_crypto=False):
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
    ld_path_tmp = squareloop_ld_path
    exe_tmp = squareloop_exe
    
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
    else:
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

    energy, latency = extract_energy_latency(model=(True if mapping_file else False))
    wall_time = end - start

    csv_str = csv_type + ', ' + arch + ', ' + model + ', ' + str(layer) + ', ' 
    if shared != None:
        csv_str += str(shared) + ', ' + str(number_engines) + ', '
    if block_size != None:
        csv_str += str(block_size) + ', '
    csv_str += str(energy) + ', ' + str(latency) + ', ' + f"{wall_time:.3f}" + '\n'
    with open(csv_file, 'a') as f:
        f.write(csv_str)

