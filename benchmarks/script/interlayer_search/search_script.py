import os
import re
import yaml
import shutil
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class LayoutNest:
    """Represents a layout nest corresponding to the C++ LayoutNest structure."""
    target: str                     # e.g., "DRAM", "shared_glb"
    layout_type: str               # "interline", "intraline", or "authblock_lines"
    factors: Dict[str, int]        # Factor for each rank (e.g., {"C": 3, "K": 16, ...})
    permutation: str               # Rank order string (e.g., "CKRSNVHWLPQ")
    ranks: List[str]               # Individual ranks derived from permutation

    def __post_init__(self):
        if self.ranks is None:
            # Extract individual ranks from permutation string
            self.ranks = list(self.permutation) if self.permutation else []

    def __str__(self):
        return f"{self.target}_{self.layout_type}"

    def __repr__(self):
        return self.__str__()


@dataclass
class LayerLayout:
    """Represents all layout information for a single layer."""
    layer_id: int
    layout_nests: List[LayoutNest]
    targets: List[str]                    # Unique targets in this layer

    def __post_init__(self):
        if self.targets is None:
            # Extract unique targets from layout nests
            self.targets = list(set(nest.target for nest in self.layout_nests))

    def get_nests_by_target(self, target: str) -> List[LayoutNest]:
        """Get all layout nests for a specific target."""
        return [nest for nest in self.layout_nests if nest.target == target]

    def get_nests_by_type(self, layout_type: str) -> List[LayoutNest]:
        """Get all layout nests for a specific type (interline/intraline/authblock_lines)."""
        return [nest for nest in self.layout_nests if nest.layout_type == layout_type]


@dataclass
class Dataspace:
    """Represents a single dataspace (Inputs, Weights, or Outputs) for a layer."""
    layer_id: int
    dataspace_type: str  # "Inputs", "Weights", or "Outputs"
    dimensions: Dict = None

    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = {}

    def __str__(self):
        return f"Layer{self.layer_id}_{self.dataspace_type}"

    def __repr__(self):
        return self.__str__()


@dataclass
class LayerDataspaces:
    """Represents all dataspaces for a single layer."""
    layer_id: int
    inputs: Dataspace
    weights: Dataspace
    outputs: Dataspace

    def __post_init__(self):
        # Initialize dataspaces if not provided
        if self.inputs is None:
            self.inputs = Dataspace(self.layer_id, "Inputs")
        if self.weights is None:
            self.weights = Dataspace(self.layer_id, "Weights")
        if self.outputs is None:
            self.outputs = Dataspace(self.layer_id, "Outputs")

    def get_all_dataspaces(self) -> List[Dataspace]:
        """Return all dataspaces for this layer."""
        return [self.inputs, self.weights, self.outputs]


@dataclass
class DataspaceLayoutNest:
    """Represents layout nest information for a specific dataspace."""
    dataspace_name: str             # "Inputs", "Weights", or "Outputs"
    target: str                     # e.g., "DRAM", "shared_glb"
    layout_type: str               # "interline", "intraline", or "authblock_lines"
    factors: Dict[str, int]        # Factor for each rank in this dataspace
    permutation: str               # Rank order string (e.g., "CKRS" for Weights)
    ranks: List[str]               # Individual ranks for this dataspace

    def __post_init__(self):
        if self.ranks is None:
            # Extract individual ranks from permutation string
            self.ranks = list(self.permutation) if self.permutation else []

    def __str__(self):
        return f"{self.dataspace_name}_{self.target}_{self.layout_type}"

    def __repr__(self):
        return self.__str__()


@dataclass
class LayerDataspaceLayouts:
    """Represents layout information for all dataspaces in a single layer."""
    layer_id: int
    inputs_layouts: List[DataspaceLayoutNest]
    weights_layouts: List[DataspaceLayoutNest]
    outputs_layouts: List[DataspaceLayoutNest]
    targets: List[str]                    # Unique targets in this layer

    def __post_init__(self):
        if self.targets is None:
            # Extract unique targets from all layout nests
            all_targets = set()
            for layouts in [self.inputs_layouts, self.weights_layouts, self.outputs_layouts]:
                for layout in layouts:
                    all_targets.add(layout.target)
            self.targets = list(all_targets)

    def get_layouts_by_dataspace(self, dataspace_name: str) -> List[DataspaceLayoutNest]:
        """Get all layout nests for a specific dataspace."""
        if dataspace_name.lower() == "inputs":
            return self.inputs_layouts
        elif dataspace_name.lower() == "weights":
            return self.weights_layouts
        elif dataspace_name.lower() == "outputs":
            return self.outputs_layouts
        else:
            return []

    def get_layouts_by_target(self, target: str) -> List[DataspaceLayoutNest]:
        """Get all layout nests for a specific target across all dataspaces."""
        result = []
        for layouts in [self.inputs_layouts, self.weights_layouts, self.outputs_layouts]:
            result.extend([layout for layout in layouts if layout.target == target])
        return result


@dataclass
class TargetBlockSize:
    """Represents block-size information for a memory target."""
    name: str                       # Target name (e.g., "DRAM", "shared_glb")
    block_size: int                 # Block size in bytes
    class_type: str                 # Class type (e.g., "DRAM", "smartbuffer_SRAM")
    mesh_count: int = 1             # Number of instances (for arrays like PE[0..167])

    def __str__(self):
        if self.mesh_count > 1:
            return f"{self.name}[0..{self.mesh_count-1}]"
        return self.name

    def __repr__(self):
        return self.__str__()


def parse_architecture_subtree(subtree_data: Dict, parent_name: str = "") -> List[TargetBlockSize]:
    """
    Recursively parse architecture subtree to extract target block-size information.

    Args:
        subtree_data: Dictionary containing subtree data from YAML
        parent_name: Name of parent component for context

    Returns:
        List of TargetBlockSize objects
    """
    targets = []

    # Process local components at this level
    if 'local' in subtree_data:
        for component in subtree_data['local']:
            name = component.get('name', '')
            class_type = component.get('class', '')
            attributes = component.get('attributes', {})

            if 'block-size' in attributes:
                block_size = attributes['block-size']

                # Handle mesh arrays (e.g., DummyBuffer[0..13])
                mesh_count = 1
                if 'meshX' in attributes:
                    mesh_count = attributes['meshX']
                elif '[' in name and '..' in name and ']' in name:
                    # Parse range notation like PE[0..167]
                    try:
                        range_part = name[name.find('[')+1:name.find(']')]
                        if '..' in range_part:
                            start, end = range_part.split('..')
                            mesh_count = int(end) - int(start) + 1
                            name = name[:name.find('[')]  # Remove range notation
                    except:
                        pass

                target = TargetBlockSize(
                    name=name,
                    block_size=block_size,
                    class_type=class_type,
                    mesh_count=mesh_count
                )
                targets.append(target)

    # Recursively process subtrees
    if 'subtree' in subtree_data:
        for subtree in subtree_data['subtree']:
            subtree_name = subtree.get('name', '')
            child_targets = parse_architecture_subtree(subtree, subtree_name)
            targets.extend(child_targets)

    return targets


def read_architecture_block_sizes(arch_file_path: str = "/home/ubuntu/squareloop/benchmarks/secureloop-cross-eval/arch/baseline.yaml") -> List[TargetBlockSize]:
    """
    Read architecture file and extract block-size information for all targets.

    Args:
        arch_file_path: Path to the architecture YAML file

    Returns:
        List of TargetBlockSize objects
    """
    try:
        with open(arch_file_path, 'r') as f:
            arch_data = yaml.safe_load(f)

        targets = []
        if 'architecture' in arch_data:
            architecture = arch_data['architecture']
            targets = parse_architecture_subtree(architecture)

        return targets

    except FileNotFoundError:
        print(f"Warning: Architecture file not found: {arch_file_path}")
        return []
    except Exception as e:
        print(f"Error reading architecture file {arch_file_path}: {e}")
        return []


def extract_cycles_from_stats(stats_file_path):
    """
    Extract cycle count from timeloop-mapper.stats.txt file.

    Args:
        stats_file_path (str): Path to the stats file

    Returns:
        int: Cycle count, or None if not found
    """
    try:
        with open(stats_file_path, 'r') as f:
            for line in f:
                if line.strip().startswith('Cycles:'):
                    # Extract the number after "Cycles:"
                    cycle_match = re.search(r'Cycles:\s*(\d+)', line)
                    if cycle_match:
                        return int(cycle_match.group(1))
        return None
    except FileNotFoundError:
        print(f"Warning: Stats file not found: {stats_file_path}")
        return None
    except Exception as e:
        print(f"Error reading {stats_file_path}: {e}")
        return None


def create_layer_dataspaces(layer_ids: List[int]) -> Dict[int, LayerDataspaces]:
    """
    Create LayerDataspaces objects for all specified layers.

    Args:
        layer_ids: List of layer IDs to create dataspaces for

    Returns:
        Dict mapping layer_id to LayerDataspaces object
    """
    layer_dataspaces = {}
    for layer_id in layer_ids:
        layer_dataspaces[layer_id] = LayerDataspaces(
            layer_id=layer_id,
            inputs=Dataspace(layer_id, "Inputs"),
            weights=Dataspace(layer_id, "Weights"),
            outputs=Dataspace(layer_id, "Outputs")
        )
    return layer_dataspaces


def parse_factors_string(factors_str: str) -> Dict[str, int]:
    """
    Parse factors string like "C=3 K=16 R=7 S=1" into a dictionary.

    Args:
        factors_str: String with rank=value pairs

    Returns:
        Dict mapping rank names to their factors
    """
    factors = {}
    if factors_str:
        # Split by spaces and parse each rank=value pair
        for pair in factors_str.split():
            if '=' in pair:
                rank, value = pair.split('=', 1)
                try:
                    factors[rank.strip()] = int(value.strip())
                except ValueError:
                    print(f"Warning: Could not parse factor value '{value}' for rank '{rank}'")
    return factors


def read_layout_file(layout_file_path: str) -> List[LayoutNest]:
    """
    Read and parse a single timeloop-mapper.layout.yaml file.

    Args:
        layout_file_path: Path to the layout YAML file

    Returns:
        List of LayoutNest objects
    """
    try:
        with open(layout_file_path, 'r') as f:
            layout_data = yaml.safe_load(f)

        layout_nests = []
        if 'layout' in layout_data:
            for entry in layout_data['layout']:
                target = entry.get('target', '')
                layout_type = entry.get('type', '')
                factors_str = entry.get('factors', '')
                permutation = entry.get('permutation', '')

                factors = parse_factors_string(factors_str)
                ranks = list(permutation) if permutation else []

                layout_nest = LayoutNest(
                    target=target,
                    layout_type=layout_type,
                    factors=factors,
                    permutation=permutation,
                    ranks=ranks
                )
                layout_nests.append(layout_nest)

        return layout_nests

    except FileNotFoundError:
        print(f"Warning: Layout file not found: {layout_file_path}")
        return []
    except Exception as e:
        print(f"Error reading layout file {layout_file_path}: {e}")
        return []


def read_all_layer_dataspace_layouts(
    base_path: str = "/home/ubuntu/squareloop/benchmarks/squareloop_resent18",
    problem_path: str = "/home/ubuntu/squareloop/benchmarks/script/crosslayer_search/test/resnet18"
) -> Dict[int, LayerDataspaceLayouts]:
    """
    Read layout files from all layer directories and split factors by dataspaces.

    Args:
        base_path: Base path to the squareloop_resent18 directory
        problem_path: Path to the directory containing layer problem definition files

    Returns:
        Dict mapping layer_id to LayerDataspaceLayouts objects
    """
    layer_dataspace_layouts = {}

    try:
        # List all directories in the base path
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and item.startswith('resnet18_layer'):
                # Extract layer number
                layer_match = re.search(r'resnet18_layer(\d+)', item)
                if layer_match:
                    layer_id = int(layer_match.group(1))

                    # Read problem definition for this layer
                    problem_file_path = os.path.join(problem_path, f'resnet18_layer{layer_id}.yaml')
                    dataspace_to_ranks = read_layer_problem_definition(problem_file_path)

                    if not dataspace_to_ranks:
                        print(f"Warning: No dataspace rank definitions found for layer {layer_id}")
                        continue

                    # Read layout file for this layer
                    layout_file_path = os.path.join(item_path, 'timeloop-mapper.layout.yaml')
                    layout_nests = read_layout_file(layout_file_path)

                    if layout_nests:
                        # Split layout nests by dataspace
                        inputs_layouts = []
                        weights_layouts = []
                        outputs_layouts = []

                        for nest in layout_nests:
                            # Split factors by dataspace
                            dataspace_factors = split_factors_by_dataspace(nest.factors, dataspace_to_ranks)

                            # Create dataspace-specific layout nests
                            for dataspace_name, factors in dataspace_factors.items():
                                if factors:  # Only create if there are factors for this dataspace
                                    # Create permutation string for this dataspace
                                    dataspace_ranks = dataspace_to_ranks[dataspace_name]
                                    dataspace_permutation = ''.join([rank for rank in nest.permutation if rank in dataspace_ranks])

                                    dataspace_layout = DataspaceLayoutNest(
                                        dataspace_name=dataspace_name,
                                        target=nest.target,
                                        layout_type=nest.layout_type,
                                        factors=factors,
                                        permutation=dataspace_permutation,
                                        ranks=dataspace_ranks
                                    )

                                    # Add to appropriate list
                                    if dataspace_name == "Inputs":
                                        inputs_layouts.append(dataspace_layout)
                                    elif dataspace_name == "Weights":
                                        weights_layouts.append(dataspace_layout)
                                    elif dataspace_name == "Outputs":
                                        outputs_layouts.append(dataspace_layout)

                        # Create LayerDataspaceLayouts object
                        layer_dataspace_layout = LayerDataspaceLayouts(
                            layer_id=layer_id,
                            inputs_layouts=inputs_layouts,
                            weights_layouts=weights_layouts,
                            outputs_layouts=outputs_layouts,
                            targets=None  # Will be computed in __post_init__
                        )
                        layer_dataspace_layouts[layer_id] = layer_dataspace_layout
                    else:
                        print(f"Warning: No layout data found for layer {layer_id}")

        return layer_dataspace_layouts

    except Exception as e:
        print(f"Error reading layer dataspace layouts: {e}")
        return {}


def read_all_layer_layouts(base_path: str = "/home/ubuntu/squareloop/benchmarks/squareloop_resent18") -> Dict[int, LayerLayout]:
    """
    Read layout files from all layer directories.

    Args:
        base_path: Base path to the squareloop_resent18 directory

    Returns:
        Dict mapping layer_id to LayerLayout objects
    """
    layer_layouts = {}

    try:
        # List all directories in the base path
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and item.startswith('resnet18_layer'):
                # Extract layer number
                layer_match = re.search(r'resnet18_layer(\d+)', item)
                if layer_match:
                    layer_id = int(layer_match.group(1))

                    # Read layout file for this layer
                    layout_file_path = os.path.join(item_path, 'timeloop-mapper.layout.yaml')
                    layout_nests = read_layout_file(layout_file_path)

                    if layout_nests:
                        layer_layout = LayerLayout(
                            layer_id=layer_id,
                            layout_nests=layout_nests,
                            targets=None  # Will be computed in __post_init__
                        )
                        layer_layouts[layer_id] = layer_layout
                    else:
                        print(f"Warning: No layout data found for layer {layer_id}")

        return layer_layouts

    except Exception as e:
        print(f"Error reading layer layouts: {e}")
        return {}


def read_layer_problem_definition(layer_file_path: str) -> Dict[str, List[str]]:
    """
    Read layer problem definition file to extract dataspace-to-rank mappings.

    Args:
        layer_file_path: Path to the layer YAML file (e.g., resnet18_layer1.yaml)

    Returns:
        Dict mapping dataspace names to their rank lists
    """
    try:
        with open(layer_file_path, 'r') as f:
            layer_data = yaml.safe_load(f)

        dataspace_to_ranks = {}
        if 'problem' in layer_data and 'shape' in layer_data['problem']:
            data_spaces = layer_data['problem']['shape'].get('data-spaces', [])

            for dataspace in data_spaces:
                name = dataspace.get('name', '')
                ranks = dataspace.get('ranks', [])
                if name and ranks:
                    dataspace_to_ranks[name] = ranks

        return dataspace_to_ranks

    except FileNotFoundError:
        print(f"Warning: Layer problem file not found: {layer_file_path}")
        return {}
    except Exception as e:
        print(f"Error reading layer problem file {layer_file_path}: {e}")
        return {}


def split_factors_by_dataspace(factors: Dict[str, int], dataspace_to_ranks: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """
    Split layout factors by dataspace based on rank definitions.

    Args:
        factors: All factors from layout (e.g., {"C": 3, "K": 16, "R": 7, ...})
        dataspace_to_ranks: Mapping of dataspace names to their ranks

    Returns:
        Dict mapping dataspace names to their specific factors
    """
    dataspace_factors = {}

    for dataspace_name, ranks in dataspace_to_ranks.items():
        dataspace_factors[dataspace_name] = {}
        for rank in ranks:
            if rank in factors:
                dataspace_factors[dataspace_name][rank] = factors[rank]

    return dataspace_factors


def parse_dataspace_dependencies(dependency_file_path: str) -> Tuple[List[List[Dataspace]], Dict[int, LayerDataspaces]]:
    """
    Parse the dependency YAML file and create dataspace dependency groups.
    Rules:
    1. Exclude weights from dependency groups
    2. Input and output of same layer are in different groups
    3. Each dataspace appears in exactly one group

    Args:
        dependency_file_path (str): Path to the dependency YAML file

    Returns:
        Tuple of:
        - List of dataspace dependency groups
        - Dict of layer_id to LayerDataspaces mapping
    """
    try:
        with open(dependency_file_path, 'r') as f:
            dependencies = yaml.safe_load(f)

        # Get all layer IDs
        all_layers = set()
        for layer, deps in dependencies.items():
            all_layers.add(layer)
            if deps:
                for dep in deps:
                    all_layers.add(dep)

        # Create layer dataspaces
        layer_dataspaces = create_layer_dataspaces(sorted(all_layers))

        # Track which dataspaces have been assigned to groups
        assigned_dataspaces = set()
        dataspace_dependencies = []

        # Build connection groups: layer output -> dependent layer input
        for layer in sorted(all_layers):
            if layer in dependencies and dependencies[layer]:
                current_input = layer_dataspaces[layer].inputs

                # Group this layer's input with all its dependency outputs
                connection_group = []

                # Add current layer's input
                if str(current_input) not in assigned_dataspaces:
                    connection_group.append(current_input)
                    assigned_dataspaces.add(str(current_input))

                # Add outputs from all dependency layers
                for dep_layer in dependencies[layer]:
                    dep_output = layer_dataspaces[dep_layer].outputs
                    if str(dep_output) not in assigned_dataspaces:
                        connection_group.append(dep_output)
                        assigned_dataspaces.add(str(dep_output))

                if connection_group:
                    dataspace_dependencies.append(connection_group)

        # Handle remaining unassigned dataspaces as individual groups
        for layer_id in sorted(all_layers):
            layer_ds = layer_dataspaces[layer_id]

            if str(layer_ds.inputs) not in assigned_dataspaces:
                dataspace_dependencies.append([layer_ds.inputs])
                assigned_dataspaces.add(str(layer_ds.inputs))

            if str(layer_ds.outputs) not in assigned_dataspaces:
                dataspace_dependencies.append([layer_ds.outputs])
                assigned_dataspaces.add(str(layer_ds.outputs))

        return dataspace_dependencies, layer_dataspaces

    except FileNotFoundError:
        print(f"Error: Dependency file not found: {dependency_file_path}")
        return [], {}
    except Exception as e:
        print(f"Error parsing dependency file {dependency_file_path}: {e}")
        return [], {}


@dataclass
class SharedLayoutConstraint:
    """Represents a shared layout constraint for a dependency group with complete layout structure."""
    group_id: str
    dataspaces: List[str]               # List of dataspace names in this group
    targets: List[str]                  # Memory targets involved
    shared_layouts: Dict[str, Dict[str, any]]  # target -> {interline, intraline, authblock_lines, permutation}
    individual_factors: Dict[str, Dict[str, any]]  # dataspace_name -> {layout_type: factors}

    def __str__(self):
        return f"Group_{self.group_id}_Constraint"

    def __repr__(self):
        return self.__str__()


@dataclass
class NetworkLatencyResult:
    """Results of network latency calculation including reorganization overhead."""
    total_latency: int                          # Total network latency
    total_compute_latency: int                  # Sum of all layer compute latencies
    total_reorganization_overhead: int          # Sum of all data reorganization costs
    reorganization_transitions: List[Dict[str, any]]  # Details of each transition
    dependency_group_sequence: List[int]        # Sequence of dependency groups executed

    def __str__(self):
        return f"NetworkLatency(total={self.total_latency}, compute={self.total_compute_latency}, overhead={self.total_reorganization_overhead})"

    def __repr__(self):
        return self.__str__()


@dataclass
class ComprehensiveLatencyResult:
    """Comprehensive latency analysis combining all latency components."""
    processing_latency: int                     # Total compute latency from all layers
    layout_reorganization_latency: int          # Total layout reorganization overhead
    rehash_latency: int                         # Total rehash latency
    interlayer_memory_latency: int              # max(layout_reorganization, rehash)
    total_latency: int                          # processing + interlayer_memory

    # Detailed ratios
    layout_reorganization_ratio: float          # layout_reorganization / processing
    rehash_ratio: float                         # rehash / processing
    interlayer_memory_ratio: float              # interlayer_memory / processing
    processing_efficiency: float                # processing / total

    # Breakdown details
    layer_latencies: Dict[int, int]             # Individual layer latencies
    group_rehash_latencies: Dict[int, int]      # Rehash latency per group
    reorganization_details: List[Dict[str, any]]  # Reorganization transition details

    def __str__(self):
        return f"ComprehensiveLatency(total={self.total_latency}, processing={self.processing_latency}, interlayer_memory={self.interlayer_memory_latency})"

    def __repr__(self):
        return self.__str__()


def gcd(a: int, b: int) -> int:
    """Calculate Greatest Common Divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Calculate Least Common Multiple of two numbers."""
    return abs(a * b) // gcd(a, b)


def lcm_multiple(numbers: List[int]) -> int:
    """Calculate LCM of multiple numbers."""
    if not numbers:
        return 1
    if len(numbers) == 1:
        return numbers[0]

    result = numbers[0]
    for i in range(1, len(numbers)):
        result = lcm(result, numbers[i])
    return result


def find_dataspace_layouts(dataspace: 'Dataspace', layer_dataspace_layouts: Dict[int, 'LayerDataspaceLayouts']) -> List['DataspaceLayoutNest']:
    """
    Find all layout information for a specific dataspace.

    Args:
        dataspace: Dataspace object (e.g., Layer2_Inputs)
        layer_dataspace_layouts: Dict mapping layer_id to LayerDataspaceLayouts

    Returns:
        List of DataspaceLayoutNest objects for this dataspace
    """
    layer_id = dataspace.layer_id
    dataspace_type = dataspace.dataspace_type

    if layer_id not in layer_dataspace_layouts:
        return []

    layer_layout = layer_dataspace_layouts[layer_id]
    return layer_layout.get_layouts_by_dataspace(dataspace_type)


def calculate_shared_layout_constraints(
    dataspace_deps: List[List['Dataspace']],
    layer_dataspace_layouts: Dict[int, 'LayerDataspaceLayouts']
) -> List[SharedLayoutConstraint]:
    """
    Calculate shared layout constraints for each dependency group.
    Uses rank mapping: N↔N, V↔L, H↔P, W↔Q between inputs and outputs.
    Creates complete layout structure with maximal intraline factors for equivalent ranks.

    Args:
        dataspace_deps: List of dependency groups, each containing Dataspace objects
        layer_dataspace_layouts: Dict mapping layer_id to LayerDataspaceLayouts

    Returns:
        List of SharedLayoutConstraint objects with complete layout structure
    """
    shared_constraints = []

    # Define rank mapping between inputs and outputs
    output_to_input_mapping = {'N': 'N', 'L': 'V', 'P': 'H', 'Q': 'W'}

    for group_id, dependency_group in enumerate(dataspace_deps, 1):
        if not dependency_group:
            continue

        # Collect all layouts organized by target
        target_layouts = {}  # target -> layout_type -> [DataspaceLayoutNest objects]
        all_targets = set()

        for dataspace in dependency_group:
            dataspace_layouts = find_dataspace_layouts(dataspace, layer_dataspace_layouts)

            for layout in dataspace_layouts:
                all_targets.add(layout.target)
                target = layout.target
                layout_type = layout.layout_type

                if target not in target_layouts:
                    target_layouts[target] = {}
                if layout_type not in target_layouts[target]:
                    target_layouts[target][layout_type] = []

                # Store layout with dataspace info
                layout_with_dataspace = {
                    'layout': layout,
                    'dataspace': dataspace,
                    'dataspace_name': str(dataspace)
                }
                target_layouts[target][layout_type].append(layout_with_dataspace)

        # Create shared constraint for each target that has complete layout data
        for target in all_targets:
            if target not in target_layouts:
                continue

            # Check if we have all required layout types
            required_layout_types = ['interline', 'intraline', 'authblock_lines']
            if not all(lt in target_layouts[target] for lt in required_layout_types):
                continue

            # Check if all dataspaces are represented
            dataspaces_in_target = set()
            for layout_type, layout_list in target_layouts[target].items():
                for layout_info in layout_list:
                    dataspaces_in_target.add(layout_info['dataspace_name'])

            expected_dataspaces = set(str(ds) for ds in dependency_group)
            if dataspaces_in_target < expected_dataspaces:
                continue

            # Build shared layout for this target
            shared_layout = {}
            individual_factors = {}

            for layout_type in required_layout_types:
                layout_list = target_layouts[target][layout_type]

                if layout_type == 'intraline':
                    # For intraline, calculate max of equivalent ranks
                    equivalent_rank_factors = {}  # equivalent_rank -> {dataspace: factor}
                    sample_permutation = None

                    for layout_info in layout_list:
                        layout = layout_info['layout']
                        dataspace = layout_info['dataspace']
                        dataspace_name = layout_info['dataspace_name']

                        # Store permutation (should be same for all)
                        if sample_permutation is None:
                            sample_permutation = layout.permutation

                        # Store individual factors
                        if dataspace_name not in individual_factors:
                            individual_factors[dataspace_name] = {}
                        individual_factors[dataspace_name][layout_type] = layout.factors.copy()

                        # Group by equivalent ranks
                        for rank, factor in layout.factors.items():
                            equivalent_rank = rank

                            # Map output ranks to input ranks for equivalent grouping
                            if dataspace.dataspace_type == "Outputs" and rank in output_to_input_mapping:
                                equivalent_rank = output_to_input_mapping[rank]

                            if equivalent_rank not in equivalent_rank_factors:
                                equivalent_rank_factors[equivalent_rank] = {}
                            equivalent_rank_factors[equivalent_rank][dataspace_name] = factor

                    # Calculate max factors for each equivalent rank
                    max_factors = {}
                    for equivalent_rank, dataspace_factors in equivalent_rank_factors.items():
                        max_factors[equivalent_rank] = max(dataspace_factors.values())

                    shared_layout[layout_type] = {
                        'factors': max_factors,
                        'permutation': sample_permutation
                    }

                else:
                    # For interline and authblock_lines, use the first dataspace's values
                    # (they should be consistent across dataspaces in dependency group)
                    if layout_list:
                        first_layout = layout_list[0]['layout']
                        shared_layout[layout_type] = {
                            'factors': first_layout.factors.copy(),
                            'permutation': first_layout.permutation
                        }

                        # Store individual factors for reference
                        for layout_info in layout_list:
                            layout = layout_info['layout']
                            dataspace_name = layout_info['dataspace_name']

                            if dataspace_name not in individual_factors:
                                individual_factors[dataspace_name] = {}
                            individual_factors[dataspace_name][layout_type] = layout.factors.copy()

            # Create shared layout constraint for this target
            if shared_layout:
                constraint = SharedLayoutConstraint(
                    group_id=f"{group_id}_{target}",
                    dataspaces=[str(ds) for ds in dependency_group],
                    targets=[target],
                    shared_layouts={target: shared_layout},
                    individual_factors=individual_factors
                )
                shared_constraints.append(constraint)

    return shared_constraints


def calculate_network_latency(
    layer_latencies: Dict[int, int],
    dataspace_deps: List[List['Dataspace']],
    shared_constraints: List[SharedLayoutConstraint],
    memory_ports: int = 2
) -> NetworkLatencyResult:
    """
    Calculate total network latency including data reorganization overhead.

    Args:
        layer_latencies: Dict mapping layer_id to compute latency
        dataspace_deps: List of dependency groups containing dataspaces
        shared_constraints: List of shared layout constraints for each group/target
        memory_ports: Number of available memory ports (default: 2)

    Returns:
        NetworkLatencyResult with total latency breakdown
    """
    # Calculate total compute latency
    total_compute_latency = sum(layer_latencies.values())

    # Build mapping from dependency group to DRAM shared constraints
    group_to_dram_constraint = {}
    for constraint in shared_constraints:
        if constraint.targets and 'DRAM' in constraint.targets:
            # Extract group number from constraint ID (e.g., "1_DRAM" -> 1)
            group_num_str = constraint.group_id.split('_')[0]
            try:
                group_num = int(group_num_str)
                group_to_dram_constraint[group_num] = constraint
            except ValueError:
                continue

    # Identify dependency group transitions (connections between groups)
    transitions = []
    group_sequence = []

    # Build a mapping of which layers belong to which dependency groups
    layer_to_group = {}
    for group_idx, dep_group in enumerate(dataspace_deps, 1):
        for dataspace in dep_group:
            layer_id = dataspace.layer_id
            if layer_id not in layer_to_group:
                layer_to_group[layer_id] = []
            layer_to_group[layer_id].append(group_idx)

    # Find transitions: when output of one group becomes input of another
    for group_idx, dep_group in enumerate(dataspace_deps, 1):
        group_sequence.append(group_idx)

        # Look for outputs in this group
        for dataspace in dep_group:
            if dataspace.dataspace_type == "Outputs":
                output_layer_id = dataspace.layer_id

                # Find which layers depend on this output (look at next layer)
                next_layer_id = output_layer_id + 1
                if next_layer_id in layer_to_group:
                    # Find which groups the next layer belongs to
                    next_groups = layer_to_group[next_layer_id]

                    for next_group_idx in next_groups:
                        if next_group_idx != group_idx:  # Different group
                            # Check if both groups have DRAM constraints
                            if group_idx in group_to_dram_constraint and next_group_idx in group_to_dram_constraint:
                                transitions.append({
                                    'from_group': group_idx,
                                    'to_group': next_group_idx,
                                    'from_layer': output_layer_id,
                                    'to_layer': next_layer_id,
                                    'dataspace_type': 'Output->Input'
                                })

    # Calculate reorganization overhead for each transition
    total_reorganization_overhead = 0
    reorganization_details = []

    for transition in transitions:
        from_group = transition['from_group']
        to_group = transition['to_group']

        # Get DRAM constraints for both groups
        from_constraint = group_to_dram_constraint[from_group]
        to_constraint = group_to_dram_constraint[to_group]

        # Get DRAM interline factors for both groups
        from_dram_layout = from_constraint.shared_layouts.get('DRAM', {})
        to_dram_layout = to_constraint.shared_layouts.get('DRAM', {})

        from_interline = from_dram_layout.get('interline', {}).get('factors', {})
        to_interline = to_dram_layout.get('interline', {}).get('factors', {})

        # Calculate number of lines for reorganization
        lines_read = 1
        for factor in from_interline.values():
            lines_read *= factor

        lines_written = 1
        for factor in to_interline.values():
            lines_written *= factor

        # Total reorganization cost = (lines read + lines written) / memory_ports
        reorganization_cost = (lines_read + lines_written) // memory_ports

        total_reorganization_overhead += reorganization_cost

        reorganization_details.append({
            'transition': f"Group {from_group} -> Group {to_group}",
            'from_layer': transition['from_layer'],
            'to_layer': transition['to_layer'],
            'lines_read': lines_read,
            'lines_written': lines_written,
            'reorganization_cost': reorganization_cost,
            'from_interline_factors': from_interline,
            'to_interline_factors': to_interline
        })

    # Calculate total network latency
    total_latency = total_compute_latency + total_reorganization_overhead

    return NetworkLatencyResult(
        total_latency=total_latency,
        total_compute_latency=total_compute_latency,
        total_reorganization_overhead=total_reorganization_overhead,
        reorganization_transitions=reorganization_details,
        dependency_group_sequence=group_sequence
    )


def read_crypto_config(crypto_config_path: str = "/home/ubuntu/squareloop/benchmarks/secureloop-cross-eval/AES-GCM-parallel.yaml") -> int:
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


def calculate_rehash_latency(
    dataspace_deps: List[List['Dataspace']],
    layer_dataspace_layouts: Dict[int, 'LayerDataspaceLayouts'],
    crypto_config_path: str = "/home/ubuntu/squareloop/benchmarks/secureloop-cross-eval/AES-GCM-parallel.yaml"
) -> Dict[int, int]:
    """
    Calculate rehash latency for each dependency group.

    Rehash assumes loading output of a layer from DRAM completely and then
    regroup data into authblock_lines and then generate a hash for it.
    The latency is calculated as accumulation of "product of intraline*interline/authblock_lines
    factors of all ranks of each data" across all data within the dependency group,
    multiplied by the authentication cycle cost per block.

    Args:
        dataspace_deps: List of dependency groups containing dataspaces
        layer_dataspace_layouts: Dict mapping layer_id to LayerDataspaceLayouts
        crypto_config_path: Path to crypto configuration file for auth cycle cost

    Returns:
        Dict mapping group_id to rehash latency
    """
    word_bits = 16
    # Read authentication cycle cost per block from crypto config
    auth_additional_cycle_per_block, auth_cycle_per_datapath, enc_cycle_per_datapath, datapath = read_crypto_config(crypto_config_path)
    print(f"Using auth_additional_cycle_per_datapath = {auth_additional_cycle_per_block}, auth_cycle_per_datapath = {auth_cycle_per_datapath}, enc_cycle_per_datapath = {enc_cycle_per_datapath} and datapath = {datapath} from crypto config")

    group_rehash_latencies = {}

    for group_idx, dep_group in enumerate(dataspace_deps, 1):
        total_rehash_latency = 0
        # Find all output dataspaces in this group
        for dataspace in dep_group:
            if dataspace.dataspace_type == "Outputs":
                # Get all layout information for this output dataspace
                dataspace_layouts = find_dataspace_layouts(dataspace, layer_dataspace_layouts)

                # Organize layouts by type for DRAM target
                layout_by_type = {}
                for layout in dataspace_layouts:
                    if layout.target == "DRAM":
                        layout_by_type[layout.layout_type] = layout

                # Check if we have all required layout types
                required_types = ["intraline", "interline", "authblock_lines"]
                if all(layout_type in layout_by_type for layout_type in required_types):

                    intraline_factors = layout_by_type["intraline"].factors
                    interline_factors = layout_by_type["interline"].factors
                    authblock_lines_factors = layout_by_type["authblock_lines"].factors

                    # Get all ranks present in any of the factor dictionaries
                    all_ranks = set(intraline_factors.keys()) | set(interline_factors.keys()) | set(authblock_lines_factors.keys())
                    # Calculate product of (intraline * interline / authblock_lines * auth_cycle_per_datapath) for all ranks
                    num_authblock_lines = 1.0
                    authblock_lines_size = 1.0
     
                    for rank in all_ranks:
                        intraline_factor = intraline_factors.get(rank, 1)
                        interline_factor = interline_factors.get(rank, 1)
                        authblock_lines_factor = authblock_lines_factors.get(rank, 1)

                        if authblock_lines_factor > 0:
                            num_authblock_lines *= np.ceil(interline_factor / authblock_lines_factor)
                            authblock_lines_size *= authblock_lines_factor * intraline_factor
                        else:
                            # If authblock_lines factor is 0, assume factor of 1 for this rank
                            num_authblock_lines *= interline_factor
                            authblock_lines_size *= authblock_lines_factor * intraline_factor

                    latency_per_authblock = authblock_lines_size * word_bits / datapath * np.max(enc_cycle_per_datapath, auth_cycle_per_datapath) + auth_additional_cycle_per_block
                    total_rehash_latency += num_authblock_lines * latency_per_authblock 
        
        group_rehash_latencies[group_idx] = int(total_rehash_latency)

    return group_rehash_latencies


def calculate_comprehensive_latency_analysis(
    layer_latencies: Dict[int, int],
    network_latency: NetworkLatencyResult,
    group_rehash_latencies: Dict[int, int]
) -> ComprehensiveLatencyResult:
    """
    Calculate comprehensive latency analysis combining all latency components.

    Args:
        layer_latencies: Dict mapping layer_id to compute latency
        network_latency: NetworkLatencyResult from network latency calculation
        group_rehash_latencies: Dict mapping group_id to rehash latency

    Returns:
        ComprehensiveLatencyResult with detailed latency breakdown and ratios
    """
    # Calculate processing latency (sum of all layer compute latencies)
    processing_latency = sum(layer_latencies.values())

    # Get layout reorganization latency
    layout_reorganization_latency = network_latency.total_reorganization_overhead

    # Calculate total rehash latency
    rehash_latency = sum(group_rehash_latencies.values())

    # Calculate interlayer memory latency as max of reorganization and rehash
    interlayer_memory_latency = max(layout_reorganization_latency, rehash_latency)

    # Calculate total latency
    total_latency = processing_latency + interlayer_memory_latency

    # Calculate detailed ratios (avoid division by zero)
    if processing_latency > 0:
        layout_reorganization_ratio = layout_reorganization_latency / processing_latency
        rehash_ratio = rehash_latency / processing_latency
        interlayer_memory_ratio = interlayer_memory_latency / processing_latency
    else:
        layout_reorganization_ratio = 0.0
        rehash_ratio = 0.0
        interlayer_memory_ratio = 0.0

    if total_latency > 0:
        processing_efficiency = processing_latency / total_latency
    else:
        processing_efficiency = 0.0

    return ComprehensiveLatencyResult(
        processing_latency=processing_latency,
        layout_reorganization_latency=layout_reorganization_latency,
        rehash_latency=rehash_latency,
        interlayer_memory_latency=interlayer_memory_latency,
        total_latency=total_latency,
        layout_reorganization_ratio=layout_reorganization_ratio,
        rehash_ratio=rehash_ratio,
        interlayer_memory_ratio=interlayer_memory_ratio,
        processing_efficiency=processing_efficiency,
        layer_latencies=layer_latencies.copy(),
        group_rehash_latencies=group_rehash_latencies.copy(),
        reorganization_details=network_latency.reorganization_transitions.copy()
    )




def merge_permutations(perm1: str, perm2: str) -> str:
    """
    Merge two permutation strings, preserving order and avoiding duplicates.
    
    Args:
        perm1: First permutation string
        perm2: Second permutation string
        
    Returns:
        Merged permutation string with all unique ranks
    """
    if not perm1:
        return perm2
    if not perm2:
        return perm1
    
    # Convert to lists for easier manipulation
    ranks1 = list(perm1)
    ranks2 = list(perm2)
    
    # Start with the first permutation as base
    merged_ranks = ranks1.copy()
    
    # Add ranks from second permutation that aren't already present
    # Insert them in positions that maintain some logical order
    for rank in ranks2:
        if rank not in merged_ranks:
            # Try to find a good position to insert the new rank
            # For now, append to the end, but could be made smarter
            merged_ranks.append(rank)
    
    return ''.join(merged_ranks)


def generate_constrained_layout_file(
    layer_id: int,
    layer_dataspace_layout: 'LayerDataspaceLayouts',
    output_file_path: str
) -> None:
    """
    Generate a layout.yaml file for a layer with constrained layouts.
    Groups all dataspaces that share the same type and target into one block.

    Args:
        layer_id: Layer identifier
        layer_dataspace_layout: LayerDataspaceLayouts object with constrained layouts
        output_file_path: Path where to write the layout.yaml file
    """
    layout_data = {'layout': []}
    
    # Combine all layout nests from inputs, weights, and outputs
    all_layouts = []
    all_layouts.extend(layer_dataspace_layout.inputs_layouts)
    all_layouts.extend(layer_dataspace_layout.weights_layouts)
    all_layouts.extend(layer_dataspace_layout.outputs_layouts)
    
    # Group layouts by (target, layout_type) combination
    grouped_layouts = {}  # (target, layout_type) -> combined_layout_info
    
    for layout_nest in all_layouts:
        key = (layout_nest.target, layout_nest.layout_type)
        
        if key not in grouped_layouts:
            # First layout for this (target, type) combination
            grouped_layouts[key] = {
                'target': layout_nest.target,
                'type': layout_nest.layout_type,
                'factors': layout_nest.factors.copy(),
                'permutation': layout_nest.permutation,
                'dataspace_count': 1,
                'dataspaces': [layout_nest.dataspace_name]
            }
        else:
            # Merge with existing layout for this (target, type) combination
            existing = grouped_layouts[key]
            
            # For factors, we should use the same values (since they're constrained to be the same)
            # But let's verify they match and take the union of all ranks
            for rank, factor in layout_nest.factors.items():
                if rank in existing['factors']:
                    if existing['factors'][rank] != factor:
                        print(f"    Warning: Factor mismatch for {key} rank {rank}: {existing['factors'][rank]} vs {factor}")
                else:
                    existing['factors'][rank] = factor
            
            # Merge permutations: combine all unique ranks while preserving order
            old_permutation = existing['permutation']
            existing['permutation'] = merge_permutations(existing['permutation'], layout_nest.permutation)
            if old_permutation != existing['permutation']:
                print(f"    Merged permutation for {key}: '{old_permutation}' + '{layout_nest.permutation}' = '{existing['permutation']}'")
            
            existing['dataspace_count'] += 1
            existing['dataspaces'].append(layout_nest.dataspace_name)
    
    # Convert grouped layouts to YAML format
    for (target, layout_type), layout_info in grouped_layouts.items():
        # Convert factors dict to string format
        factors_str = " ".join([f"{rank}={factor}" for rank, factor in sorted(layout_info['factors'].items())])
        
        layout_entry = {
            'target': layout_info['target'],
            'type': layout_info['type'],
            'factors': factors_str,
            'permutation': layout_info['permutation']
        }
        layout_data['layout'].append(layout_entry)
        
        # Log grouping information
        if layout_info['dataspace_count'] > 1:
            print(f"    Grouped {layout_info['dataspace_count']} dataspaces for {target}_{layout_type}: {layout_info['dataspaces']}")
    
    # Write to YAML file
    try:
        with open(output_file_path, 'w') as f:
            yaml.dump(layout_data, f, default_flow_style=False, sort_keys=False)
        print(f"  Generated layout file: {output_file_path} with {len(layout_data['layout'])} grouped layout blocks")
    except Exception as e:
        print(f"  Error writing layout file {output_file_path}: {e}")



def compare_layout_nests(
    original_layouts: List['DataspaceLayoutNest'],
    constrained_layouts: List['DataspaceLayoutNest'],
    context: str
) -> int:
    """
    Compare two lists of DataspaceLayoutNest objects and count differences.

    Args:
        original_layouts: Original layout nests
        constrained_layouts: Constrained layout nests
        context: Description for logging

    Returns:
        Number of layout changes detected
    """
    changes = 0
    
    # Create lookup dictionaries by target and layout_type
    original_lookup = {}
    for layout in original_layouts:
        key = (layout.target, layout.layout_type)
        original_lookup[key] = layout
    
    constrained_lookup = {}
    for layout in constrained_layouts:
        key = (layout.target, layout.layout_type)
        constrained_lookup[key] = layout
    
    # Compare layouts
    all_keys = set(original_lookup.keys()) | set(constrained_lookup.keys())
    
    for key in all_keys:
        target, layout_type = key
        
        if key not in original_lookup:
            print(f"    {context} - New layout: {target}_{layout_type}")
            changes += 1
        elif key not in constrained_lookup:
            print(f"    {context} - Removed layout: {target}_{layout_type}")
            changes += 1
        else:
            original_layout = original_lookup[key]
            constrained_layout = constrained_lookup[key]
            
            # Compare factors
            if original_layout.factors != constrained_layout.factors:
                print(f"    {context} - Changed factors for {target}_{layout_type}:")
                print(f"      Original: {original_layout.factors}")
                print(f"      Constrained: {constrained_layout.factors}")
                changes += 1
            
            # Compare permutation
            if original_layout.permutation != constrained_layout.permutation:
                print(f"    {context} - Changed permutation for {target}_{layout_type}:")
                print(f"      Original: {original_layout.permutation}")
                print(f"      Constrained: {constrained_layout.permutation}")
                changes += 1
    
    return changes


if __name__ == "__main__":
  layer_num = 19
  layer_latency = {}
  for i in range(1, layer_num):
    base_path = f"/home/ubuntu/squareloop/benchmarks/squareloop_resent18/resnet18_layer{i}/timeloop-mapper.stats.txt"
    layer_cycles = extract_cycles_from_stats(base_path)
    layer_latency[i] = layer_cycles
  print("Layer latencies:", layer_latency)

  # Parse dependency groups
  dependency_file = "/home/ubuntu/squareloop/benchmarks/script/crosslayer_search/test/resnet18/resnet18_dependent.yaml"

  print("\n" + "="*60)
  print("DATASPACE-BASED DEPENDENCY GROUPS (NO WEIGHTS)")
  print("="*60)
  dataspace_deps, layer_dataspaces = parse_dataspace_dependencies(dependency_file)
  for i, group in enumerate(dataspace_deps):
    print(f"Dataspace Group {i+1}: {group}")

  # Read dataspace-split layer layouts
  print("\n" + "="*60)
  print("DATASPACE-SPLIT LAYER LAYOUTS")
  print("="*60)
  base_path = "/home/ubuntu/squareloop/benchmarks/squareloop_resent18_old"
  problem_path = "/home/ubuntu/squareloop/benchmarks/script/crosslayer_search/test/resnet18"
  layer_dataspace_layouts = read_all_layer_dataspace_layouts(base_path, problem_path)
  print(f"Successfully read dataspace layouts for {len(layer_dataspace_layouts)} layers")

  # Read architecture block-size information
  print("\n" + "="*60)
  print("ARCHITECTURE BLOCK SIZES")
  print("="*60)
  target_block_sizes = read_architecture_block_sizes()
  print(f"Successfully read block sizes for {len(target_block_sizes)} targets")

  for target in target_block_sizes:
    mesh_info = f" (x{target.mesh_count})" if target.mesh_count > 1 else ""
    print(f"{target.name}: {target.block_size} data ({target.class_type}){mesh_info}")

  # Summary table
  print(f"\nBlock Size Summary:")
  block_size_groups = {}
  for target in target_block_sizes:
    if target.block_size not in block_size_groups:
      block_size_groups[target.block_size] = []
    block_size_groups[target.block_size].append(target.name)

  for block_size in sorted(block_size_groups.keys()):
    targets = block_size_groups[block_size]
    print(f"  {block_size} data: {', '.join(targets)}")

  # Calculate shared layout constraints for dependency groups
  print("\n" + "="*60)
  print("SHARED LAYOUT CONSTRAINTS FOR DEPENDENCY GROUPS")
  print("="*60)
  shared_constraints = calculate_shared_layout_constraints(dataspace_deps, layer_dataspace_layouts)
  print(f"Generated {len(shared_constraints)} shared layout constraints")

  # Display detailed constraint information
  for constraint in shared_constraints[:3]:  # Show first 3 as examples
    print(f"\n{constraint}:")
    print(f"  Dataspaces: {constraint.dataspaces}")
    print(f"  Targets: {constraint.targets}")

    # Show shared layout structure for each target
    for target, layout_info in constraint.shared_layouts.items():
      print(f"  Shared Layout for {target}:")
      for layout_type, layout_data in layout_info.items():
        factors = layout_data['factors']
        permutation = layout_data['permutation']
        if layout_type == 'intraline':
          print(f"    {layout_type} (MAX): {factors}")
        else:
          print(f"    {layout_type}: {factors}")
        print(f"      permutation: {permutation}")

    # Show individual factors for comparison
    print(f"  Individual Factors by Layout Type:")
    for dataspace_name, layout_factors in constraint.individual_factors.items():
      print(f"    {dataspace_name}:")
      for layout_type, factors in layout_factors.items():
        print(f"      {layout_type}: {factors}")

  if len(shared_constraints) > 3:
    print(f"\n  ... and {len(shared_constraints) - 3} more constraints")

  # Summary of constraint complexity
  print(f"\nConstraint Summary:")
  total_intraline_factors = 0
  rank_usage = {}

  for constraint in shared_constraints:
    for target, layout_info in constraint.shared_layouts.items():
      if 'intraline' in layout_info:
        intraline_factors = layout_info['intraline']['factors']
        total_intraline_factors += len(intraline_factors)

        # Count rank usage
        for rank in intraline_factors.keys():
          rank_usage[rank] = rank_usage.get(rank, 0) + 1

  print(f"  Total MAX intraline factors across all constraints: {total_intraline_factors}")
  print(f"  Most frequently constrained ranks:")
  for rank, count in sorted(rank_usage.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"    {rank}: appears in {count} constraints")

  # Show constraint distribution by target
  target_counts = {}
  for constraint in shared_constraints:
    target = constraint.targets[0] if constraint.targets else "unknown"
    target_counts[target] = target_counts.get(target, 0) + 1

  print(f"\nConstraints per target:")
  for target, count in sorted(target_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"    {target}: {count} constraints")

  # Calculate network latency with reorganization overhead
  print("\n" + "="*60)
  print("NETWORK LATENCY WITH REORGANIZATION OVERHEAD")
  print("="*60)
  network_latency = calculate_network_latency(layer_latency, dataspace_deps, shared_constraints)

  print(f"Network Latency Summary:")
  print(f"  Total Network Latency: {network_latency.total_latency:,} cycles")
  print(f"  Total Compute Latency: {network_latency.total_compute_latency:,} cycles")
  print(f"  Total Reorganization Overhead: {network_latency.total_reorganization_overhead:,} cycles")

  if network_latency.total_compute_latency > 0:
    overhead_percentage = (network_latency.total_reorganization_overhead / network_latency.total_compute_latency) * 100
    print(f"  Reorganization Overhead: {overhead_percentage:.2f}% of compute latency")

  print(f"\nData Reorganization Transitions:")
  if network_latency.reorganization_transitions:
    for i, transition in enumerate(network_latency.reorganization_transitions, 1):
      print(f"  Transition {i}: {transition['transition']}")
      print(f"    Layers: {transition['from_layer']} -> {transition['to_layer']}")
      print(f"    Lines Read: {transition['lines_read']:,}")
      print(f"    Lines Written: {transition['lines_written']:,}")
      print(f"    Reorganization Cost: {transition['reorganization_cost']:,} cycles")
      print(f"    From Interline Factors: {transition['from_interline_factors']}")
      print(f"    To Interline Factors: {transition['to_interline_factors']}")
      print()
  else:
    print("    No reorganization transitions detected")

  print(f"Dependency Group Execution Sequence: {network_latency.dependency_group_sequence}")

  # Calculate rehash latency for dependency groups
  print("\n" + "="*60)
  print("REHASH LATENCY FOR DEPENDENCY GROUPS")
  print("="*60)
  group_rehash_latencies = calculate_rehash_latency(dataspace_deps, layer_dataspace_layouts)

  print(f"Rehash Latency Summary:")
  total_rehash_latency = sum(group_rehash_latencies.values())
  print(f"  Total Rehash Latency: {total_rehash_latency:,} cycles")

  print(f"\nRehash Latency by Dependency Group:")
  for group_idx in sorted(group_rehash_latencies.keys()):
    latency = group_rehash_latencies[group_idx]
    print(f"  Group {group_idx}: {latency:,} cycles")

    # Show which dataspaces are in this group
    if group_idx <= len(dataspace_deps):
      group_dataspaces = dataspace_deps[group_idx - 1]
      output_dataspaces = [ds for ds in group_dataspaces if ds.dataspace_type == "Outputs"]
      if output_dataspaces:
        print(f"    Output dataspaces: {[str(ds) for ds in output_dataspaces]}")

  # Compare rehash latency with compute and reorganization overhead
  if network_latency.total_compute_latency > 0:
    rehash_vs_compute = (total_rehash_latency / network_latency.total_compute_latency) * 100
    print(f"\nRehash Latency Comparison:")
    print(f"  Rehash vs Compute Latency: {rehash_vs_compute:.2f}% of compute latency")

    if network_latency.total_reorganization_overhead > 0:
      rehash_vs_reorg = (total_rehash_latency / network_latency.total_reorganization_overhead) * 100
      print(f"  Rehash vs Reorganization Overhead: {rehash_vs_reorg:.2f}% of reorganization overhead")

    total_with_rehash = network_latency.total_latency + total_rehash_latency
    print(f"  Total Latency with Rehash: {total_with_rehash:,} cycles")

    rehash_percentage = (total_rehash_latency / total_with_rehash) * 100
    print(f"  Rehash Overhead: {rehash_percentage:.2f}% of total latency with rehash")

  # Calculate comprehensive latency analysis
  print("\n" + "="*60)
  print("COMPREHENSIVE LATENCY ANALYSIS")
  print("="*60)
  comprehensive_latency = calculate_comprehensive_latency_analysis(
      layer_latency, network_latency, group_rehash_latencies
  )

  print(f"Comprehensive Latency Breakdown:")
  print(f"  Processing Latency (layers): {comprehensive_latency.processing_latency:,} cycles")
  print(f"  Layout Reorganization Latency: {comprehensive_latency.layout_reorganization_latency:,} cycles")
  print(f"  Rehash Latency: {comprehensive_latency.rehash_latency:,} cycles")
  print(f"  Interlayer Memory Latency (max): {comprehensive_latency.interlayer_memory_latency:,} cycles")
  print(f"  Total Latency: {comprehensive_latency.total_latency:,} cycles")

  print(f"\nDetailed Ratios (relative to processing latency):")
  print(f"  Layout Reorganization Ratio: {comprehensive_latency.layout_reorganization_ratio:.4f} ({comprehensive_latency.layout_reorganization_ratio*100:.2f}%)")
  print(f"  Rehash Ratio: {comprehensive_latency.rehash_ratio:.4f} ({comprehensive_latency.rehash_ratio*100:.2f}%)")
  print(f"  Interlayer Memory Ratio: {comprehensive_latency.interlayer_memory_ratio:.4f} ({comprehensive_latency.interlayer_memory_ratio*100:.2f}%)")
  print(f"  Processing Efficiency: {comprehensive_latency.processing_efficiency:.4f} ({comprehensive_latency.processing_efficiency*100:.2f}%)")

  print(f"\nLatency Component Analysis:")
  print(f"  Processing vs Total: {comprehensive_latency.processing_efficiency*100:.2f}%")
  print(f"  Interlayer Memory vs Total: {(comprehensive_latency.interlayer_memory_latency/comprehensive_latency.total_latency)*100:.2f}%")
