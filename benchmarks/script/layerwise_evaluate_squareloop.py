import os
import sys


def extract_layer_folders(base_path):
  """
  Extract all folders within the given path that start with 'layer'

  Args:
    base_path (str): The directory path to search in

  Returns:
    list: List of folder names that start with 'layer'
  """
  layer_folders = []

  if not os.path.exists(base_path):
    print(f"Warning: Path {base_path} does not exist")
    return layer_folders

  try:
    # Get all items in the directory
    items = os.listdir(base_path)

    # Filter for directories that start with 'layer'
    for item in items:
      item_path = os.path.join(base_path, item)
      if os.path.isdir(item_path) and item.startswith('layer'):
        layer_folders.append(item)

    # Sort the folders for consistent ordering
    layer_folders.sort()

  except PermissionError:
    print(f"Error: Permission denied to access {base_path}")
  except Exception as e:
    print(f"Error reading directory {base_path}: {e}")

  return layer_folders

def run_timeloop_model(arch_path, arch_name_prefix, crypto_path, mapping_path, layout_path, layer_path, layer_name_prefix, result_path):
  if not os.path.exists(result_path):
    os.makedirs(result_path)
  os.chdir(result_path)
  print("Running timeloop-model in {}".format(os.getcwd()))
  os.system("timeloop-model {} {} {} {} {} {}".format(arch_path, arch_name_prefix, crypto_path, mapping_path, layout_path, layer_path))

if __name__ == "__main__":
  path_to_cross_eval = "/home/ubuntu/squareloop/benchmarks/secureloop-cross-eval"
  arch_path = f"{path_to_cross_eval}/arch/baseline.yaml"
  arch_name_component = f"{path_to_cross_eval}/arch/components/*"
  arch_search_constraint = f"{path_to_cross_eval}/arch_constraint/eyeriss_like_arch_constraints.yaml"
  map_search_constraint = f"{path_to_cross_eval}/arch_constraint/eyeriss_like_map_constraints.yaml"
  mapper_path = "/home/ubuntu/squareloop/benchmarks/mapper/mapper_squarelooop.yaml"
  crypto_path = f"{path_to_cross_eval}/crypto.yaml"

  layer_folders = extract_layer_folders(path_to_cross_eval)
  layer_folders.sort()
  print("Found layer folders:", layer_folders)
  for layer_folder in layer_folders:
    # Extract layer ID from folder name (e.g. "layer5" -> 5)
    layer_id = int(layer_folder.replace("layer", ""))
    layer_path = f"/home/ubuntu/squareloop/benchmarks/script/crosslayer_search/test/resnet18/resnet18_layer{layer_id}.yaml"
    mapping_path = f"{path_to_cross_eval}/{layer_folder}/resnet18_batch1_layer{layer_id}.yaml"
    layout_path = f"{path_to_cross_eval}/{layer_folder}/layout{layer_id}.yaml"
    layer_name_prefix = f"resnet18_layer{layer_id}"
    result_path = f"/home/ubuntu/squareloop/benchmarks/squareloop_resent18_secureloop/{layer_name_prefix}"
    run_timeloop_model(arch_path, arch_name_component, crypto_path, mapping_path, layout_path, layer_path, layer_name_prefix, result_path)
