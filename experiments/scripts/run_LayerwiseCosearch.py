import os
import sys

sys.path.append(os.path.dirname(__file__))
# Configure a path prefix. Defaults to absolute workspace path; can be overridden
# by env var PATH_PREFIX or the first CLI argument.
path_prefix = os.environ.get("PATH_PREFIX", "/home/ubuntu/squareloop/")
if len(sys.argv) > 1:
  path_prefix = sys.argv[1]
if path_prefix and not path_prefix.endswith("/"):
  path_prefix = path_prefix + "/"

def run_timeloop_mapper(arch_path, arch_name_prefix, layer_path, layer_name_prefix, mapper_path, arch_search_constraint, map_search_constraint, result_path):
  if not os.path.exists(result_path):
    os.makedirs(result_path)
  os.chdir(result_path)
  print("Running timeloop-mapper in {}".format(os.getcwd()))
  os.system("timeloop-mapper {} {} {} {} {} {}".format(arch_path, arch_name_prefix, layer_path, mapper_path, arch_search_constraint, map_search_constraint))

def run_LayerwiseCosearch():
  print("LayerwiseCosearch")

  result_dir = path_prefix + 'results/LayerwiseCosearch/'

  csv_file = result_dir + 'stats.csv'
    
  arch_path = path_prefix + "benchmarks/arch_designs/eyeriss_like/arch/eyeriss_like.yaml"
  arch_name_component = path_prefix + "benchmarks/arch_designs/eyeriss_like/arch/components/*"
  arch_search_constraint = path_prefix + "benchmarks/arch_designs/eyeriss_like/constraints/eyeriss_like_arch_constraints.yaml"
  map_search_constraint = path_prefix + "benchmarks/arch_designs/eyeriss_like/constraints/eyeriss_like_map_constraints.yaml"
  mapper_path = path_prefix + "benchmarks/mapper/mapper_squareloop.yaml"

  for layer_idx in range(1, 21):
    layer_path = f"{path_prefix}benchmarks/layer_shapes/resnet18/resnet18_batch1_layer{layer_idx}.yaml"
    layer_name_prefix = f"resnet18_layer{layer_idx}"
    result_path = f"{path_prefix}experiments/results/LayerwiseCosearch/resnet18/{layer_name_prefix}"
    run_timeloop_mapper(arch_path, arch_name_component, layer_path, layer_name_prefix, mapper_path, arch_search_constraint, map_search_constraint, result_path)

run_LayerwiseCosearch()
