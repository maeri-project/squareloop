import os
import sys

def run_timeloop_mapper(arch_path, arch_name_prefix, layer_path, layer_name_prefix, mapper_path, arch_search_constraint, map_search_constraint, result_path):
  if not os.path.exists(result_path):
    os.makedirs(result_path)
  os.chdir(result_path)
  print("Running timeloop-mapper in {}".format(os.getcwd()))
  os.system("timeloop-mapper {} {} {} {} {} {}".format(arch_path, arch_name_prefix, layer_path, mapper_path, arch_search_constraint, map_search_constraint))

if __name__ == "__main__":
  arch_path = "/home/ubuntu/squareloop/benchmarks/secureloop-cross-eval/arch/baseline.yaml"
  arch_name_component = "/home/ubuntu/squareloop/benchmarks/secureloop-cross-eval/arch/components/*"
  arch_search_constraint = "/home/ubuntu/squareloop/benchmarks/secureloop-cross-eval/arch_constraint/eyeriss_like_arch_constraints.yaml"
  map_search_constraint = "/home/ubuntu/squareloop/benchmarks/secureloop-cross-eval/arch_constraint/eyeriss_like_map_constraints.yaml"
  mapper_path = "/home/ubuntu/squareloop/benchmarks/secureloop-sanity-check/test_mapper.yaml"
  
  for layer_idx in range(1, 20):
    layer_path = f"/home/ubuntu/squareloop/benchmarks/script/crosslayer_search/test/resnet18/resnet18_layer{layer_idx}.yaml"
    layer_name_prefix = f"resnet18_layer{layer_idx}"
    result_path = f"/home/ubuntu/squareloop/benchmarks/squareloop_resent18/{layer_name_prefix}"
    run_timeloop_mapper(arch_path, arch_name_component, layer_path, layer_name_prefix, mapper_path, arch_search_constraint, map_search_constraint, result_path)
