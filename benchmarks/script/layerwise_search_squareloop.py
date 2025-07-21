import os
import sys

def run_timeloop_mapper(arch_path, arch_name_prefix, layer_path, layer_name_prefix, mapper_path, result_path):
  if not os.path.exists(result_path):
    os.makedirs(result_path)
  os.chdir(result_path)
  print("Running timeloop-mapper in {}".format(os.getcwd()))
  os.system("timeloop-mapper {} {} {} {}".format(arch_path, arch_name_prefix, layer_path, mapper_path))

if __name__ == "__main__":
  arch_path = "/home/ubuntu/squareloop/benchmarks/secureloop-cross-eval/arch/baseline.yaml"
  arch_name_component = "/home/ubuntu/squareloop/benchmarks/secureloop-cross-eval/arch/components/*"
  mapper_path = "/home/ubuntu/squareloop/benchmarks/secureloop-sanity-check/test_mapper.yaml"
  
  for layer_idx in range(1, 21):
    layer_path = f"/home/ubuntu/squareloop/benchmarks/secureloop-cross-eval/layer{layer_idx}/resnet18_batch1_layer{layer_idx}.yaml"
    layer_name_prefix = f"resnet18_layer{layer_idx}"
    result_path = f"/home/ubuntu/squareloop/benchmarks/squareloop_resent18/{layer_name_prefix}"
    run_timeloop_mapper(arch_path, arch_name_component, layer_path, layer_name_prefix, mapper_path, result_path)
