import subprocess
import sys
import os

sys.path.append(os.path.dirname(__file__))
from utils import *

rtl_dir = benchmarks_dir+"rtl-validation/"

cmd_crypto = squareloop_ld_path+" "+squareloop_model+" "+rtl_dir+"arch/baseline.yaml "+rtl_dir+"arch/components/* "+rtl_dir+"conv_2_batch1/layer2/mapping.yaml "+rtl_dir+"workloads/conv_2_batch1_layer2.yaml "+rtl_dir+"layout/layer2.yaml "+rtl_dir+"ASCON-128a.yaml"
cmd_plain = squareloop_ld_path+" "+squareloop_model+" "+rtl_dir+"arch/baseline.yaml "+rtl_dir+"arch/components/* "+rtl_dir+"conv_2_batch1/layer2/mapping.yaml "+rtl_dir+"workloads/conv_2_batch1_layer2.yaml "+rtl_dir+"layout/layer2.yaml"

print("Plain:")
result = subprocess.run(cmd_plain, shell=True, capture_output=True, text=True)
lines = result.stdout.strip().splitlines()
if lines:
    print(lines[-1])

print("\nCrypto:")
result = subprocess.run(cmd_crypto, shell=True, capture_output=True, text=True)
lines = result.stdout.strip().splitlines()
if lines:
    print(lines[-1])