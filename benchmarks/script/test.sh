source ~/.setup.sh && mkdir layer1 && cd layer1 && timeloop-mapper ../arch_designs/benchmarks/arch_designs/eyeriss_like/arch/* /home/ubuntu/squareloop/benchmarks/arch_designs/eyeriss_like/constraints_gemm/* /home/ubuntu/squareloop/benchmarks/layer_shapes/resnet50/resnet50_1.yaml && cd ..

source ~/.setup.sh && mkdir layer2 && cd layer2 && timeloop-mapper ../arch_designs/benchmarks/arch_designs/eyeriss_like/arch/* /home/ubuntu/squareloop/benchmarks/arch_designs/eyeriss_like/constraints_gemm/* /home/ubuntu/squareloop/benchmarks/layer_shapes/resnet50/resnet50_2.yaml && cd ..

source ~/.setup.sh && mkdir layer3 && cd layer3 && timeloop-mapper ../arch_designs/benchmarks/arch_designs/eyeriss_like/arch/* /home/ubuntu/squareloop/benchmarks/arch_designs/eyeriss_like/constraints_gemm/* /home/ubuntu/squareloop/benchmarks/layer_shapes/resnet50/resnet50_3.yaml && cd ..

source ~/.setup.sh && mkdir layer4 && cd layer4 && timeloop-mapper ../arch_designs/benchmarks/arch_designs/eyeriss_like/arch/* /home/ubuntu/squareloop/benchmarks/arch_designs/eyeriss_like/constraints_gemm/* /home/ubuntu/squareloop/benchmarks/layer_shapes/resnet50/resnet50_4.yaml && cd ..

source ~/.setup.sh && mkdir layer5 && cd layer5 && timeloop-mapper ../arch_designs/benchmarks/arch_designs/eyeriss_like/arch/* /home/ubuntu/squareloop/benchmarks/arch_designs/eyeriss_like/constraints_gemm/* /home/ubuntu/squareloop/benchmarks/layer_shapes/resnet50/resnet50_5.yaml && cd ..

source ~/.setup.sh && mkdir layer6 && cd layer6 && timeloop-mapper ../arch_designs/benchmarks/arch_designs/eyeriss_like/arch/* /home/ubuntu/squareloop/benchmarks/arch_designs/eyeriss_like/constraints_gemm/* /home/ubuntu/squareloop/benchmarks/layer_shapes/resnet50/resnet50_6.yaml && cd ..
