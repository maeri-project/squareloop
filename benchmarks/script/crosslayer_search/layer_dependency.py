import os
import sys

import copy
import yaml

from typing import List, Set, Dict
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import torchlens as tl

import pandas as pd

conv_workload_template = {'problem': {'instance': {'C': 1, 'M': 1, 'N': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1, 'Wdilation': 1, 'Wstride': 1, 'Hdilation': 1, 'Hstride': 1}, \
                                      'shape': {'name': 'CNN_Layer', 'dimensions': ['C', 'M', 'R', 'S', 'N', 'P', 'Q'], \
                                                'coefficients': [{'default': 1, 'name': 'Wstride'}, {'default': 1, 'name': 'Hstride'}, {'default': 1, 'name': 'Wdilation'}, {'default': 1, 'name': 'Hdilation'}], \
                                                'data-spaces': [{'name': 'Weights', 'projection': [[['C']], [['M']], [['R']], [['S']]], 'ranks': ['C', 'M', 'R', 'S']}, {'name': 'Inputs', 'projection': [[['N']], [['C']], [['R', 'Wdilation'], ['P', 'Wstride']], [['S', 'Hdilation'], ['Q', 'Hstride']]], 'ranks': ['N', 'C', 'H', 'W']}, {'name': 'Outputs', 'projection': [[['N']], [['M']], [['P']], [['Q']]], 'read_write': True, 'ranks': ['N', 'M', 'P', 'Q']}]}}}
conv_dw_workload_template = {'problem': {'instance': {'C': 1, 'N': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1, 'Wdilation': 1, 'Wstride': 1, 'Hdilation': 1, 'Hstride': 1}, \
                                         'shape': {'name': 'CNN_Layer', 'dimensions': ['C', 'R', 'S', 'N', 'P', 'Q'], \
                                                   'coefficients': [{'default': 1, 'name': 'Wstride'}, {'default': 1, 'name': 'Hstride'}, {'default': 1, 'name': 'Wdilation'}, {'default': 1, 'name': 'Hdilation'}], \
                                                   'data-spaces': [{'name': 'Weights', 'projection': [[['C']], [['R']], [['S']]], 'ranks': ['C', 'R', 'S']}, {'name': 'Inputs', 'projection': [[['N']], [['C']], [['R', 'Wdilation'], ['P', 'Wstride']], [['S', 'Hdilation'], ['Q', 'Hstride']]], 'ranks': ['N', 'C', 'H', 'W']}, {'name': 'Outputs', 'projection': [[['N']], [['C']], [['P']], [['Q']]], 'read_write': True, 'ranks': ['N', 'C', 'P', 'Q']}]}}}
# linear workload -> same as conv, but set P=Q=R=S=1
linear_workload_template = {'problem': {'instance': {'C': 1, 'M': 1, 'N': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1, 'Wdilation': 1, 'Wstride': 1, 'Hdilation': 1, 'Hstride': 1}, \
                                        'shape': {'name': 'Linear_Layer', 'dimensions': ['C', 'M', 'R', 'S', 'N', 'P', 'Q'], \
                                                  'coefficients': [{'default': 1, 'name': 'Wstride'}, {'default': 1, 'name': 'Hstride'}, {'default': 1, 'name': 'Wdilation'}, {'default': 1, 'name': 'Hdilation'}], \
                                                  'data-spaces': [{'name': 'Weights', 'projection': [[['C']], [['M']], [['R']], [['S']]], 'ranks': ['C', 'M', 'R', 'S']}, {'name': 'Inputs', 'projection': [[['N']], [['C']], [['R', 'Wdilation'], ['P', 'Wstride']], [['S', 'Hdilation'], ['Q', 'Hstride']]], 'ranks': ['N', 'C', 'H', 'W']}, {'name': 'Outputs', 'projection': [[['N']], [['M']], [['P']], [['Q']]], 'read_write': True, 'ranks': ['N', 'M', 'P', 'Q']}]}}}


def generate_torchlens_history(module, input, save_to):
    module_history = tl.log_forward_pass(module, input)
    df = module_history.to_pandas()
    df.to_csv(save_to, index=False)

def generate_workload_yaml(df, model, workload_name, workload_save_to):
    layer_cnt = 1
    supported_layer_types = ['conv2d', 'linear']
    # type_to_torch_module = {'conv2d': nn.Conv2d, 'linear': nn.Linear}
    print("Currently supported layer types: {}".format(' '.join(supported_layer_types)))

    # get model parameter sizes for each named modules
    param_list = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            param_list[name] = {'C': m.in_channels, 'M': m.out_channels, 'RS': m.kernel_size, 'stride': m.stride, 'dilation': m.dilation, 'groups': m.groups}
        elif isinstance(m, nn.Linear):
            param_list[name] = {'C': m.in_features, 'M': m.out_features}
        else:
            continue

    for idx, row in df.iterrows():
        if row['layer_type'] in supported_layer_types:
            output_tensor_shape = eval(row['tensor_shape'])

            module_name_list = eval(row['modules_exited'])
            module_name = None
            if len(module_name_list) > 1:
                for m in module_name_list:
                    if m in param_list.keys():
                        module_name = m
                        break
            else:
                module_name = module_name_list[0]
            weight_tensor_info = param_list[module_name]
            
            # while there can be multiple parents (i.e., bmm between two matrices in self attention)
            # we are currently looking for conv2d and linear -> #parents should be 1
            parent_layer_name = eval(row['parent_layers'])[0]
            input_tensor_shape = eval(df.loc[df['layer_label'] == parent_layer_name, 'tensor_shape'].values[0])

            # generate the yaml file
            N = output_tensor_shape[0]
            P = output_tensor_shape[2] if len(output_tensor_shape) == 3 else 1
            Q = output_tensor_shape[3] if len(output_tensor_shape) == 3 else 1
            M = weight_tensor_info['M']
            C = weight_tensor_info['C']
            
            if row['layer_type'] == 'conv2d':
                R, S = weight_tensor_info['RS']
                Hstride, Wstride = weight_tensor_info['stride']
                Hdilation, Wdilation = weight_tensor_info['dilation']
            else:
                R, S = 1, 1
                Hstride, Wstride = 1, 1
                Hdilation, Wdilation = 1, 1
            
            if row['layer_type'] == 'conv2d':
                # depthwise convolution?
                if (weight_tensor_info['groups'] > 1) and (weight_tensor_info['groups'] == input_tensor_shape[1]):
                    workload = copy.deepcopy(conv_dw_workload_template)
                    workload['problem']['instance']['N'] = N
                    workload['problem']['instance']['C'] = C
                    workload['problem']['instance']['P'] = P
                    workload['problem']['instance']['Q'] = Q
                    workload['problem']['instance']['R'] = R
                    workload['problem']['instance']['S'] = S
                    workload['problem']['instance']['Hdilation'] = Hdilation
                    workload['problem']['instance']['Wdilation'] = Wdilation
                    workload['problem']['instance']['Hstride'] = Hstride
                    workload['problem']['instance']['Wstride'] = Wstride
                
                # normal convolution -- note that grouped conv with groups > 1 but not depthwise is not supported
                elif (weight_tensor_info['groups'] > 1):
                    raise NotImplementedError("Grouped convolution that is NOT depthwise-separable is not supported!")

                else:
                    workload = copy.deepcopy(conv_workload_template)
                    workload['problem']['instance']['N'] = N
                    workload['problem']['instance']['M'] = M
                    workload['problem']['instance']['C'] = C
                    workload['problem']['instance']['P'] = P
                    workload['problem']['instance']['Q'] = Q
                    workload['problem']['instance']['R'] = R
                    workload['problem']['instance']['S'] = S
                    workload['problem']['instance']['Hdilation'] = Hdilation
                    workload['problem']['instance']['Wdilation'] = Wdilation
                    workload['problem']['instance']['Hstride'] = Hstride
                    workload['problem']['instance']['Wstride'] = Wstride

            elif row['layer_type'] == 'linear':
                workload = copy.deepcopy(linear_workload_template)
                workload['problem']['instance']['N'] = N
                workload['problem']['instance']['M'] = M
                workload['problem']['instance']['C'] = C

            else:
                raise NotImplementedError("Unsupported layer type {}".format(row['layer_type']))
            
            with open(os.path.join(workload_save_to, '{}_layer{}.yaml'.format(workload_name, layer_cnt)), 'w') as f:
                yaml.dump(workload, f)
            
            layer_cnt += 1
    
def determine_dependency(df, workload_name, save_to):
    # iterate through conv2d/linear layers, and find the parent layer
    # if parent is another conv2d/lienar -> direct dependency
    # if parent is non-conv2d/linear layer
    # --> if the parent is in the list of 'on-the-fly' operations, check its parent
    # --> if not 'on-the-fly', not dependent

    supported_layer_types = ['conv2d', 'linear']
    onthefly_types = ['relu', 'dropout']

    def find_consecutive_layers(name):
        result = set()
        visited = set()

        def dfs(curr_name):
            if curr_name in visited:
                return
            visited.add(curr_name)
            parents = eval(df.loc[df['layer_label'] == curr_name, 'parent_layers'].values[0])
            if len(parents) == 0:
                return
            for p in parents:
                p_type = df.loc[df['layer_label'] == p, 'layer_type'].values[0]
                if p_type in supported_layer_types:
                    result.add(p)
                else:
                    dfs(p)
        
        dfs(name)
        return list(result)
    
    # First, build a directed graph representing the layer dependencies
    # We'll represent it as an adjacency list where key is a node and value is its children
    graph: Dict[str, List[str]] = {}
    
    # Also build a reverse graph for parents
    parents: Dict[str, List[str]] = {}
    
    # Initialize empty lists for each node
    for layer in df['layer_label']:
        graph[layer] = []
        parents[layer] = []

    layer_label_to_idx: Dict[str, int] = {}
    layer_cnt = 1
    
    # Fill in the graph based on parent_layers relationships
    for _, row in df.iterrows():
        layer = row['layer_label']
        parent_layers_str = row['parent_layers']
        
        # Skip if there are no parent layers
        if not parent_layers_str or pd.isna(parent_layers_str):
            continue
            
        # Convert parent_layers string to list (assuming comma-separated format)
        parent_layers = eval(parent_layers_str)
        
        # Add edges from parents to this layer
        for parent in parent_layers:
            if parent in graph:
                graph[parent].append(layer)
                parents[layer].append(parent)

        if row['layer_type'] in supported_layer_types:
            layer_label_to_idx[layer] = layer_cnt
            layer_cnt += 1
                
    def get_operations_between(node1, node2):
        
        # Check if both nodes exist in the graph
        if node1 not in graph or node2 not in graph:
            return []
        
        # Find all nodes reachable from node1 using BFS
        reachable_from_node1 = set()
        queue = deque([node1])
        while queue:
            current = queue.popleft()
            reachable_from_node1.add(current)
            for child in graph[current]:
                if child not in reachable_from_node1:
                    queue.append(child)
        
        # If node2 is not reachable from node1, return empty list
        if node2 not in reachable_from_node1:
            return []
        
        # Find all ancestors of node2 using reverse BFS
        ancestors_of_node2 = set()
        queue = deque([node2])
        while queue:
            current = queue.popleft()
            ancestors_of_node2.add(current)
            for parent in parents[current]:
                if parent not in ancestors_of_node2:
                    queue.append(parent)
        
        # Find nodes that are both reachable from node1 and ancestors of node2
        # These are the nodes on paths between node1 and node2
        between_nodes = reachable_from_node1.intersection(ancestors_of_node2)
        
        # Remove node1 and node2 from the result
        between_nodes.discard(node1)
        between_nodes.discard(node2)
        
        # Return as a sorted list
        return sorted(list(between_nodes))
    
    # iterate through model and find all conv2d/linear layers
    consecutive_dict: Dict[int, List[int]] = {}
    dependent_dict: Dict[int, List[int]] = {}

    for label, idx in layer_label_to_idx.items():
        consecutive = find_consecutive_layers(label)
        consecutive_dict[idx] = [layer_label_to_idx[x] for x in consecutive]

        # determine whether each element in consecutive is dependent as well
        dependent = []
        for c in consecutive:
            between = get_operations_between(c, label)
            between_types = [df.loc[df['layer_label'] == b, 'layer_type'].values[0] for b in between]
            if all(element in onthefly_types for element in between_types):
                dependent.append(c)
            else:
                continue
        dependent_dict[idx] = [layer_label_to_idx[x] for x in dependent] if len(dependent) > 0 else []
    
    with open(os.path.join(save_to, '{}_consecutive.yaml'.format(workload_name)), 'w') as f:
        yaml.dump(consecutive_dict, f)
    with open(os.path.join(save_to, '{}_dependent.yaml'.format(workload_name)), 'w') as f:
        yaml.dump(dependent_dict, f)

def model_analysis(model, input, save_dir, model_tag):
    # check if save_dir exists, if not create
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    torchlens_save_path = os.path.join(save_dir, 'torchlens.csv')
    generate_torchlens_history(model, input, torchlens_save_path)
    df = pd.read_csv(torchlens_save_path)
    generate_workload_yaml(df, model, model_tag, save_dir)
    determine_dependency(df, model_tag, save_dir)
    



            
            

