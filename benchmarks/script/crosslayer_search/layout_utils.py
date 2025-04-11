import os
import yaml

# Parse layout factors into a dict
def parse_layout_to_dict(factors):
    assert (type(factors) == str)

    a = factors.split(" ")
    b = {}
    for aa in a:
        k = aa.split('=')[0]
        v = eval(aa.split('=')[1])
        b[k] = v
    return b

# Determine if two layer's layouts satisfy the cross-layer dependency condition
# (i.e., previous layer's output layout should match next layer's input layout)
def check_layout_dependency(layout1, layout2, workload1, workload2):
    """
    Layout1: previous layer's layout definition (layout.yaml)
    Layout2: next layer's layout definition (layout.yaml)
    Workload1: previous layer's layer definition (layer.yaml)
    Workload2: next layer's layout definition (layer.yaml)

    Constraint:
    Layout1's output layout == Layout2's input layout
    """
    # print(layout1, layout2)
    # 1) Check two workloads and determine ranks and projections
    with open(workload1, 'r') as f:
        w1 = yaml.safe_load(f)
    
    with open(workload2, 'r') as f:
        w2 = yaml.safe_load(f)

    prev_output_rank = None
    prev_output_projection = None
    for dataspace in w1['problem']['shape']['data-spaces']:
        if dataspace['name'] == 'Outputs':
            prev_output_rank = dataspace['ranks']
            prev_output_projection = dataspace['projection']

    next_input_rank = None
    next_input_projection = None
    for dataspace in w2['problem']['shape']['data-spaces']:
        if dataspace['name'] == 'Inputs':
            next_input_rank = dataspace['ranks']
            next_input_projection = dataspace['projection']

    prev_output_workload_dim = w1['problem']['instance']
    next_input_workload_dim = w2['problem']['instance']

    # 2) Get the intra/inter-line factors for the ranks of our interest
    with open(layout1, 'r') as f:
        l1 = yaml.safe_load(f)
    
    with open(layout2, 'r') as f:
        l2 = yaml.safe_load(f)

    prev_layout_inter = None
    prev_layout_intra = None
    for layout in l1['layout']:
        if layout['target'] == 'MainMemory':
            if layout['type'] == 'interline':
                prev_layout_inter = parse_layout_to_dict(layout['factors'])
            elif layout['type'] == 'intraline':
                prev_layout_intra = parse_layout_to_dict(layout['factors'])
            else:
                continue
    
    next_layout_inter = None
    next_layout_intra = None
    for layout in l2['layout']:
        if layout['target'] == 'MainMemory':
            if layout['type'] == 'interline':
                next_layout_inter = parse_layout_to_dict(layout['factors'])
            elif layout['type'] == 'intraline':
                next_layout_intra = parse_layout_to_dict(layout['factors'])
            else:
                continue

    # 3) Interline layout eval
    interline_layout_match = True
    for idx, (o_rank, i_rank) in enumerate(zip(prev_output_rank, next_input_rank)):
        # o_proj = prev_output_projection[idx]
        # i_proj = next_input_projection[idx]
        # print(o_rank, i_rank, prev_layout_inter[o_rank], next_layout_inter[i_rank])
        interline_layout_match = interline_layout_match & (prev_layout_inter[o_rank] == next_layout_inter[i_rank])

    # 4) Intraline layout eval
    intraline_layout_match = True
    for idx, (o_rank, i_rank) in enumerate(zip(prev_output_rank, next_input_rank)):
        # o_proj = prev_output_projection[idx]
        # i_proj = next_input_projection[idx]
        interline_layout_match = interline_layout_match & (prev_layout_intra[o_rank] == next_layout_intra[i_rank])

    # print(intraline_layout_match, interline_layout_match)
    return (interline_layout_match & intraline_layout_match)

if __name__ == '__main__':
    # Match case: Should be true
    print(check_layout_dependency('../../layout/alexnet/vector_256_SRCQPMNHW_Wx8Hx4_3.yaml', \
                                  '../../layout/alexnet/vector_256_SRCQPMNHW_Wx8Hx4_4.yaml', \
                                  'alexnet/AlexNet_layer3.yaml', \
                                  'alexnet/AlexNet_layer4.yaml'))