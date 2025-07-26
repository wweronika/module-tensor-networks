import numpy as np
import opt_einsum as oe
from collections import defaultdict

# Conversion function
def convert_to_complex(number_str):
    if 'i' in number_str:
        number_str = number_str.replace('i', 'j')  # Replace 'i' with Python's imaginary unit 'j'
        return complex(number_str)
    else:
        return complex(float(number_str), 0)

def get_H_local_and_d_from_files(labels_file, values_file, size_file):
    # Read the labels from the first file
    with open(labels_file, 'r') as lf:
        labels = [tuple(map(int, line.strip().strip('()').split(','))) for line in lf]

    # Read the complex values from the second file
    with open(values_file, 'r') as vf:
        values = [convert_to_complex(line) for line in vf]

    with open(size_file, 'r') as sf:
        max_tensor_size_tuple = tuple(map(int, sf.readline().strip().split(',')))[4:]
        max_tensor_size_list = list(max_tensor_size_tuple)

    if len(labels) != len(values):
        raise ValueError("The number of labels and values must be the same.")

    # Initialize the dictionary of tensors
    tensor_dict = {}
    d_dict = get_tensor_shape_for_module_quadruple(labels, max_tensor_size_tuple)
    # print(d_dict)

    for i, label_set in enumerate(labels):
        # Split the label set into keys and indices
        keys = label_set[:4]
        indices = label_set[4:]

        # Subtract 1 to account for Matlab indexing
        keys = tuple([k - 1 for k in keys])
        indices = tuple([j - 1 for j in indices])

        A, B, C, D = keys

        if keys not in tensor_dict:
            # Determine the size of the tensor dynamically
            tensor_shape = [max_tensor_size_list[0]]
            tensor_shape += [d_dict[A, B], d_dict[C, D]]
            tensor_shape += max_tensor_size_list[3:]
            tensor_shape = tuple(tensor_shape)
            # print(f"{keys} -> {tensor_shape}")
            tensor_dict[keys] = np.zeros(tensor_shape, dtype=complex)

        # Assign the value to the tensor at the specified indices
        tensor_dict[keys][indices] = values[i]
    return tensor_dict, d_dict

def get_H_and_d_from_files(N, labels_file, values_file, size_file):
    H_local, d_dict = get_H_local_and_d_from_files(labels_file, values_file, size_file)
    coeffs = np.array([1, 1, 1, 1])
    # coeffs = np.array([1, 0, 0, 0, 1])
    for key in H_local.keys(): 
        H_local[key] = oe.contract('abcde,e->abcd', H_local[key], coeffs)
        # H_local[key] = np.transpose(H_local[key],(0,2,3,1))
    H = [H_local for i in range(N)]
    return H, d_dict

def get_H_for_vumps(values_file):
    data = []

    max_a = max_b = 0
    shape_tracker = defaultdict(lambda: [0, 0, 0, 0])

    # Assume format of 1 line is: (a, b) (A, B, C, D) (i, j, k, l) value
    with open(values_file, "r") as f:
        for line in f:
            parts = line.strip().split(';')
            a, b = eval(parts[0])
            A, B, C, D = eval(parts[1])
            i, j, k, l = eval(parts[2])
            value = float(parts[3])

            data.append(((a, b), (A, B, C, D), (i, j, k, l), value))

            max_a = max(max_a, a)
            max_b = max(max_b, b)
            key = (a, b, A, B, C, D)

            # Infer shape of tensors
            shape_tracker[key][0] = max(shape_tracker[key][0], i + 1)
            shape_tracker[key][1] = max(shape_tracker[key][1], j + 1)
            shape_tracker[key][2] = max(shape_tracker[key][2], k + 1)
            shape_tracker[key][3] = max(shape_tracker[key][3], l + 1)

    # Initialize H
    H = [[defaultdict(lambda: None) for _ in range(max_b + 1)] for _ in range(max_a + 1)]

    # Allocate tensors
    for (a, b, A, B, C, D), (dim_i, dim_j, dim_k, dim_l) in shape_tracker.items():
        H[a][b][(A, B, C, D)] = np.zeros((dim_i, dim_j, dim_k, dim_l))

    # Fill values
    for (a, b), (A, B, C, D), (i, j, k, l), value in data:
        H[a][b][(A, B, C, D)][i, j, k, l] = value

    return H

def get_tensor_shape_for_module_quadruple(indices_list, max_tensor_size_tuple):
    max_dim_map = defaultdict(list)
    
    # Process each index entry
    for indices in indices_list:
        if len(indices) < 6:
            raise ValueError("Each index tuple must have at least 6 elements.")
        
        A, B, C, D = [i-1 for i in indices[:4]]  # First four indices
        v1 = indices[5] - 1 # First inferred dimension
        v2 = indices[6] - 1 # Second inferred dimension
        
        # Update max value for (A, B)
        if (A, B) not in max_dim_map:
            max_dim_map[(A, B)] = v1 + 1
        else:
            max_dim_map[(A, B)] = max(max_dim_map[(A, B)], v1 + 1)
        
        # Update max value for (C, D)
        if (C, D) not in max_dim_map:
            max_dim_map[(C, D)] = v2 + 1
        else:
            max_dim_map[(C, D)] = max(max_dim_map[(C, D)], v2 + 1)
    
    return max_dim_map

def get_allowed_module_pairs_from_H(H):
    allowed_module_pairs = set() # Horizontal, i.e. along the MPS
    allowed_vertical_module_pairs = set() # Vertical, i.e. across the virtual bond of the MPO
    for key in H[0].keys():
        A, B, C, D = key
        allowed_module_pairs.add((A, B))
        allowed_module_pairs.add((C, D))
        allowed_vertical_module_pairs.add((A, C))
        allowed_vertical_module_pairs.add((B, D))
    return allowed_module_pairs, allowed_vertical_module_pairs

def get_allowed_module_pairs_from_H_vumps(H):
    max_a = len(H)
    max_b = len(H[0])

    allowed_module_pairs = set() # Horizontal, i.e. along the MPS
    allowed_vertical_module_pairs = [set() for i in range(max_a)]# Vertical, i.e. across the virtual bond of the MPO

    for a in range(max_a):
        for b in range(max_b):
            for key in H[a][b].keys():
                A, B, C, D = key
                allowed_module_pairs.add((A, B))
                allowed_module_pairs.add((C, D))
                allowed_vertical_module_pairs[a].add((A, C))
                allowed_vertical_module_pairs[b].add((B, D))
    return allowed_module_pairs, allowed_vertical_module_pairs

def get_chi_mpo(allowed_vertical_module_pairs, H):
    # Assumes H has indices (ABCD) (aijb).
    # Version for DMRG and other algorithms with no explicit upper triangular structure.
    # Does not perform consistency checks.
    local_H_shape = next(iter(H[0].values())).shape # Shape of any tensor on the 0th site
    chi_mpo = {}
    for pair in allowed_vertical_module_pairs:
        chi_mpo[pair] = local_H_shape[0]
    return chi_mpo

# Read dimensions of the MPO for each combination of module indices.
# Assumes lines have the format (a, b) (A, B, C, D) (a2, i, j, b2) and a2 = b2
# Works for VUMPS and other algorithms with explicit upper triangular structure.
# chi_CA: dimension of the cellular automaton index
def get_chi_mpo_vumps(input_file, chi_CA):
    chi_mpo = [{} for i in range(chi_CA)]

    with open(input_file, "r") as f:
        for line in f:
            parts = line.strip().split(';')
            print(parts)
            a, b = eval(parts[0])
            A, B, C, D = eval(parts[1])
            a2, i, j, b2 = eval(parts[2])
            chi_mpo[a][A, C] = a2

    return chi_mpo

# def get_chi_mpo_vumps(H):
#     # Assumes H has indices (ab) (ABCD) (ijkl). Range of a must be the same as range of b.
#     # Version for VUMPS and other algorithms with explicit upper triangular structure.
#     # Performs consistency checks.

#     max_a = len(H)
    
#     # chi[a][b][(A, C)] = dim_i (vertical)
#     chi_mpo = [defaultdict(int) for _ in range(max_a)]     
#     # d[(A, B)] = dim_j (horizontal), independent of (a, b)
#     d = {}

#     for a in range(max_a):
#         for b in range(max_a):
#             for (A, B, C, D), tensor in H[a][b].items():
#                 if tensor is None or tensor.ndim != 4:
#                     continue

#                 dim_i, dim_j, dim_k, dim_l = tensor.shape

#                 key_vertical = (A, C)
#                 key_horizontal = (A, B)

#                 # chi[a][b][A, C] = dim_i
#                 if key_vertical in chi_mpo[a]:
#                     if chi_mpo[a][key_vertical] != dim_i:
#                         raise ValueError(
#                             f"Inconsistent chi[{a}][{key_vertical}]: {chi_mpo[a][key_vertical]} vs {dim_i}"
#                         )
#                 else:
#                     chi_mpo[a][key_vertical] = dim_i

#                 # d[(A, B)] = dim_j
#                 if key_horizontal in d:
#                     if d[key_horizontal] != dim_j:
#                         raise ValueError(
#                             f"Inconsistent d[{key_horizontal}]: {d[key_horizontal]} vs {dim_j}"
#                         )
#                 else:
#                     d[key_horizontal] = dim_j

#     # Convert chi_mpo to regular dicts
#     chi_mpo = [dict(chi_a) for chi_a in chi_mpo]

#     return chi_mpo, d