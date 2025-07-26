import numpy as np
import opt_einsum as oe
from itertools import product

# Matrix product state
def mps_vec_to_dict(mps_vec, allowed_module_pairs_sorted, chi, d):
    mps_dict = {}
    i = 0
    for A, B in allowed_module_pairs_sorted:
        size = chi[A] * d[A, B] * chi[B]
        mps_dict[A, B] = mps_vec[i : i + size].reshape(chi[A], d[A, B], chi[B])
        i += size
    return mps_dict

# Matrix product state
def mps_dict_to_vec(mps_dict, allowed_module_pairs_sorted):
    vecs = []
    for A, B in allowed_module_pairs_sorted:
        vecs.append(mps_dict[A, B].flatten())  # Flatten to 1D
    return np.concatenate(vecs)

# Two-site matrix product state
def two_site_mps_fixed_B_dict_to_vec(two_site_dict, B, i_bond, chi, d, modules_sorted, N):
    # Assume keys are (A, C)
    result = None
    for A in modules_sorted:
        row_block = None
        for C in modules_sorted:
            if (A, C) in two_site_dict.keys():
                if i_bond == 0:
                    fixed_C_column_block = two_site_dict[A, C].reshape(d[A, B], d[B, C] * chi[i_bond + 1][C])
                elif i_bond == N - 2:
                    fixed_C_column_block = two_site_dict[A, C].reshape(chi[i_bond-1][A] * d[A, B], d[B, C])
                else:
                    fixed_C_column_block = two_site_dict[A, C].reshape(chi[i_bond-1][A] * d[A, B], d[B, C] * chi[i_bond + 1][C])
                if row_block is None:
                    row_block = fixed_C_column_block
                else:
                    row_block = np.concatenate((row_block, fixed_C_column_block), axis=1) # concatenate horizontally
        if result is None:
            result = row_block
        elif row_block is not None:
            result = np.concatenate((result, row_block), axis=0)
    return result.flatten()

# Two-site matrix product state
def two_site_mps_fixed_B_vec_to_dict(vec, B, i_bond, chi, d, modules_sorted, allowed_module_pairs, full_env_dims_C, full_env_dims_A, N):
    result = {}
    row_start = 0
    mat = vec.reshape(full_env_dims_A[i_bond][B], full_env_dims_C[i_bond][B])
    for A in modules_sorted:
        column_start = 0
        for C in modules_sorted:
            if (A, B) in allowed_module_pairs and (B, C) in allowed_module_pairs:
                if i_bond == 0:
                    result[A, C] = mat[row_start : row_start + d[A, B], column_start : column_start + d[B, C] * chi[i_bond + 1][C]]
                    result[A, C] = result[A, C].reshape(d[A, B], d[B, C], chi[i_bond + 1][C])
                elif i_bond == N - 2:
                    result[A, C] = mat[row_start : row_start + chi[i_bond - 1][A] * d[A, B], column_start : column_start + d[B, C]]
                    result[A, C] = result[A, C].reshape(chi[i_bond - 1][A], d[A, B], d[B, C])
                else:
                    result[A, C] = mat[row_start : row_start + chi[i_bond - 1][A] * d[A, B], column_start : column_start + d[B, C] * chi[i_bond + 1][C]]
                    result[A, C] = result[A, C].reshape(chi[i_bond - 1][A], d[A, B], d[B, C], chi[i_bond + 1][C])
                if i_bond == N - 2:
                    column_start += d[B, C]
                else:
                    column_start += d[B, C] * chi[i_bond + 1][C]
        if (A, B) in allowed_module_pairs:
            if i_bond == 0:
                row_start += d[A, B]
            else:
                row_start += chi[i_bond - 1][A] * d[A, B]
    return result

def two_site_mps_all_B_dict_to_vec(two_site_dict, i_bond, modules_sorted, chi, d, N):
    vec_all_B = []
    for B in modules_sorted:
        if B in two_site_dict.keys():
            vec_fixed_B = two_site_mps_fixed_B_dict_to_vec(two_site_dict[B], B, i_bond, chi, d, modules_sorted, N)
            vec_all_B.append(vec_fixed_B)
    result = np.concatenate(vec_all_B)
    return result

def two_site_mps_all_B_vec_to_dict(two_site_vec, i_bond, chi, d, modules_sorted, allowed_module_pairs, full_env_dims_A, full_env_dims_C, N):
    start_index = 0
    result = {}
    for B in modules_sorted:
        result[B] = {}
        total_env_size_fixed_B = full_env_dims_A[i_bond][B] * full_env_dims_C[i_bond][B]
        vec_fixed_B = two_site_vec[start_index : start_index + total_env_size_fixed_B]
        start_index += total_env_size_fixed_B
        result[B] = two_site_mps_fixed_B_vec_to_dict(vec_fixed_B, B, i_bond, chi, d, modules_sorted, allowed_module_pairs, full_env_dims_A, full_env_dims_C, N)
    return result

def mps_site_fixed_B_left_grouped_dict_to_matrix(mps_site, B, modules_sorted, allowed_module_pairs, is_left_boundary=False, is_right_boundary=False):
    
    if is_left_boundary:
        M_list = [
            mps_site[A, B]
            for A in modules_sorted if (A, B) in allowed_module_pairs
        ]
    elif is_right_boundary:
        M_list = [
            mps_site[A, B].reshape(mps_site[A, B].shape[0] * mps_site[A, B].shape[1], 1)
            for A in modules_sorted if (A, B) in allowed_module_pairs
        ]
    else:
        M_list = [
            mps_site[A, B].reshape(mps_site[A, B].shape[0] * mps_site[A, B].shape[1], mps_site[A, B].shape[2])
            for A in modules_sorted if (A, B) in allowed_module_pairs
        ]
    M = np.concatenate(M_list, axis=0) # Concatenate vertically
    return M

def mps_site_fixed_A_right_grouped_dict_to_matrix(mps_site, A, modules_sorted, allowed_module_pairs, is_left_boundary=False, is_right_boundary=False):
    if is_right_boundary:
        M_list = [
            mps_site[A, B] 
            for B in modules_sorted if (A, B) in allowed_module_pairs
            ]
    elif is_left_boundary:
        M_list = [
            mps_site[A, B].reshape(1, mps_site[A, B].shape[0] * mps_site[A, B].shape[1])
            for B in modules_sorted if (A, B) in allowed_module_pairs
        ]
    else:
        M_list = [
            mps_site[A, B].reshape(mps_site[A, B].shape[0], mps_site[A, B].shape[1] * mps_site[A, B].shape[2]) 
            for B in modules_sorted if (A, B) in allowed_module_pairs
            ]
    M = np.concatenate(M_list, axis=1) # Concatenate horizontally
    return M

# Contracts 2 MPS sites into a two-site dictionary. Assumes at most 1 MPS site of the pair is a boundary (rank-2).
def contract_mps_pair(modules, mps_site_1, mps_site_2, is_left_boundary, is_right_boundary):

    result = {B: {} for B in modules}
    combinations = product(modules, repeat=3)
    for A, B, C in combinations:
        if (A, B) in mps_site_1.keys() and (B, C) in mps_site_2.keys():
            if is_left_boundary:
                result[B][A, C] = oe.contract('ia,ajb->ijb', mps_site_1[A, B], mps_site_2[B, C])
            elif is_right_boundary:
                result[B][A, C] = oe.contract('aib,bj->aij', mps_site_1[A, B], mps_site_2[B, C])
            else:
                result[B][A, C] = oe.contract('aib,bjc->aijc', mps_site_1[A, B], mps_site_2[B, C])
    return result

# Transfer matrix
def T_dict_to_matrix(total_shape, T_dict, modules_sorted, chi):
    M = np.zeros((total_shape, total_shape), dtype=np.complex128)
    i = 0
    for A in modules_sorted:
        j = 0
        for B in modules_sorted:
            if (A, B) in T_dict.keys():
                M[i : i + chi[A]**2, j : j + chi[B]**2] = T_dict[A, B].reshape(chi[A]**2, chi[B]**2)
            j += chi[B]**2
        i += chi[A]**2
    return M

# Any matrix applied to the MPS bonds 
def X_vec_to_dict(X_vec, chi, modules_sorted):
    X_dict = {}
    i = 0
    for A in modules_sorted:
        size = chi[A] ** 2
        X_dict[A] = X_vec[i : i + size].reshape(chi[A], chi[A])
        i += size
    return X_dict

# Any matrix applied to the MPS bonds 
def X_dict_to_vec(X_dict, modules_sorted):
    vecs = []
    for A in modules_sorted:
        vecs.append(X_dict[A].flatten())  # Flatten to 1D
    return np.concatenate(vecs)

