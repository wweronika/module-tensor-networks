import numpy as np
from data_conversion import *
from state_initialisation import *
from gauge_fixing import *


def T_eigenvector_test(A_dict, T_dict):
    pass

def mps_vec_to_dict_test():
    d = {(0, 0): 2, (0, 1): 2, (1, 0): 2, (1, 1): 2}
    bond_dim = 15
    chi = {0: bond_dim, 1: bond_dim}
    allowed_module_pairs = {(0, 0), (0, 1), (1, 0), (1, 1)}
    allowed_module_pairs_sorted = [(0, 0), (0, 1), (1, 0), (1, 1)]

    mps_dict = get_random_mps_site(d, chi, allowed_module_pairs)

    mps_vec = mps_dict_to_vec(mps_dict, allowed_module_pairs_sorted)
    mps_dict_new = mps_vec_to_dict(mps_vec, allowed_module_pairs_sorted, chi, d)

    for pair in allowed_module_pairs:
        if not np.all(np.isclose(mps_dict[pair], mps_dict_new[pair])):
            return False
    return True

def put_mps_into_right_canonical_form_test():

    d = {(0, 0): 2, (0, 1): 2, (1, 0): 2, (1, 1): 2}
    N = 10
    bond_dim = 15
    chi = [{0: bond_dim, 1: bond_dim} for i in range(N-1)]
    allowed_module_pairs = {(0, 0), (0, 1), (1, 0), (1, 1)}
    modules_sorted = [0, 1]
    modules = {0, 1}
    mps = get_random_mps(N, d, bond_dim, allowed_module_pairs)

    put_mps_into_right_canonical_form(mps, modules_sorted, allowed_module_pairs, d, chi, N)

    print(is_right_canonical_form_overall(modules, chi, mps))
    assert is_right_canonical_form_overall(modules, chi, mps)


def put_mps_into_left_canonical_form_test():

    d = {(0, 0): 2, (0, 1): 2, (1, 0): 2, (1, 1): 2}
    N = 10
    bond_dim = 15
    chi = [{0: bond_dim, 1: bond_dim} for i in range(N-1)]
    allowed_module_pairs = {(0, 0), (0, 1), (1, 0), (1, 1)}
    modules_sorted = [0, 1]
    modules = {0, 1}
    mps = get_random_mps(N, d, bond_dim, allowed_module_pairs)

    put_mps_into_left_canonical_form(mps, modules_sorted, allowed_module_pairs, d, chi, N)
    
    print(is_left_canonical_form_overall(modules, chi, mps))
    assert is_left_canonical_form_overall(modules, chi, mps)

# print(mps_vec_to_dict_test())
put_mps_into_right_canonical_form_test()
put_mps_into_left_canonical_form_test()