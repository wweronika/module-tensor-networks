import numpy as np
import opt_einsum as oe
import pickle

from scipy.sparse.linalg import eigs, LinearOperator
from scipy.linalg import sqrtm, polar
from math import exp, sqrt, log
from copy import copy

from util.mpo_constructions import get_Ising_T
from util.state_initialisation import *
from util.gauge_fixing import *
from util.data_conversion import *

is_Z2 = True
chi_bond = 15

if is_Z2:

    allowed_vertical_module_pairs = {(0, 0), (0, 1), (1, 0), (1, 1)}
    allowed_vertical_module_pairs_sorted = [(0, 0), (0, 1), (1, 0), (1, 1)]
    allowed_module_pairs = {(0, 0), (0, 1), (1, 0), (1, 1)}
    allowed_module_pairs_sorted = [(0, 0), (0, 1), (1, 0), (1, 1)]
    modules = {0, 1}
    modules_sorted = [0, 1]

    chi_A = {0: chi_bond, 1: chi_bond}
    chi_B = {0: chi_bond, 1: chi_bond}
    d_A = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    d_B = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}

    total_A_shape = 1 * chi_bond**2 * len(allowed_module_pairs)
    total_B_shape = 1 * chi_bond**2 * len(allowed_module_pairs)
    total_T_eigenvec_shape = chi_bond**2 * len(modules)

else:

    allowed_vertical_module_pairs = {(0, 0)}
    allowed_vertical_module_pairs_sorted = [(0, 0)]
    allowed_module_pairs = {(0, 0)}
    allowed_module_pairs_sorted = [(0, 0)]
    modules = {0}
    modules_sorted = [0]

    chi_A = {0: chi_bond}
    chi_B = {0: chi_bond}
    d_A = {(0, 0): 2}
    d_B = {(0, 0): 2}

    total_A_shape = 2 * chi_bond**2 * len(allowed_module_pairs)
    total_B_shape = 2 * chi_bond**2 * len(allowed_module_pairs)
    total_T_eigenvec_shape = chi_bond**2 * len(modules)

A = get_random_mps_site(d_A, chi_A, allowed_module_pairs)
normalise_mps_site(A, is_boundary=False)

B = get_random_mps_site(d_B, chi_B, allowed_module_pairs)
normalise_mps_site(B, is_boundary=False)


def get_Ising_MPO(beta):
    T = np.zeros((2, 2, 2, 2))
    T[0, 0, 0, 0] = 1
    T[1, 1, 1, 1] = 1
    W = np.array([[exp(beta), exp(-beta)], [exp(-beta), exp(beta)]])
    W_sqrt = sqrtm(W)
    T_WW = oe.contract('ijkl,ai,bj,ck,dl->abcd', T, W_sqrt, W_sqrt, W_sqrt, W_sqrt)
    T_WW_dict = {(0, 0, 0, 0): T_WW}
    return T_WW_dict

def apply_transfer_matrix_to_A(v, T_B):
    A_dict = mps_vec_to_dict(v, allowed_vertical_module_pairs_sorted, chi_A, d_A)
    result = {}
    for B, D in allowed_vertical_module_pairs:
        result[B, D] = np.zeros((chi_A[B], d_A[B, D], chi_B[D]), dtype=np.complex128)
        for A, C in allowed_vertical_module_pairs:
            result[B, D] += oe.contract('ace,acebdf->bdf', A_dict[A, C], T_B[A, B, C, D])
    return mps_dict_to_vec(result, allowed_vertical_module_pairs_sorted)

def apply_transfer_matrix_to_B(v, T_A):
    B_dict = mps_vec_to_dict(v, allowed_module_pairs_sorted, chi_B, d_B)
    result = {}
    for C, D in allowed_module_pairs:
        result[C, D] = np.zeros((chi_B[C], d_B[C, D], chi_B[D]), dtype=np.complex128)
        for A, B in allowed_module_pairs:
            result[C, D] += oe.contract('cid,ajbcid->ajb', B_dict[A, B], T_A[C, A, D, B])
    return mps_dict_to_vec(result, allowed_module_pairs_sorted)

def get_T_A(A_dict, T):
    result = {}
    for A, B, C, D in T.keys():
        result[C, A, D, B] = oe.contract('akc,kijl,bld->ajbcid', A_dict[C, A], T[A, B, C, D], A_dict[D, B].conj())
    return result

def get_T_B(B_dict, T):
    result = {}
    for A, B, C, D in T.keys():
        result[A, B, C, D] = oe.contract('aib,cijd,ejf->acebdf', B_dict[A, B], T[A, B, C, D], B_dict[C, D].conj())
    return result

def pulling_through_iteration_A(A_R, T):

    T_A = get_T_A(A_R, T)

    O_2 = LinearOperator(
        (total_B_shape, total_B_shape),
        matvec=lambda vec: apply_transfer_matrix_to_B(vec, T_A)
    )

    lambda_B, B_vec = eigs(O_2, k=1, which='LR')
    B = mps_vec_to_dict(B_vec, allowed_module_pairs_sorted, chi_B, d_B)
    print(f"lambda_B: {lambda_B}")
    
    B_L, X_B = put_mps_site_into_left_canonical_form(B, allowed_module_pairs, chi_B)
    # print(is_left_canonical_form_single_site(modules_sorted, chi_B, B_L))

    T_B = get_T_B(B_L, T)

    O_1 = LinearOperator(
        (total_A_shape, total_A_shape),
        matvec=lambda vec: apply_transfer_matrix_to_A(vec, T_B)
    )
    
    lambda_A, A_vec = eigs(O_1, k=1, which='LR')
    A = mps_vec_to_dict(A_vec, allowed_vertical_module_pairs_sorted, chi_A, d_A)
    print(f"lambda_A: {lambda_A}")
    A_R, X_A = put_mps_site_into_right_canonical_form(A, allowed_vertical_module_pairs, chi_A)
    # print(is_right_canonical_form_single_site(modules_sorted, chi_A, A_R))
    
    error = np.linalg.norm(X_A[0] - X_B[0])
    print(f"error: {error}")

    return A_R

def pulling_through_iteration(B_L, T):

    global allowed_module_pairs, allowed_module_pairs_sorted, allowed_vertical_module_pairs, allowed_vertical_module_pairs_sorted
    global modules, modules_sorted

    T_B = get_T_B(B_L, T)

    O_1 = LinearOperator(
        (total_A_shape, total_A_shape),
        matvec=lambda vec: apply_transfer_matrix_to_A(vec, T_B)
    )
    
    lambda_A, A_vec = eigs(O_1, k=1, which='LR')
    A = mps_vec_to_dict(A_vec, allowed_vertical_module_pairs_sorted, chi_A, d_A)
    print(f"lambda_A: {lambda_A}")
    A_R, X_A = put_mps_site_into_right_canonical_form(A, allowed_vertical_module_pairs, modules_sorted, chi_A)
    print(is_right_canonical_form_single_site(modules_sorted, chi_A, A_R))
    T_A = get_T_A(A_R, T)

    O_2 = LinearOperator(
        (total_B_shape, total_B_shape),
        matvec=lambda vec: apply_transfer_matrix_to_B(vec, T_A)
    )

    lambda_B, B_vec = eigs(O_2, k=1, which='LR')
    B = mps_vec_to_dict(B_vec, allowed_module_pairs_sorted, chi_B, d_B)
    print(f"lambda_B: {lambda_B}")
    
    B_L, X_B = put_mps_site_into_left_canonical_form(B, allowed_module_pairs, modules_sorted, chi_B)
    print(is_left_canonical_form_single_site(modules_sorted, chi_B, B_L))

    error = np.linalg.norm(X_A[0] - X_B[0])
    print(f"error: {error}")

    return B_L

if is_Z2:
    T = get_Ising_T(module_category_name="Z2")
else: 
    T = get_Ising_T(module_category_name="Vec")


N = 10

B_L, X_B = put_mps_site_into_left_canonical_form(B, allowed_vertical_module_pairs, modules_sorted, chi_B)
for i in range(N):
    B_L = pulling_through_iteration(B_L, T)

# Alternatively: start from A_R
# A_R, X_A = put_mps_site_into_right_canonical_form(A, allowed_module_pairs, modules_sorted, chi_A)
# for i in range(N):
#     A_R = pulling_through_iteration_A(A_R, T)
