import numpy as np
import opt_einsum as oe
from scipy.sparse.linalg import eigs

from util.data_conversion import *
from util.state_initialisation import *

def is_left_canonical_form(modules, chi, mps):
    N = len(mps)
    results = []
    results_all_modules = []
    for i in range(N):
        result_single_site = {}
        result_all_modules_single_site = True
        for B in modules:
            tensor = None
            for A in modules:
                if (A, B) in mps[i].keys():
                    if i == 0:
                        if tensor is None:
                            tensor = oe.contract("ib,id->bd", mps[i][A, B], mps[i][A, B].conj())
                        else:
                            tensor += oe.contract("ib,id->bd", mps[i][A, B], mps[i][A, B].conj())
                    elif i == N-1:
                        if tensor is None:
                            tensor = oe.contract("ai,ai->", mps[i][A, B], mps[i][A, B].conj())
                        else:
                            tensor += oe.contract("ai,ai->", mps[i][A, B], mps[i][A, B].conj())
                    else:
                        if tensor is None:
                            tensor = oe.contract("aib,aid->bd", mps[i][A, B], mps[i][A, B].conj())
                        else:
                            tensor += oe.contract("aib,aid->bd", mps[i][A, B], mps[i][A, B].conj())
            if i == N - 1:
                result_single_site[A] = np.isclose(1, tensor) # Expect tensor = trace of last mps site 
            else:
                result_single_site[A] = np.isclose(np.eye(tensor.shape[0]), tensor)
            result_all_modules_single_site = result_all_modules_single_site and result_single_site[A].all()
        results.append(result_single_site)
        results_all_modules.append(result_all_modules_single_site)
    return results, results_all_modules

def is_right_canonical_form(modules, chi, mps):
    N = len(mps)
    results = []
    results_all_modules = []
    for i in range(N):
        result_single_site = {}
        result_all_modules_single_site = True
        for A in modules:
            tensor = None
            for B in modules:
                if (A, B) in mps[i].keys():
                    if i == 0:
                        if tensor is None:
                            tensor = oe.contract("ib,ib->", mps[i][A, B], mps[i][A, B].conj())
                        else:
                            tensor += oe.contract("ib,ib->", mps[i][A, B], mps[i][A, B].conj())
                        pass
                    elif i == N-1:
                        if tensor is None:
                            tensor = oe.contract("ai,ci->ac", mps[i][A, B], mps[i][A, B].conj())
                        else:
                            tensor += oe.contract("ai,ci->ac", mps[i][A, B], mps[i][A, B].conj())
                    else:
                        if tensor is None:
                            tensor = oe.contract("aib,cib->ac", mps[i][A, B], mps[i][A, B].conj())
                        else:
                            tensor += oe.contract("aib,cib->ac", mps[i][A, B], mps[i][A, B].conj())
            if i == 0:
                result_single_site[A] = np.isclose(1, tensor) # Expect tensor = trace of last mps site 
            else:
                result_single_site[A] = np.isclose(np.eye(tensor.shape[0]), tensor)
            result_all_modules_single_site = result_all_modules_single_site and result_single_site[A].all()
        results.append(result_single_site)
        results_all_modules.append(result_all_modules_single_site)
    return results, results_all_modules

def is_right_canonical_form_overall(modules, chi, mps):
    _, results_all_modules = is_right_canonical_form(modules, chi, mps)
    return np.all(results_all_modules)

def is_left_canonical_form_overall(modules, chi, mps):
    _, results_all_modules = is_left_canonical_form(modules, chi, mps)
    return np.all(results_all_modules)

def is_left_canonical_form_single_site(modules, chi, mps_site):
    results_per_module = []
    for B in modules:
        t = np.zeros((chi[B], chi[B]), dtype=np.complex128)
        for A in modules:
            if (A, B) in mps_site.keys():
                t += oe.contract("aib,aid->bd", mps_site[A, B], mps_site[A, B].conj())
        results_per_module.append(np.all(np.isclose(np.eye(chi[B]), t, atol=1e-6)))
        print(f"Left gauge error for module {B}: {np.linalg.norm(np.eye(chi[B]) - t)}")
    return results_per_module

def is_right_canonical_form_single_site(modules, chi, mps_site):
    results_per_module = []
    for A in modules:
        t = np.zeros((chi[A], chi[A]), dtype=np.complex128)
        for B in modules:
            if (A, B) in mps_site.keys():
                t += oe.contract("aib,cib->ac", mps_site[A, B], mps_site[A, B].conj())
        results_per_module.append(np.all(np.isclose(np.eye(chi[A]), t, atol=1e-6)))
        print(f"Right gauge error for module {A}: {np.linalg.norm(np.eye(chi[A]) - t)}")
    return results_per_module

def put_mps_into_right_canonical_form(mps, modules_sorted, allowed_module_pairs, d, chi, N):

        for i in range(N-1, -1, -1):

            is_left_boundary = (i == 0)
            is_right_boundary = (i == N-1)

            for A in modules_sorted:
                if not is_left_boundary:
                    M = mps_site_fixed_A_right_grouped_dict_to_matrix(mps[i], A, modules_sorted, allowed_module_pairs, is_left_boundary=is_left_boundary, is_right_boundary=is_right_boundary)
                    U, S, V_H = np.linalg.svd(M, full_matrices=False)
                    US = U @ np.diag(S)
                    new_chi_L, _ = V_H.shape

                    set_mps_site_to_matrix_fixed_A(mps[i], A, V_H, modules_sorted, allowed_module_pairs, d, is_left_boundary=is_left_boundary, is_right_boundary=is_right_boundary)

                    # Contracting the remaining part of old mps into the previous site
                    mps_times_matrix(mps[i-1], A, US, modules_sorted, allowed_module_pairs, is_left_boundary=(i == 1)) 
                    chi[i-1][A] = new_chi_L
                else:
                    normalise_mps_site_for_fixed_A(mps[i], A, is_boundary=True, modules=modules_sorted)


def put_mps_into_left_canonical_form(mps, modules_sorted, allowed_module_pairs, d, chi, N):

    for i in range(0, len(mps)):

        is_left_boundary = (i == 0)
        is_right_boundary = (i == N-1)

        for B in modules_sorted:
       
            M = mps_site_fixed_B_left_grouped_dict_to_matrix(mps[i], B, modules_sorted, allowed_module_pairs, is_left_boundary=is_left_boundary, is_right_boundary=is_right_boundary)
       
            U, S, V_H = np.linalg.svd(M, full_matrices=False)
            S = np.diag(S)
            SV = S @ V_H
            _, new_chi_R = U.shape
       
            set_mps_site_to_matrix_fixed_B(mps[i], B, U, modules_sorted, allowed_module_pairs, d, is_left_boundary=is_left_boundary, is_right_boundary=is_right_boundary)
            
            # Contracting the remaining part of old mps into the next site
            if not is_right_boundary:    
                matrix_times_mps(mps[i+1], B, SV, modules_sorted, allowed_module_pairs, is_right_boundary=(i == N-2))
                chi[i][B] = new_chi_R
            else:
                normalise_mps_site_for_fixed_B(mps[i], B, is_boundary=True, modules=modules_sorted)

def put_mps_site_into_left_canonical_form(A_dict, allowed_module_pairs, modules_sorted, chi):
    T = {}
    for A, B in allowed_module_pairs:
        T[A, B] = oe.contract('aib,cid->acbd', A_dict[A, B], A_dict[A, B].conj())

    total_T_eigenvec_shape = get_T_right_eigenvec_shape(A_dict, chi)

    T_matrix = T_dict_to_matrix(total_T_eigenvec_shape, T, modules_sorted, chi)
    eigval, X_A_square = eigs(T_matrix.T.conj(), k=1, which='LR')
    print(f"left eigval: {eigval}")

    X_A_square_dict = X_vec_to_dict(X_A_square, chi, modules_sorted)
    X_A_norm = 0
    for M in modules_sorted:
        X_A_norm += np.linalg.norm(X_A_square_dict[M])

    for M, X_A_square in X_A_square_dict.items():
        X_A_square /= X_A_norm
        X_A_square_dict[M] = 1/2 * (X_A_square + X_A_square.conj().T)

    X_A = {}
    X_A_inverse = {}

    for M in modules_sorted:
        U, S, Vh = np.linalg.svd(X_A_square_dict[M])
        X_A[M] = U @ np.diag(np.sqrt(S)) @ U.T.conj()
        X_A_inverse[M] = U @ np.diag(1/np.sqrt(S)) @ U.T.conj()

    A_L = {}
    for A, C in allowed_module_pairs:
        A_L[A, C] = oe.contract('ab,bic,cd->aid', X_A[A], A_dict[A, C], X_A_inverse[C]) / np.sqrt(eigval)
    return A_L, X_A

def put_mps_site_into_right_canonical_form(A_dict, allowed_module_pairs, modules_sorted, chi):
    T = {}
    for A, C in allowed_module_pairs:
        T[A, C] = oe.contract('aib,cid->acbd', A_dict[A, C], A_dict[A, C].conj())
    
    total_T_eigenvec_shape = get_T_right_eigenvec_shape(A_dict, chi)

    T_matrix = T_dict_to_matrix(total_T_eigenvec_shape, T, modules_sorted, chi)
    eigval, X_A_square = eigs(T_matrix, k=1, which='LR')
    print(f"right eigval: {eigval}")

    X_A_square_dict = X_vec_to_dict(X_A_square, chi, modules_sorted)

    X_A_norm = 0
    for M in modules_sorted:
        X_A_norm += np.linalg.norm(X_A_square_dict[M])

    for M, X_A_square in X_A_square_dict.items():
        X_A_square /= X_A_norm
        X_A_square_dict[M] = 1/2 * (X_A_square + X_A_square.conj().T)

    X_A = {}
    X_A_inverse = {}

    for M in modules_sorted:
        U, S, Vh = np.linalg.svd(X_A_square_dict[M])
        X_A[M] = U @ np.diag(np.sqrt(S)) @ U.T.conj()
        X_A_inverse[M] = U @ np.diag(1/np.sqrt(S)) @ U.T.conj()

    A_R = {}
    for A, C in allowed_module_pairs:
        A_R[A, C] = oe.contract('ab,bic,cd->aid', X_A_inverse[A], A_dict[A, C], X_A[C]) / np.sqrt(eigval)

    return A_R, X_A




