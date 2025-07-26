from itertools import product
from scipy.sparse.linalg import LinearOperator, eigs

from datetime import datetime, timedelta

from util.data_conversion import *
from util.state_initialisation import *
from util.gauge_fixing import *
from util.input_parsing import *

def initialise_env_dims(N):
    env_dims_left = [{} for i in range(N - 1)]
    env_dims_right = [{} for i in range(N - 1)]
    for i in range(N-1):
        for M in modules:
            update_env_dims_left(env_dims_left, i, M)
            update_env_dims_right(env_dims_right, i, M)
    return env_dims_left, env_dims_right

def update_env_dims_left(env_dims_left, i_bond, B):
    env_dim_left = 0
    for A in modules:
        if (A, B) in mps[i_bond].keys():
            if i_bond == 0:
                env_dim_left += d[A, B]
            else:
                env_dim_left += chi[i_bond - 1][A] * d[A, B]
    env_dims_left[i_bond][B] = env_dim_left

def update_env_dims_right(env_dims_right, i_bond, A):
    env_dim_right = 0
    for B in modules:
        if (A, B) in mps[i_bond].keys():
            if i_bond == N - 2:
                env_dim_right += d[A, B]
            else:
                env_dim_right += d[A, B] * chi[i_bond + 1][B]
    env_dims_right[i_bond][A] = env_dim_right

def get_boundary_vectors(chi_left, chi_right, allowed_vertical_module_pairs, boundary_modules_left, boundary_modules_right):
    vs_right = {}
    vs_left = {}

    for A, C in allowed_vertical_module_pairs:
        v_left = np.zeros(chi_left[A, C])
        vs_left[A, C] = v_left

    for A, C in boundary_modules_left:
        v_left[0] = 1
        vs_left[A, C] = v_left

    for B, D in allowed_vertical_module_pairs:
        v_right = np.zeros(chi_right[B, D])
        vs_right[B, D] = v_right
        
    
    for B, D in boundary_modules_right:
        v_right[-1] = 1
        vs_right[B, D] = v_right

    return vs_left, vs_right

def get_left_environments(mps, v_left, H, allowed_vertical_module_pairs, N):
    left_envs = [None for i in range(N)]
    for i in range(N):
        # Rightmost has no right environment
        left_env = {key: None for key in allowed_vertical_module_pairs}
        if i == 0:
            continue 
        # Second to rightmost requires contraction with edge vector
        elif i == 1:
            for B, D in allowed_vertical_module_pairs:
                for A, C in allowed_vertical_module_pairs:
                    if (A, B, C, D) in H[i-1].keys():
                        if left_env[B, D] is None:
                            left_env[B, D] = oe.contract('c,ib,cijd,jf->bdf', v_left[A, C], mps[i-1][A, B], H[i-1][A, B, C, D], mps[i-1][C, D].conj())
                        else:
                            left_env[B, D] += oe.contract('c,ib,cijd,jf->bdf', v_left[A, C], mps[i-1][A, B], H[i-1][A, B, C, D], mps[i-1][C, D].conj()) 
        # Regular case (interior of MPS)
        else:
            for B, D in allowed_vertical_module_pairs:
                for A, C in allowed_vertical_module_pairs:
                    if (A, B, C, D) in H[i-1].keys():
                        if left_env[B, D] is None:
                            left_env[B, D] = oe.contract('ace,aib,cijd,ejf->bdf', left_envs[i-1][A, C], mps[i-1][A, B], H[i-1][A, B, C, D], mps[i-1][C, D].conj())
                        else:
                            left_env[B, D] += oe.contract('ace,aib,cijd,ejf->bdf', left_envs[i-1][A, C], mps[i-1][A, B], H[i-1][A, B, C, D], mps[i-1][C, D].conj()) 
        left_envs[i] = left_env
    return left_envs

def get_right_environments(mps, v_right, H, allowed_vertical_module_pairs, N):

    right_envs = [None for i in range(N)]
    for i in range(N-1, -1, -1):
        # Rightmost has no right environment
        right_env = {key: None for key in allowed_vertical_module_pairs}
        if i == N-1:
            continue 
        # Second to rightmost requires contraction with edge vector
        elif i == N-2:
            for A, C in allowed_vertical_module_pairs:
                for B, D in allowed_vertical_module_pairs:
                    if (A, B, C, D) in H[i+1].keys():
                        if right_env[A, C] is None:
                            right_env[A, C] = oe.contract('ai,bijc,dj,c->abd', mps[i+1][A, B], H[i+1][A, B, C, D], mps[i+1][C, D].conj(), v_right[B, D])
                        else: 
                            right_env[A, C] += oe.contract('ai,bijc,dj,c->abd', mps[i+1][A, B], H[i+1][A, B, C, D], mps[i+1][C, D].conj(), v_right[B, D])
            right_env[A, C] /= np.linalg.norm(right_env[A, C])
        # Regular case (interior of MPS)
        else:
            for A, C in allowed_vertical_module_pairs:
                for B, D in allowed_vertical_module_pairs:
                    if (A, B, C, D) in H[i+1].keys():
                        if right_env[A, C] is None:
                            right_env[A, C] = oe.contract('aib,cijd,ejf,bdf->ace', mps[i+1][A, B], H[i+1][A, B, C, D], mps[i+1][C, D].conj(), right_envs[i+1][B, D])
                        else:
                            right_env[A, C] += oe.contract('aib,cijd,ejf,bdf->ace', mps[i+1][A, B], H[i+1][A, B, C, D], mps[i+1][C, D].conj(), right_envs[i+1][B, D])
            right_env[A, C] /= np.linalg.norm(right_env[A, C])
        right_envs[i] = right_env  
    return right_envs 

# Assumes all left_envs <= i-1 and mps <= i-1 are updated
def update_left_environment(i):
    global allowed_vertical_module_pairs, v_left, mps, H, left_envs
    new_left_env = {key: None for key in allowed_vertical_module_pairs}
    if i == 0:
        pass
    elif i == 1:
        for B, D in allowed_vertical_module_pairs:
            for A, C in allowed_vertical_module_pairs:
                if (A, B, C, D) in H[i-1].keys():
                    if new_left_env[B, D] is None:
                        new_left_env[B, D] = oe.contract('c,ib,cijd,jf->bdf', v_left[A, C], mps[i-1][A, B], H[i-1][A, B, C, D], mps[i-1][C, D].conj())
                    else:
                        new_left_env[B, D] += oe.contract('c,ib,cijd,jf->bdf', v_left[A, C], mps[i-1][A, B], H[i-1][A, B, C, D], mps[i-1][C, D].conj()) 
    else:
        for B, D in allowed_vertical_module_pairs:
            for A, C in allowed_vertical_module_pairs:
                if (A, B, C, D) in H[i-1].keys():
                    if new_left_env[B, D] is None:
                        new_left_env[B, D] = oe.contract('ace,aib,cijd,ejf->bdf', left_envs[i-1][A, C], mps[i-1][A, B], H[i-1][A, B, C, D], mps[i-1][C, D].conj())
                    else:
                        new_left_env[B, D] += oe.contract('ace,aib,cijd,ejf->bdf', left_envs[i-1][A, C], mps[i-1][A, B], H[i-1][A, B, C, D], mps[i-1][C, D].conj()) 
    left_envs[i] = new_left_env

# Assumes all right_envs >= i+1 and mps >= i+1 are updated
def update_right_environment(i):

    global allowed_vertical_module_pairs, v_right, mps, H, right_envs

    new_right_env = {key: None for key in allowed_vertical_module_pairs}
    if i == N-1:
        pass
    elif i == N-2:
        for A, C in allowed_vertical_module_pairs:
            for B, D in allowed_vertical_module_pairs:
                if (A, B, C, D) in H[i+1].keys():
                    if new_right_env[A, C] is None:
                        new_right_env[A, C] = oe.contract('ai,bijc,dj,c->abd', mps[i+1][A, B], H[i+1][A, B, C, D], mps[i+1][C, D].conj(), v_right[B, D])
                    else: 
                        new_right_env[A, C] += oe.contract('ai,bijc,dj,c->abd', mps[i+1][A, B], H[i+1][A, B, C, D], mps[i+1][C, D].conj(), v_right[B, D])
    else:
        for A, C in allowed_vertical_module_pairs:
            for B, D in allowed_vertical_module_pairs:
                if (A, B, C, D) in H[i+1].keys():
                    if new_right_env[A, C] is None:
                        new_right_env[A, C] = oe.contract('aib,cijd,ejf,bdf->ace', mps[i+1][A, B], H[i+1][A, B, C, D], mps[i+1][C, D].conj(), right_envs[i+1][B, D])
                    else: 
                        new_right_env[A, C] += oe.contract('aib,cijd,ejf,bdf->ace', mps[i+1][A, B], H[i+1][A, B, C, D], mps[i+1][C, D].conj(), right_envs[i+1][B, D])
    right_envs[i] = new_right_env



def apply_effective_H(A_vec, i_bond, left_env, right_env, H_local_1, H_local_2, path):

    global modules_sorted, allowed_module_pairs, env_dims_left, env_dims_right, chi, d, N

    A_dict_all_B = two_site_mps_all_B_vec_to_dict(A_vec, i_bond, chi, d, modules_sorted, allowed_module_pairs, env_dims_right, env_dims_left, N)

    is_left_boundary = (i_bond == 0)
    is_right_boundary = (i_bond == N-2)

    combinations = product(modules_sorted, repeat=6)
    result_dict = {E: {} for E in modules_sorted}
    for A, B, C, D, E, F in combinations:
        if (A, B, D, E) in H_local_1.keys() and (B, C, E, F) in H_local_2.keys():
            if is_left_boundary:
                tensor = - oe.contract('ijb,cikg,gjld,c,bdf->klf', A_dict_all_B[B][A, C], H_local_1[A, B, D, E], H_local_2[B, C, E, F], v_left[A, D], right_env[C, F], optimize=path)
            elif is_right_boundary:
                tensor = - oe.contract('aij,cikg,gjld,ace,d->ekl', A_dict_all_B[B][A, C], H_local_1[A, B, D, E], H_local_2[B, C, E, F], left_env[A,D], v_right[C,F], optimize=path)
            else:
                tensor = - oe.contract('aijb,cikg,gjld,ace,bdf->eklf', A_dict_all_B[B][A, C], H_local_1[A, B, D, E], H_local_2[B, C, E, F], left_env[A, D], right_env[C, F], optimize=path)
            if (D, F) in result_dict[E].keys():
                result_dict[E][D, F] += tensor
            else:
                result_dict[E][D, F] = tensor
    vec = two_site_mps_all_B_dict_to_vec(result_dict, i_bond, modules_sorted, chi, d, N)
    return vec

def optimise_site_pair(i, is_moving_right):

    global modules, modules_sorted, chi, d, env_dims_left, env_dims_right, left_envs, right_envs, H, mps, tolerance_in_S, N

    is_left_boundary = (i == 0)
    is_right_boundary = (i == N-2)

    left_env = left_envs[i]
    right_env = right_envs[i+1]
    H_1 = H[i]
    H_2 = H[i+1]

    for E in modules_sorted:
        update_env_dims_left(env_dims_left, i, E)
        update_env_dims_right(env_dims_right, i, E)

    two_site_dict = contract_mps_pair(modules, mps[i], mps[i+1], is_left_boundary, is_right_boundary)
    two_site_vec = two_site_mps_all_B_dict_to_vec(two_site_dict, i, modules_sorted, chi, d, N)
    full_env_dim = two_site_vec.size

    full_env = LinearOperator(
            (full_env_dim, full_env_dim),
                matvec=lambda vec: apply_effective_H(
                    vec,
                    i,
                    left_env, 
                    right_env, 
                    H_1, 
                    H_2, 
                    opt_path
                    )
            )
    
    eigvals, eigvecs = eigs(full_env, k=1, which='LR', v0=two_site_vec, tol=1e-8, maxiter=100)

    # SVD and MPS update performed for each B separately
    start_index = 0
    eigvec_fixed_E = None
    for E in modules_sorted:
        vec_size_fixed_E = env_dims_left[i][E] * env_dims_right[i][E]
        eigvec_fixed_E = eigvecs[start_index : start_index + vec_size_fixed_E,:]
        start_index += vec_size_fixed_E

        eigvec_matrix = eigvec_fixed_E.reshape(env_dims_left[i][E], env_dims_right[i][E])
        M, S, N_H = np.linalg.svd(eigvec_matrix, full_matrices=False)
        count_significant = max(1, np.sum(S > tolerance_in_S))
        max_bond_dim = min(count_significant, max_chi_mps) # choose the desired dimension or limit to available S size
        M = M[:, :max_bond_dim]  # Trim columns of U
        S = S[:max_bond_dim]  # Trim singular values
        N_H = N_H[:max_bond_dim, :]  # Trim rows of V_H
        chi[i][E] =  max_bond_dim # New bond index of beta
        
        if is_moving_right:
            S_N = np.diag(S) @ N_H
            set_mps_site_to_matrix_fixed_B(mps[i], E, M, modules_sorted, allowed_module_pairs, d, is_left_boundary, False)
            set_mps_site_to_matrix_fixed_A(mps[i+1], E, S_N, modules_sorted, allowed_module_pairs, d, False, is_right_boundary)

        else:
            M_S = M @ np.diag(S)
            set_mps_site_to_matrix_fixed_B(mps[i], E, M_S, modules_sorted, allowed_module_pairs, d, is_left_boundary, False)
            set_mps_site_to_matrix_fixed_A(mps[i+1], E, N_H, modules_sorted, allowed_module_pairs, d, False, is_right_boundary)

def get_energy_density():
    global modules_sorted, v_left, v_right, modules_sorted, mps, H, N
    left_contracted_tensor = {}
    for B, D in product(modules_sorted, repeat=2):
        left_contracted_tensor[B, D] = None
        for A, C in product(modules_sorted, repeat=2):
            if (A, B) in mps[0].keys() and  (C, D) in mps[0].keys() and (A, C) in v_left.keys() and (A, B, C, D) in H[0].keys():
                if left_contracted_tensor[B, D] is None:
                    left_contracted_tensor[B, D] = oe.contract("ib,c,cijd,jf->bdf", mps[0][A, B], v_left[A, C], H[0][A, B, C, D], mps[0][C, D].conj())
                else:
                    left_contracted_tensor[B, D] += oe.contract("ib,c,cijd,jf->bdf", mps[0][A, B], v_left[A, C], H[0][A, B, C, D], mps[0][C, D].conj())

    for i in range(1, N-1):
        new_left_contracted_tensor = {}
        for B, D in product(modules_sorted, repeat=2):
            new_left_contracted_tensor[B, D] = None
            for A, C in product(modules_sorted, repeat=2):
                if (A, B) in mps[i].keys() and  (C, D) in mps[i].keys() and (A, C) in left_contracted_tensor.keys() and (A, B, C, D) in H[i].keys():
                    if new_left_contracted_tensor[B, D] is None:
                        new_left_contracted_tensor[B, D] = oe.contract("aib,ace,cijd,ejf->bdf", mps[i][A, B], left_contracted_tensor[A, C], H[i][A, B, C, D], mps[i][C, D].conj())
                    else:
                        new_left_contracted_tensor[B, D] += oe.contract("aib,ace,cijd,ejf->bdf", mps[i][A, B], left_contracted_tensor[A, C], H[i][A, B, C, D], mps[i][C, D].conj())
        left_contracted_tensor = new_left_contracted_tensor

    energy = 0
    energies = {}
    for B, D in product(modules_sorted, repeat=2):
        energies[B, D] = 0
        for A, C in product(modules_sorted, repeat=2):
            if (A, B) in mps[N-1].keys() and  (C, D) in mps[N-1].keys() and (A, C) in left_contracted_tensor.keys() and (A, B, C, D) in H[N-1].keys():
                partial_energy = oe.contract("ai,cijd,ej,ace,d->", mps[N-1][A, B], H[N-1][A, B, C, D], mps[N-1][C, D].conj(), left_contracted_tensor[A, C], v_right[B, D])
                energies[B, D] += partial_energy
                energy += partial_energy
    return energy / (N - 1)



def run(n_sweeps):

    total_time = timedelta()

    for n in range(n_sweeps):
        print(f"n: {n}")
        t0 = datetime.now()
        # Sweep from left to right (0 to i-2 inclusive)
        for i in range(N-1):
            print(f"Updating site {i}, sweeping right")
            update_left_environment(i)
            optimise_site_pair(i, is_moving_right=True)

        # Sweep from right to left (i-2 to 0 inclusive)
        for i in range(N-2, -1, -1):
            update_right_environment(i+1)
            optimise_site_pair(i, is_moving_right=False)
        
        t1 = datetime.now()

        energy = get_energy_density()
        total_time += t1-t0
        print(f"Energy per site: {energy}\n")
        print(f"Time of optimisation for sweep {n}: {t1-t0}\n")

    print(f"Total time: {total_time}")
    return energy


N = 10
max_chi_mps = 15
tolerance_in_S = 1e-4
opt_path = [(0, 3), (0, 3), (0, 2), (0, 1)]

module_name = "RepPsiA4"
labels_file = f"input/mpoHam_A4/{module_name}_ind.txt"
values_file = f"input/mpoHam_A4/{module_name}_converted_var.txt"
size_file = f"input/mpoHam_A4/{module_name}_size.txt"
H, d = get_H_and_d_from_files(N, labels_file, values_file, size_file)

allowed_module_pairs, allowed_vertical_module_pairs = get_allowed_module_pairs_from_H(H)
chi_mpo = get_chi_mpo(allowed_vertical_module_pairs, H)
boundary_modules_left = [(0, 0), (1, 1), (2, 2), (3, 3)] # top, bottom
boundary_modules_right = [(0, 0), (1, 1), (2, 2), (3, 3)] # top, bottom

modules = {M for pair in allowed_module_pairs for M in pair} # unique module labels
modules_sorted = sorted(modules)

mps = get_random_mps(N, d, max_chi_mps, allowed_module_pairs)
chi = get_chi_from_mps(N, mps)

env_dims_left, env_dims_right = initialise_env_dims(N)

put_mps_into_right_canonical_form(mps, modules_sorted, allowed_module_pairs, d, chi, N)
a = is_right_canonical_form_overall(modules, chi, mps)
v_left, v_right = get_boundary_vectors(chi_mpo, chi_mpo, allowed_vertical_module_pairs, boundary_modules_left, boundary_modules_right)

right_envs = get_right_environments(mps, v_right, H, allowed_vertical_module_pairs, N)
left_envs = [{pair : None for pair in allowed_vertical_module_pairs} for i in range(N)]

run(1)












