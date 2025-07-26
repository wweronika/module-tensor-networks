from scipy.linalg import polar, sqrtm
from scipy.sparse.linalg import bicgstab, eigs, LinearOperator
from math import sqrt

from util.input_parsing import *
from util.mpo_constructions import *
from util.state_initialisation import *
from util.gauge_fixing import *
from util.data_conversion import *

H = get_H_for_vumps('input-vumps/ising_Vec_values_no_minus.txt')
H = get_H_for_vumps('input-vumps/ising_RepZ2_values.txt')

chi_CA = len(H)
chi_mpo = get_chi_mpo_vumps('input-vumps/ising_RepZ2_shapes.txt', chi_CA)
d = {(0, 0) : 1, (0, 1) : 1, (1, 0) : 1, (1, 1) : 1}

allowed_module_pairs, allowed_vertical_module_pairs = get_allowed_module_pairs_from_H_vumps(H)
print(allowed_module_pairs, allowed_vertical_module_pairs)
allowed_module_pairs_sorted = sorted(allowed_module_pairs)

modules = {M for pair in allowed_module_pairs for M in pair} # unique module labels
modules_sorted = sorted(modules)

# CHI_CONST = 8
# chi = {M: CHI_CONST for M in modules} # Can be made uniform
chi = {0: 7, 1: 8}

mps = get_random_mps_site(d, chi, allowed_module_pairs)

total_mps_shape = 0
for A, B in allowed_module_pairs:
    total_mps_shape += chi[A] * d[A, B] * chi[B]

total_T_eigenvec_shape = 0
for A in modules:
    total_T_eigenvec_shape += chi[A]**2

def get_T(mps_site):
    T = {}
    for A, B in allowed_module_pairs:
        T[A, B] = oe.contract('aib,cid->acbd', mps_site[A, B], mps_site[A, B].conj())
    return T

def get_T_H(mps, H):
    max_a = len(H) # Outermost dimension of H, i.e. dim of cellular automata. Same as max_b.
    T_H = [[defaultdict(lambda: None) for _ in range(max_a)] for _ in range(max_a)]

    for a in range(max_a):
        for b in range(max_a):
            for A, B, C, D in H[a][b].keys():
                T_H[a][b][A, B, C, D] = oe.contract('aib,cijd,ejf->acebdf', mps[A, B], H[a][b][A, B, C, D], mps[C, D].conj())
    T_H = [[dict(T_R_H_ab) for T_R_H_ab in T_R_H_a] for T_R_H_a in T_H]
    return T_H

def get_T_H_reshaped_to_matrix(T_H):
    max_a = len(T_H)
    T_H_reshaped = [[defaultdict(lambda: None) for _ in range(max_a)] for _ in range(max_a)]

    for a in range(max_a):
        for b in range(max_a):
            for A, B, C, D in T_H[a][b].keys():
                chi_A, chi_mpo_a_AC, chi_C, chi_B, chi_mpo_b_BD, chi_D = T_H[a][b][A, B, C, D].shape
                new_shape = (chi_A * chi_mpo_a_AC * chi_C, chi_B * chi_mpo_b_BD * chi_D)
                T_H_reshaped[a][b][A, B, C, D] = T_H[a][b][A, B, C, D].reshape(new_shape)
    
    return T_H_reshaped

def mps_dict_to_matrix_fixed_B(mps, B):
    result = []
    for A in modules_sorted:
        if (A, B) in mps.keys():
            result.append(mps[A, B].reshape(chi[A] * d[A, B], chi[B]))
    return np.concatenate(result, axis=0) if result else np.zeros((0, chi[B]))

def mps_dict_to_matrix_fixed_A(mps, A):
    result = []
    for B in modules_sorted:
        if (A, B) in mps.keys():
            result.append(mps[A, B].reshape(chi[A], d[A, B] * chi[B]))

    return np.concatenate(result, axis=1) if result else np.zeros((chi[A], 0))

def mps_vec_to_dict(mps_vec):
    mps_dict = {}
    i = 0
    for A, B in allowed_module_pairs_sorted:
        size = chi[A] * d[A, B] * chi[B]
        mps_dict[A, B] = mps_vec[i : i + size].reshape(chi[A], d[A, B], chi[B])
        i += size
    return mps_dict

def mps_dict_to_vec(mps_dict):
    vecs = []
    for A, B in allowed_module_pairs_sorted:
        vecs.append(mps_dict[A, B].flatten())  # Flatten to 1D
    return np.concatenate(vecs)

def C_vec_to_dict(C_vec):
    C_dict = {}
    i = 0
    for A in modules_sorted:
        size = chi[A] ** 2
        C_dict[A] = C_vec[i : i + size].reshape(chi[A], chi[A])
        i += size
    return C_dict

def C_dict_to_vec(C_dict):
    vecs = []
    for A in modules_sorted:
        vecs.append(C_dict[A].flatten())  # Flatten to 1D
    return np.concatenate(vecs)

def mps_matrix_to_dict_fixed_B(mps_matrix, B):
    mps = {}
    row_start = 0
    for A in modules_sorted:
        if (A, B) in allowed_module_pairs:
            rows = chi[A] * d[A, B]
            mps_block = mps_matrix[row_start : row_start + rows, :]
            mps[A, B] = mps_block.reshape(chi[A], d[A, B], chi[B])
            row_start += rows
    return mps

def mps_matrix_to_dict_fixed_A(mps_matrix, A):
    mps = {}
    col_start = 0
    for B in modules_sorted:
        if (A, B) in allowed_module_pairs:
            cols = d[A, B] * chi[B]
            mps_block = mps_matrix[:, col_start : col_start + cols]
            mps[A, B] = mps_block.reshape(chi[A], d[A, B], chi[B])
            col_start += cols
    return mps

# TODO unify with the other conversion functions
def convert_vec_to_dict_diagonal(R_vec):
    result = {}
    i = 0
    for A in modules_sorted:
        result[A] = R_vec[i : i + chi[A]**2]
        i += chi[A]**2
    return result

def convert_dict_diagonal_to_vec(R):
    result = []
    for A in modules_sorted:
        result.append(R[A, A].flatten())
    return np.concatenate(result)

def generate_random_starting_mps():
    mps = {}
    T = {}
    A_L = {}
    A_R = {}
    for A, B in allowed_module_pairs:
        mps[A, B] = np.random.rand(chi[A], d[A, B], chi[B]) + 1j * np.random.rand(chi[A], d[A, B], chi[B])
        mps_norm = oe.contract('aib,aib->', mps[A, B], mps[A, B].conj())
        mps[A, B] /= sqrt(float(mps_norm.real))

        T[A, B] = oe.contract('aib,cid->acbd', mps[A, B], mps[A, B].conj())

    T_matrix = T_dict_to_matrix(total_T_eigenvec_shape, T, modules_sorted, chi)
    eigval, C_tilde = eigs(T_matrix, k=1, which='LR')

    C_tilde = C_vec_to_dict(C_tilde)
    C_tilde_sqrt = {}
    for M in modules:
        C_tilde_sqrt[M] = sqrtm(C_tilde[M])

    C_tilde_sqrt_mps = {}
    
    for B in modules_sorted:
        A_L_matrix_fixed_B = None
        while True:
            for A in modules_sorted:
                if (A, B) in allowed_module_pairs:
                    C_tilde_sqrt_mps[A, B] = oe.contract('ab,bic->aic', C_tilde_sqrt[A], mps[A, B]).reshape(chi[A] * d[A, B], chi[B])
            
            C_tilde_sqrt_mps_fixed_B = mps_dict_to_matrix_fixed_B(C_tilde_sqrt_mps, B)
            A_L_matrix_fixed_B, C_tilde_sqrt_2_fixed_B = polar(C_tilde_sqrt_mps_fixed_B, side="right")
            error = np.linalg.norm(C_tilde_sqrt_2_fixed_B - C_tilde_sqrt[B])
            C_tilde_sqrt[B] = C_tilde_sqrt_2_fixed_B
            print(f"error L: {error}")
            if error < 1e-8:
                break
        A_L_dict_fixed_B = mps_matrix_to_dict_fixed_B(A_L_matrix_fixed_B, B)
        A_L.update(A_L_dict_fixed_B)

    T_L = {}
    for A, B in allowed_module_pairs:
        T_L[A, B] = oe.contract('aib,cid->acbd', A_L[A, B], A_L[A, B].conj())
    T_L_matrix = T_dict_to_matrix(total_T_eigenvec_shape, T_L, modules_sorted, chi)

    eigval, C_R_tilde = eigs(T_L_matrix, k=1, which='LR')
    C_R_tilde = C_vec_to_dict(C_R_tilde)
    C_R_tilde_sqrt = {}
    for M in modules:
        C_R_tilde_sqrt[M] = sqrtm(C_R_tilde[M])

    A_L_C_R_tilde_sqrt = {}
    for A in modules:
        A_R_matrix_fixed_A = None
        while True:
            # TODO optimise
            for B in modules:
                if (A, B) in allowed_module_pairs:
                    A_L_C_R_tilde_sqrt[A, B] = oe.contract('aib,bc->aic', A_L[A, B], C_R_tilde_sqrt[B]).reshape(chi[A], d[A, B] * chi[B])

            A_L_C_R_tilde_sqrt_fixed_A = mps_dict_to_matrix_fixed_A(A_L_C_R_tilde_sqrt, A)
            A_R_matrix_fixed_A, C_R_tilde_sqrt_2_fixed_A = polar(A_L_C_R_tilde_sqrt_fixed_A, side="left")
            error = np.linalg.norm(C_R_tilde_sqrt_2_fixed_A - C_R_tilde_sqrt[A])
            C_R_tilde_sqrt[A] = C_R_tilde_sqrt_2_fixed_A

            if error < 1e-8:
                break
        A_R_dict_fixed_A = mps_matrix_to_dict_fixed_A(A_R_matrix_fixed_A, A)
        A_R.update(A_R_dict_fixed_A)

    A_C = {}
    for (A, B) in allowed_module_pairs:
        A_C[A, B] = oe.contract('ab,bic->aic', C_R_tilde_sqrt[A], A_R[A, B])

    return A_L, A_C, A_R, C_R_tilde_sqrt
    
def initialise_L(chi_CA, T_L_eigenvec_L):
    L = [{} for a in range(chi_CA)]
    for A in modules:
        L[0][A, A] = T_L_eigenvec_L[A]
    return L

def initialise_R(chi_CA, T_R_eigenvec_R):
    R = [{} for a in range(chi_CA)]
    for A in modules:
        R[-1][A, A] = T_R_eigenvec_R[A]
    return R

def get_A_L_and_A_R(A_C, C):
    A_L = {}
    A_R = {}

    for M in modules:
        A_C_l_matrix_module_B = mps_dict_to_matrix_fixed_B(A_C, M)
        C_l_matrix_module_B = C[M]

        U_A_C_l, P_A_C_l = polar(A_C_l_matrix_module_B, side="right")
        U_C_l, P_C_l = polar(C_l_matrix_module_B, side="right")

        A_L_matrix_module_B = U_A_C_l @ U_C_l.T.conj()
        A_L_dict_module_B = mps_matrix_to_dict_fixed_B(A_L_matrix_module_B, M)

        A_L.update(A_L_dict_module_B)
    
    for M in modules:

        A_C_r_matrix_module_A = mps_dict_to_matrix_fixed_A(A_C, M)
        C_r_matrix_module_A = C[M]

        U_A_C_r, P_A_C_r = polar(A_C_r_matrix_module_A, side="left")
        U_C_r, P_C_r = polar(C_r_matrix_module_A, side="left")

        A_R_matrix_module_A = U_C_r.T.conj() @  U_A_C_r
        A_R_dict_module_A = mps_matrix_to_dict_fixed_A(A_R_matrix_module_A, M)

        A_R.update(A_R_dict_module_A)
    
    return A_L, A_R

def apply_effective_H_to_mps(A_vec, L, R, H):
    A_dict = mps_vec_to_dict(A_vec)
    result = {(C, D) : np.zeros((chi[C], d[C, D], chi[D]), dtype=np.complex128) for (C, D) in allowed_module_pairs}

    for a in range(chi_CA):
        for b in range(chi_CA):
            for C, D in allowed_module_pairs:
                # result[C, D] = np.zeros((chi[C], d[C, D], chi[D]), dtype=np.complex128)
                for A, B in allowed_module_pairs:
                    if (A, B, C, D) in H[a][b].keys():
                        L_tensor = L[a][A, C].T.reshape(chi[A], chi_mpo[a][A, C], chi[C])
                        R_tensor = R[b][B, D].reshape(chi[B], chi_mpo[b][B, D], chi[D])
                        result[C, D] += oe.contract('ace,aib,cijd,bdf->ejf', L_tensor, A_dict[A, B], H[a][b][A, B, C, D], R_tensor)

    result_vec = mps_dict_to_vec(result)
    return result_vec

def apply_effective_H_to_C(C_vec, L, R):
    C_dict = C_vec_to_dict(C_vec)
    result = {M : np.zeros((chi[M], chi[M]), dtype=np.complex128) for M in modules}
    for a in range(chi_CA):
        for A, C in allowed_vertical_module_pairs[a]:
            L_tensor = L[a][A, C].T.reshape(chi[A], chi_mpo[a][A, C], chi[C])
            R_tensor = R[a][A, C].reshape(chi[A], chi_mpo[a][A, C], chi[C])
            result[C] += oe.contract('ace,ab,bcf->ef', L_tensor, C_dict[A], R_tensor)
            result_vec = C_dict_to_vec(result) 
    return result_vec

def get_L(chi, chi_CA, chi_mpo, T_L, T_L_eigenvec_R, T_L_eigenvec_L, T_L_H_reshaped):

    L = initialise_L(chi_CA, T_L_eigenvec_L)
    Y_L = [{} for i in range(chi_CA)]

    for a in range(1, chi_CA):
        for B, D in allowed_vertical_module_pairs[a]:

            # Constructing Y_L
            # Y_L[a][B, C] = sum_b sum_AC T^{ab}_{ABCD} L_b_AC

            Y_L[a][B, D] = np.zeros((chi[B] * chi_mpo[a][B, D] * chi[D], 1), dtype=np.complex128)
            for b in range(a-1, -1, -1):
                for A, C in allowed_vertical_module_pairs[b]:
                    if (A, B, C, D) in T_L_H_reshaped[b][a].keys():
                        Y_L[a][B, D] += T_L_H_reshaped[b][a][A, B, C, D].T @ L[b][A, C]

            # T_L_H[a][a][A, B, C, D] is zero for entries non-diagonal in the module label
            if B != D or a != chi_CA:
                L[a][B, D] = Y_L[a][B, D]

        # Now, solve the A = C case for all module labels together.
        # T_L_H[a][a][A, B, A, B] can be non-zero. 
        # For now, assume it's identity, i.e. eigenvalue = 1.
        # Then, T_L_H[a][a][A, B, A, B] = T_R[A, B].
        if a == chi_CA - 1:
            Y_L_projected = {}
            for A in modules:
                Y_L_projected[A, A] = (np.eye((chi[A]**2)) - np.outer(T_L_eigenvec_R[A], T_L_eigenvec_L[A])).T @ Y_L[a][A, A]
            y = convert_dict_diagonal_to_vec(Y_L_projected)
            total_shape = y.shape[0]
            M_dict = {}
            for A, B in T_L.keys():
                M_dict[A, B] = T_L[A, B].reshape(chi[A]**2, chi[B]**2) - np.outer(T_L_eigenvec_R[A], T_L_eigenvec_L[B])
            M = np.eye(total_shape) - T_dict_to_matrix(total_shape, M_dict, modules_sorted, chi)
            M_mat = T_dict_to_matrix(total_shape, M_dict, modules_sorted, chi)
            x_vec, info = bicgstab(M.T, y)
            x = convert_vec_to_dict_diagonal(x_vec)
            for A in modules_sorted:
                L[a][A, A] = x[A].reshape((chi[A] * chi_mpo[a][A, A] * chi[A], 1))

    return L, Y_L

def get_R(chi, chi_CA, chi_mpo, T_R, T_R_eigenvec_L, T_R_eigenvec_R, T_R_H_reshaped):
    R = initialise_R(chi_CA, T_R_eigenvec_R)
    Y_R = [{} for i in range(chi_CA)]


    for a in range(chi_CA - 2, -1, -1):
        for A, C in allowed_vertical_module_pairs[a]:

            # Constructing Y_R
            # Y_R[a][A, C] = sum_b sum_BD T^{ab}_{ABCD} R_b_BD

            Y_R[a][A, C] = np.zeros((chi[A] * chi_mpo[a][A, C] * chi[C], 1), dtype=np.complex128)
            for b in range(a + 1, chi_CA):
                for B, D in allowed_vertical_module_pairs[b]:
                    if (A, B, C, D) in T_R_H_reshaped[a][b].keys():
                        Y_R[a][A, C] += T_R_H_reshaped[a][b][A, B, C, D] @ R[b][B, D]

            # T_R_H[a][a][A, B, C, D] is zero for entries non-diagonal in the module label
            if A != C or a != 0:
                R[a][A, C] = Y_R[a][A, C]

        # Now, solve the A = C case for all module labels together.
        # T_R_H[a][a][A, B, A, B] can be non-zero. 
        # For now, assume it's identity, i.e. eigenvalue = 1.
        # Then, T_R_H[a][a][A, B, A, B] = T_R[A, B].
        if a == 0:
            Y_R_projected = {}
            for A in modules:
                Y_R_projected[A, A] = ((np.eye((chi[A]**2)) - np.outer(T_R_eigenvec_R[A], T_R_eigenvec_L[A]))) @ Y_R[a][A, A]
            y = convert_dict_diagonal_to_vec(Y_R_projected)
            total_shape = y.shape[0]
            M_dict = {}
            for A, B in T_R.keys():
                M_dict[A, B] = T_R[A, B].reshape(chi[A]**2, chi[B]**2) - np.outer(T_R_eigenvec_R[A], T_R_eigenvec_L[B])
            M = np.eye(total_shape) - T_dict_to_matrix(total_shape, M_dict, modules_sorted, chi)
            x_vec, info = bicgstab(M, y)
            x = convert_vec_to_dict_diagonal(x_vec)
            for A in modules_sorted:
                R[a][A, A] = x[A].reshape((chi[A] * chi_mpo[a][A, A] * chi[A], 1))

    return R, Y_R

def apply_T_L_H_to_L(T_L_H_reshaped, L, allowed_vertical_module_pairs):
    chi_CA = len(L)
    result = [{} for a in range(chi_CA)]
    for a in range(chi_CA):
        for B, D in allowed_vertical_module_pairs[a]:
            result[a][B, D] = np.zeros((chi[B] * chi_mpo[a][B, D] * chi[D], 1), dtype=np.complex128)
            for b in range(a + 1):
                for A, C in allowed_vertical_module_pairs[b]:
                    if (A, B, C, D) in T_L_H_reshaped[b][a].keys():
                        result[a][B, D] += T_L_H_reshaped[b][a][A, B, C, D].T @ L[b][A, C]
    return result

def apply_T_R_H_to_R(T_R_H_reshaped, R, allowed_vertical_module_pairs):
    chi_CA = len(R)
    result = [{} for a in range(chi_CA)]
    for a in range(chi_CA):
        for A, C in allowed_vertical_module_pairs[a]:
            result[a][A, C] = np.zeros((chi[A] * chi_mpo[a][A, C] * chi[C], 1), dtype=np.complex128)
            for b in range(a, chi_CA):
                for B, D in allowed_vertical_module_pairs[b]:
                    if (A, B, C, D) in T_R_H_reshaped[a][b].keys():
                        result[a][A, C] += T_R_H_reshaped[a][b][A, B, C, D] @ R[b][B, D]
    return result

T_R_eigenvec_R = {A: np.eye(chi[A]).reshape(chi[A]**2, 1) for A in modules}
T_R_eigenvec_R_vec = C_dict_to_vec(T_R_eigenvec_R)
T_L_eigenvec_L = {A: np.eye(chi[A]).reshape(chi[A]**2, 1) for A in modules}
T_L_eigenvec_L_vec = C_dict_to_vec(T_L_eigenvec_L)


def vumps_iteration(A_L, A_R, C, H):

    A_C = {}
    A_C_2 = {}
    for A, B in allowed_module_pairs:
        A_C[A, B] = oe.contract('aib,bc->aic', A_L[A, B], C[B])
        A_C_2[A, B] = oe.contract('ab,bic->aic', C[A], A_R[A, B])
        print(f"BEFORE ||A_C - A_C_2|| modules {A}, {B}:{np.linalg.norm(A_C[A, B] - A_C_2[A, B])}")

    T_L = get_T(A_L)
    T_R = get_T(A_R)
    T_L_matrix = T_dict_to_matrix(total_T_eigenvec_shape, T_L, modules_sorted, chi)
    T_R_matrix = T_dict_to_matrix(total_T_eigenvec_shape, T_R, modules_sorted, chi)

    T_L_H = get_T_H(A_L, H)
    T_R_H = get_T_H(A_R, H)
    T_R_H_reshaped = get_T_H_reshaped_to_matrix(T_R_H)
    T_L_H_reshaped = get_T_H_reshaped_to_matrix(T_L_H)

    _, T_L_eigenvec_R_vec = eigs(T_L_matrix, k=1, which='LR')
    _, T_R_eigenvec_L_vec = eigs(T_R_matrix.T, k=1, which='LR')

    # Normalisation: (L|R) = 1
    T_L_eigenvec_R_vec /= np.dot(T_L_eigenvec_L_vec, T_L_eigenvec_R_vec)
    T_R_eigenvec_L_vec /= np.dot(T_R_eigenvec_R_vec, T_R_eigenvec_L_vec)

    T_L_eigenvec_R = C_vec_to_dict(T_L_eigenvec_R_vec)
    T_R_eigenvec_L = C_vec_to_dict(T_R_eigenvec_L_vec)

    R, Y_R = get_R(chi, chi_CA, chi_mpo, T_R, T_R_eigenvec_L, T_R_eigenvec_R, T_R_H_reshaped)
    L, Y_L = get_L(chi, chi_CA, chi_mpo, T_L, T_L_eigenvec_R, T_L_eigenvec_L, T_L_H_reshaped)
    

    L_T_L_H = apply_T_L_H_to_L(T_L_H_reshaped, L, allowed_vertical_module_pairs)
    T_R_H_R = apply_T_R_H_to_R(T_R_H_reshaped, R, allowed_vertical_module_pairs)

    results_L = {}
    results_R = {}
    for M in modules: 
        result_L = L_T_L_H[-1][M, M] - L[-1][M, M]
        result_R = T_R_H_R[0][M, M] - R[0][M, M]

        results_L[M] = result_L[0, 0]
        results_R[M] = result_R[0, 0]

        print(f"Energy density L for module {M}?: {result_L[0,0]}")
        print(f"Energy density R for module {M}?: {result_R[0,0]}")

    full_env_mps = LinearOperator(
        (total_mps_shape, total_mps_shape),
            matvec=lambda vec: apply_effective_H_to_mps(vec, L, R, H)
    )
    full_env_C = LinearOperator(
            (total_T_eigenvec_shape, total_T_eigenvec_shape),
            matvec=lambda vec: apply_effective_H_to_C(vec, L, R)
    )
    _, A_C_new_vec = eigs(full_env_mps, k=1, which='LR', v0=mps_dict_to_vec(A_C))
    _, C_new_vec = eigs(full_env_C, k=1, which='LR', v0=C_dict_to_vec(C))

    A_C_new_dict = mps_vec_to_dict(A_C_new_vec)
    C_new_dict = C_vec_to_dict(C_new_vec)

    A_L_new, A_R_new = get_A_L_and_A_R(A_C_new_dict, C_new_dict)

    tmp1 = {}
    tmp2 = {}
    for A, B in allowed_module_pairs:
        tmp1[A, B] = oe.contract('aib,bc->aic', A_L_new[A, B], C_new_dict[B])
        tmp2[A, B] = oe.contract('ab,bic->aic', C_new_dict[A], A_R_new[A, B])
        print(f"AFTER ||A_C - A_C_2|| modules {A}, {B}:{np.linalg.norm(tmp1[A, B] - tmp2[A, B])}")

    return A_L_new, A_R_new, C_new_dict

def run_vumps(n_iter, H):
    A_L, A_C, A_R, C = generate_random_starting_mps()
    A_L, A_R = get_A_L_and_A_R(A_C, C)
    for i in range(n_iter):
        print(f"i={i}")
        A_L, A_R, C = vumps_iteration(A_L, A_R, C, H)

run_vumps(10, H)
