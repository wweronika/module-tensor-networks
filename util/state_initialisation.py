import numpy as np
import opt_einsum as oe

# Helper function to generate bond dimensions as consecutive powers of d from the edges, truncated at max_bond_dim 
def generate_bond_dim_pairs(N, d, max_bond_dim):
        pairs = []
        bond_dim_L = max(d, 2)
        for i in range(N // 2 + N % 2):
            bond_dim_R = min(bond_dim_L * max(d, 2), max_bond_dim)
            pairs.append((bond_dim_L, bond_dim_R))
            bond_dim_L = bond_dim_R
            
        if N % 2 == 0:
            pairs += [(b,a) for (a,b) in reversed(pairs)]
        else:
            pairs += [(b,a) for (a,b) in reversed(pairs[:-1])]
        
        return pairs

# Generate a random mMPS given the maximum bond dimension.
# The bond dimensions only vary with the site number and are the same for all module labels.
def get_random_mps(N, d_dict, max_bond_dim, allowed_module_pairs):
    mps = [{} for i in range(N)]
    for module_pair in allowed_module_pairs:

        d = d_dict[module_pair]
        A_leftmost = np.random.rand(d, max(d, 2)) + 1.0j * np.random.rand(d, max(d, 2))
        A_rightmost = np.random.rand(max(d, 2), d) + 1.0j * np.random.rand(max(d, 2), d)

        mps[0][module_pair] = A_leftmost
        mps[N-1][module_pair] = A_rightmost

        bond_dim_pairs = generate_bond_dim_pairs(N-2, d, max_bond_dim)

        for i, (bond_dim_L, bond_dim_R) in enumerate(bond_dim_pairs):
            A = np.random.rand(bond_dim_L, d, bond_dim_R) + 1.0j *  np.random.rand(bond_dim_L, d, bond_dim_R)
            mps[i+1][module_pair] = A
    return mps

# Get a single random mMPS tensor given physical dimensions d[A, B] and chi[A] at the single site
def get_random_mps_site(d, chi, allowed_module_pairs):
    mps = {}
    for A, B in allowed_module_pairs:
        mps[A, B] = np.random.rand(chi[A], d[A, B], chi[B]) + 1.0j * np.random.rand(chi[A], d[A, B], chi[B])
    return mps

# Helper function to deduce bond dimensions from mMPS
def get_chi_from_mps(N, mps):
    chis = []
    for i in range(N - 1):
        chi = {}
        for (A, B), mps_tensor in mps[i].items():
            if B not in chi.keys():
                if i == 0:
                    chi[B] = mps_tensor.shape[1]
                else:
                    chi[B] = mps_tensor.shape[2]
        chis.append(chi)
    return chis

# Helper function to deduce bond dimensions for a single mMPS site
def get_chi_from_mps_site(N, mps):
    chi = {}
    for (A, B), mps_tensor in mps.items():
        # Alternatively can read off chi[A] = mps_tensor.shape[0]. 
        # These should be identical.
        if B not in chi.keys():
            chi[B] = mps_tensor.shape[2]
    return chi

# Normalise all mMPS sites
def normalise_mps(mps):
    for i in range(len(mps)):
        normalise_mps_site(i)

# Normalise ith site when passing the entire mMPS
def normalise_mps_site(mps, i):
    norm = get_mps_norm(i)
    for A, B in mps[i].keys():
        mps[i][A, B] /= np.sqrt(norm)

# Normalise a single mMPS site
def normalise_mps_site(mps_site, is_boundary):
    norm = get_mps_site_norm(mps_site, is_boundary)
    for A, B in mps_site.keys():
        mps_site[A, B] /= np.sqrt(norm)
    norm = get_mps_site_norm(mps_site, is_boundary)

def normalise_mps_site_for_fixed_A(mps_site, A, is_boundary, modules):
    norm_A = 0
    for B in modules:
        if (A, B) in mps_site.keys():
            if is_boundary:
                norm_A += oe.contract('ab,ab->', mps_site[A, B], mps_site[A, B].conj())
            else:
                norm_A += oe.contract('aib,aib->', mps_site[A, B], mps_site[A, B].conj())  
    for B in modules:
        if (A, B) in mps_site.keys():
            mps_site[A, B] /= np.sqrt(norm_A)

def normalise_mps_site_for_fixed_B(mps_site, B, is_boundary, modules):
    norm_B = 0
    for A in modules:
        if (A, B) in mps_site.keys():
            if is_boundary:
                norm_B += oe.contract('ab,ab->', mps_site[A, B], mps_site[A, B].conj())
            else:
                norm_B += oe.contract('aib,aib->', mps_site[A, B], mps_site[A, B].conj())  
    for A in modules:
        if (A, B) in mps_site.keys():
            mps_site[A, B] /= np.sqrt(norm_B)

# Get the norm of the ith site when passing the entire mMPS
def get_mps_norm(mps, i):
    norm = 0
    N = len(mps)

    for A, B in mps[i].keys():
        if i == 0:
            norm += oe.contract('ib,ib->', mps[i][A, B], mps[i][A, B].conj())
        elif i == N - 1:
            norm += oe.contract('ai,ai->', mps[i][A, B], mps[i][A, B].conj())
        else:
            norm += oe.contract('aib,aib->', mps[i][A, B], mps[i][A, B].conj())
    return norm

# Get the norm of a single mps site
def get_mps_site_norm(mps_site, is_boundary):
    norm = 0
    for A, B in mps_site.keys():
        if is_boundary:
            norm += oe.contract('ab,ab->', mps_site[A, B], mps_site[A, B].conj())
        else:
            norm += oe.contract('aib,aib->', mps_site[A, B], mps_site[A, B].conj())
    return norm

# Get the total dimension of a right eigenvector of the transfer matrix of an MPS site
def get_T_right_eigenvec_shape(mps_site, chi):
    right_modules_of_mps_site = set()
    for A, B in mps_site.keys():
        right_modules_of_mps_site.add(B)
    shape = 0
    for B in right_modules_of_mps_site:
        shape += chi[B] ** 2
    return shape

# Assign new values to the MPS, where the matrix M keeps B (right module index) fixed
def set_mps_site_to_matrix_fixed_B(mps_site, B, M, modules_sorted, allowed_module_pairs, d, is_left_boundary, is_right_boundary):
    start_row = 0
    if is_left_boundary:
        for A in modules_sorted:
            if (A, B) in allowed_module_pairs:
                mps_site[A, B] = M[start_row:start_row + d[A, B], :]
                start_row += d[A, B]
    # TODO check if correct, perhaps transpose needed as well as transpose somewhere else
    # this version should work but may be different to usual notation
    elif is_right_boundary:
        for A in modules_sorted:
            if (A, B) in allowed_module_pairs:
                mps_site[A, B] = M[start_row:start_row + d[A, B], :]
                start_row += d[A, B]
    else:
        _, new_chi_R = M.shape
        for A in modules_sorted:
            if (A, B) in allowed_module_pairs:
                chi_L = mps_site[A, B].shape[0]
                new_shape = (chi_L, d[A, B], new_chi_R)
                d_times_chi_L = chi_L * d[A, B]
                mps_site[A, B] = M[start_row:start_row + d_times_chi_L, :].reshape(new_shape)
                start_row += d_times_chi_L

# Assign new values to the MPS, where the matrix M keeps A (left module index) fixed
def set_mps_site_to_matrix_fixed_A(mps_site, A, M, modules_sorted, allowed_module_pairs, d, is_left_boundary, is_right_boundary):
    start_column = 0
    if is_left_boundary:
        for B in modules_sorted:
            if (A, B) in allowed_module_pairs:
                chi_R = mps_site[A, B].shape[1]
                new_shape = (d[A, B], chi_R)
                d_times_chi_R = chi_R * d[A, B]
                mps_site[A, B] = M[:, start_column : start_column + d_times_chi_R].reshape(new_shape)
                start_column += d_times_chi_R

    elif is_right_boundary:
        for B in modules_sorted:
            if (A, B) in allowed_module_pairs:
                mps_site[A, B] = M[:, start_column : start_column + d[A, B]]
                start_column += d[A, B]

    else:
        new_chi_L, _ = M.shape
        for B in modules_sorted:
            if (A, B) in allowed_module_pairs:
                chi_R = mps_site[A, B].shape[2]
                new_shape = (new_chi_L, d[A, B], chi_R)
                d_times_chi_R = chi_R * d[A, B]
                mps_site[A, B] = M[:, start_column : start_column + d_times_chi_R].reshape(new_shape)
                start_column += d_times_chi_R

# Multiply mMPS by a matrix from the right. Assumes M is the Bth entry of the matrix dict.
def mps_times_matrix(mps_site, B, M, modules_sorted, allowed_module_pairs, is_left_boundary=False, is_right_boundary=False):
    if is_right_boundary:
        pass
    for A in modules_sorted:
        if (A, B) in allowed_module_pairs:
            if is_left_boundary:
                mps_site[A, B] = oe.contract('ib,bc->ic', mps_site[A, B], M)
            else:
                mps_site[A, B] = oe.contract('aib,bc->aic', mps_site[A, B], M)

# Multiply mMPS by a matrix from the left. Assumes M is the Ath entry of the matrix dict.
def matrix_times_mps(mps_site, A, M, modules_sorted, allowed_module_pairs, is_left_boundary=False, is_right_boundary=False):
    if is_left_boundary:
        pass
    for B in modules_sorted:
        if (A, B) in allowed_module_pairs:
            if is_right_boundary:
                mps_site[A, B] = oe.contract('ab,bi->ai', M, mps_site[A, B])
            else:
                mps_site[A, B] = oe.contract('ab,bic->aic', M, mps_site[A, B])


