# Code for three-site exact diagonalisation

from util.mpo_constructions import get_XXZ_dmrg
from util.input_parsing import get_d_from_H, get_allowed_module_pairs_from_H, get_chi_mpo

import numpy as np
import opt_einsum as oe
from itertools import product

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

N = 3
n_mod = 5
q = 1.1
module_name = 'Rep(Uq(sl(2)))'
# module_name = 'Vec'
H_local = get_XXZ_dmrg(module_name, q, n_mod)
H = [H_local for i in range(N)]
d = 2

allowed_module_pairs, allowed_vertical_module_pairs = get_allowed_module_pairs_from_H(H)
chi_mpo = get_chi_mpo(allowed_vertical_module_pairs, H)

modules = {M for pair in allowed_module_pairs for M in pair} # unique module labels
modules_sorted = sorted(modules)

boundary_modules_left = [(0, 0)] # top, bottom
boundary_modules_right = [(M, M) for M in modules_sorted] # top, bottom
# boundary_modules_right = [(0, 0)]

H_3_site_dict = {}
H_2_site_dict = {}
vl, vr = get_boundary_vectors(chi_mpo, chi_mpo, allowed_vertical_module_pairs, boundary_modules_left, boundary_modules_right)

for A, B, C, D, E, F, G, H8 in product(modules_sorted, repeat=8):
    if (A, B, E, F) in H[0].keys() and (B, C, F, G) in H[1].keys() and (C, D, G, H8) in H[2].keys():
        print((A, B, C, D, E, F, G, H8))
        H_3_site_dict[A, B, C, D, E, F, G, H8] = oe.contract('a,aijb,bklc,cmnd,d->ikmjln', vl[A, E], H[0][A, B, E, F], H[1][B, C, F, G], H[2][C, D, G, H8], vr[D, H8])


size_tuple = (n_mod, d, n_mod, d, n_mod, d, n_mod, n_mod, d, n_mod, d, n_mod, d, n_mod)
H_3_site_tensor = np.zeros(size_tuple, dtype=np.complex128)

for (A, B, C, D, E, F, G, H8), tensor in H_3_site_dict.items():
    H_3_site_tensor[A, :, B, :, C, :, D,    E, :, F, :, G, :, H8] = tensor

env_size = n_mod * d * n_mod * d * n_mod * d * n_mod

H_3_site_matrix = H_3_site_tensor.reshape(env_size, env_size)
print(np.linalg.norm(H_3_site_matrix))

for A, B, C, D, E, F in product(modules_sorted, repeat=6):
    if (A, B, D, E) in H[0].keys() and (B, C, E, F) in H[1].keys():
        H_2_site_dict[A, B, C, D, E, F] = oe.contract('a,aikb,bjlc,c->ijkl', vl[A, D], H[0][A, B, D, E], H[1][B, C, E, F], vr[C, F])

size_tuple = (n_mod, d, n_mod, d, n_mod,      n_mod, d, n_mod, d, n_mod)
H_2_site_tensor = np.zeros(size_tuple, dtype=np.complex128)

for (A, B, C, D, E, F), tensor in H_2_site_dict.items():
    H_2_site_tensor[A, :, B, :, C,     D, :, E, :, F] = tensor

env_size = n_mod * d * n_mod * d * n_mod

H_2_site_matrix = H_2_site_tensor.reshape(env_size, env_size)
print(np.linalg.norm(H_2_site_matrix))


eigenvalues, eigenvectors = np.linalg.eig(H_2_site_matrix)
print(set(np.abs(eigenvalues)))

eigenvalues, eigenvectors = np.linalg.eig(H_3_site_matrix)
print(set(np.abs(eigenvalues)))