import numpy as np
from itertools import product

def get_Ising_T(module_category_name):
    # Cayley table for Z2 group
    C = np.zeros((2, 2), dtype=int)
    C[0, 0] = 0  # Z2 elements: 0 = identity, 1 = other
    C[0, 1] = 1
    C[1, 0] = 1
    C[1, 1] = 0

    # Initialize F-symbols for Z2 theory
    FZ2 = np.zeros((2, 2, 2, 2, 2, 2), dtype=complex)
    FVec = np.zeros((1, 2, 2, 1, 1, 2), dtype=complex)

    # Populate F-symbols
    for i, j in product(range(2), repeat=2):
        FVec[0, i, j, 0, 0, C[i, j]] = 1
        for k in range(2):
            FZ2[i, j, k, C[C[i, j], k], C[i, j], C[j, k]] = 1

    F = None
    # Choose which F-symbol to use
    if module_category_name == "Z2":
        F = FZ2
    elif module_category_name == "Vec":
        F = FVec

    nM = F.shape[0]
    T = np.zeros((nM, nM, nM, nM, 2, 2, 2, 2), dtype=complex)

    # Boltzmann weights
    x = np.sqrt(1 + np.sqrt(2)) # critical point
    boltz = np.array([1 / x, x])

    # Build the transfer matrix tensor
    for A, B, C_, D in product(range(nM), repeat=4):
        for i, j, k, l in product(range(2), repeat=4):
            s = 0
            for m in range(2):
                s += (F[A, j, l, D, B, m] *
                    np.conj(F[A, i, k, D, C_, m]) *
                    boltz[i]**0.5 * boltz[j]**0.5 *
                    boltz[k]**0.5 * boltz[l]**0.5)
            T[A, B, C_, D, i, j, k, l] = s

    # print(f"Non-zero count: {np.count_nonzero(T)}")
    # input()

    T_dict = {}

    for A, B, C_, D in product(range(nM), repeat=4):
        key = (A, B, C_, D)
        if module_category_name == "Vec":
            value = T[A, B, C_, D, :, :, :, :]  # shape (2, 2, 2, 2)
        elif module_category_name == "Z2":
            value = np.sum(T[A, B, C_, D, :, :, :, :]).reshape(1, 1, 1, 1)
            # print(value)
        T_dict[key] = value

    return T_dict


def get_Ising_H_fixed_point_XX():
    X = [[0,1],[1,0]]
    Z = [[1, 0], [0, -1]]
    I = [[1,0],[0,1]]
    O = [[0,0],[0,0]]
    local_H = [[I, X, O], [O, O, X], [O, O, I]]
    local_H = np.array(local_H)
    local_H = np.transpose(local_H, (0,2,3,1))
    return local_H

def get_Ising_H_fixed_point_Z():
    Z = [[1,0],[0,-1]]
    I = [[1,0],[0,1]]
    O = [[0,0],[0,0]]
    local_H = [[I, Z], [O, I]]
    local_H = np.array(local_H)
    local_H = np.transpose(local_H, (0,2,3,1))
    return local_H

def get_Ising_H(g):
    X = [[0,1],[1,0]]
    Z = [[g, 0], [0, -g]] 
    I = [[1,0],[0,1]]
    O = [[0,0],[0,0]]
    local_H = [[I, X, Z], [O, O, X], [O, O, I]]
    local_H = np.array(local_H)
    local_H = np.transpose(local_H, (0,2,3,1))
    return local_H

def get_Heisenberg_H():
    X = [[0,1],[1,0]]
    Y = [[0,-1j],[1j,0]]
    Z = [[1,0],[0,-1]] 
    I = [[1,0],[0,1]]
    O = [[0,0],[0,0]]
    local_H = [[I,X,Y,Z,O],[O,O,O,O,X],[O,O,O,O,Y],[O,O,O,O,Z],[O,O,O,O,I]]
    local_H = np.array(local_H)
    local_H = np.transpose(local_H, (0,2,3,1))
    return local_H