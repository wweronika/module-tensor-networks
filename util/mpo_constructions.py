import numpy as np
from itertools import product

import numpy as np
from itertools import product

def get_XXZ_vumps(module_category_name,q,trunc):
    T_list = [[{} for i in range(3)] for i in range(3)]
    if module_category_name == "Rep(Uq(sl(2)))":
        nM = trunc
        for A, B in product(range(nM), repeat = 2):
            if abs(A - B) != 1:
                continue
            T_list[0][0][A,B,A,B] = np.reshape(1,(1,1,1,1))
            T_list[2][2][A,B,A,B] = np.reshape(1,(1,1,1,1))
            for D in range(nM):
                if abs(A - D) != 1:
                    continue
                T_list[0][1][A,B,A,D] = np.reshape(calcF(A,1,2,D,B,1,q**2),(1,1,1,1))
            for C in range(nM):
                if abs(B - C) != 1:
                    continue
                T_list[1][2][A,B,C,B] = np.reshape(calcF(A,2,1,B,C,1,q**2),(1,1,1,1))

    elif module_category_name == "Vec":
        T_list[0][0][0,0,0,0] = np.reshape(np.identity(2),(1,2,2,1))
        T_list[2][2][0,0,0,0] = np.reshape(np.identity(2),(1,2,2,1))
        T_list[0][1][0,0,0,0] = np.zeros((1, 2, 2, 3), dtype=complex)
        T_list[1][2][0,0,0,0] = np.zeros((3, 2, 2, 1), dtype=complex)
        for i, j, k in product(range(2),range(2),range(3)):
            T_list[0][1][0,0,0,0][0,i,j,k] = calcW(1,2,1,i,k,j,q**2)
            print(T_list[0][1][0,0,0,0][0,i,j,k])
            T_list[1][2][0,0,0,0][k,i,j,0] = calcW(2,1,1,k,j,i,q**2)
            print(T_list[1][2][0,0,0,0][k,i,j,0])

    return T_list

def get_XXZ_dmrg(module_category_name,q,trunc):
    T_dict = {}
    if module_category_name == "Rep(Uq(sl(2)))":
        nM = trunc
        for A, B in product(range(nM), repeat = 2):
            if abs(A - B) != 1:
                continue
            T_dict[A,B,A,B] = np.zeros((3,1,1,3))
            T_dict[A,B,A,B][0,0,0,0] = 1
            T_dict[A,B,A,B][2,0,0,2] = 1
            for D in range(nM):
                if abs(A - D) != 1:
                    continue
                T_dict[A,B,A,D] = np.zeros((3,1,1,3))
                T_dict[A,B,A,D][0,0,0,1] = calcF(A,1,2,D,B,1,q**2)
            for C in range(nM):
                if abs(B - C) != 1:
                    continue
                T_dict[A,B,C,B] = np.zeros((3,1,1,3))
                T_dict[A,B,C,B][1,0,0,2] = calcF(A,2,1,B,C,1,q**2)

    elif module_category_name == "Vec":
        T_dict[0,0,0,0] = np.zeros((5,2,2,5))
        T_dict[0,0,0,0][0,:,:,0] = np.identity(2)
        T_dict[0,0,0,0][4,:,:,4] = np.identity(2)
        for i, j, k in product(range(2),range(2),range(3)):
            T_dict[0,0,0,0][0,i,j,k+1] = calcW(1,2,1,i,k,j,q**2)
            T_dict[0,0,0,0][k+1,i,j,4] = calcW(2,1,1,k,j,i,q**2)

    return T_dict

def calcW(a, b, c, k, l, m, q):

    if k > a or l > b or m > c:
        return 0

    a = a / 2
    b = b / 2
    c = c / 2
    k = mlabel(a, k)
    l = mlabel(b, l)
    m = mlabel(c, m)

    if k + l - m != 0:
        return 0

    if c not in np.arange(abs(a - b), a + b + 1):
        return 0

    output = (
        q ** (((a + b - c) * (a + b + c + 1) + 2 * (a * l - b * k)) / 4)
        * Delta(a, b, c, q)
        * np.sqrt(qfac(a - k, q))
        * np.sqrt(qfac(a + k, q))
        * np.sqrt(qfac(b - l, q))
        * np.sqrt(qfac(b + l, q))
        * np.sqrt(qfac(c - m, q))
        * np.sqrt(qfac(c + m, q))
        * np.sqrt(qn(2 * c + 1, q))
    )

    lower = int(np.ceil(max(0, -(c - b + k), -(c - a - l))))
    upper = int(np.floor(min(a + b - c, a - k, b + l)))

    total_sum = 0
    if output != 0:
        for n in range(lower, upper + 1):
            term = (
                (-1) ** n
                * q ** (-(n * (a + b + c + 1)) / 2)
                / qfac(n, q)
                / qfac(a - k - n, q)
                / qfac(b + l - n, q)
                / qfac(a + b - c - n, q)
                / qfac(c - b + k + n, q)
                / qfac(c - a - l + n, q)
            )
            total_sum += term

    return output * total_sum

def calcF(a, b, c, d, e, f, q):

    a = a / 2
    b = b / 2
    c = c / 2
    d = d / 2
    e = e / 2
    f = f / 2

    if (e not in np.arange(a - b, a + b + 1) or
        f not in np.arange(b - c, b + c + 1) or
        d not in np.arange(e - c, e + c + 1) or
        d not in np.arange(a - f, a + f + 1)):
        return 0

    output = (
        (-1) ** int(a + b + c + d)
        * Delta(a, b, e, q)
        * Delta(c, d, e, q)
        * Delta(b, c, f, q)
        * Delta(a, d, f, q)
        * np.sqrt(qn(2 * e + 1, q))
        * np.sqrt(qn(2 * f + 1, q))
    )

    lower = int(np.ceil(max(a + b + e, c + d + e, b + c + f, a + d + f)))
    upper = int(np.floor(min(a + b + c + d, a + c + e + f, b + d + e + f)))

    total_sum = 0
    if output != 0:
        for n in range(lower, upper + 1):
            term = (
                (-1) ** n
                * qfac(n + 1, q)
                / qfac(a + b + c + d - n, q)
                / qfac(a + c + e + f - n, q)
                / qfac(b + d + e + f - n, q)
                / qfac(n - a - b - e, q)
                / qfac(n - c - d - e, q)
                / qfac(n - b - c - f, q)
                / qfac(n - a - d - f, q)
            )
            total_sum += term

    return output * total_sum


def qn(n, q):
    return (q ** (n / 2) - q ** (-n / 2)) / (q ** (1 / 2) - q ** (-1 / 2))


def qfac(n, q):
    if int(n) != n:
        raise ValueError("ERROR: non-integer input to qfac")
    if n < 0:
        raise ValueError("ERROR: negative input to qfac")

    n = int(n)
    result = 1
    for i in range(1, n + 1):
        result *= qn(i, q)
    return result


def Delta(a, b, c, q):
    if (a > b + c or b > a + c or c > a + b or (a + b + c) % 1 != 0):
        return 0
    return (
        np.sqrt(qfac(a + b - c, q))
        * np.sqrt(qfac(a - b + c, q))
        * np.sqrt(qfac(-a + b + c, q))
        / np.sqrt(qfac(a + b + c + 1, q))
    )

def mlabel(j, m):
    temp = np.arange(-j, j + 1)
    return temp[int(m)]
    

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

    T_list = {}

    for A, B, C_, D in product(range(nM), repeat=4):
        key = (A, B, C_, D)
        if module_category_name == "Vec":
            value = T[A, B, C_, D, :, :, :, :]  # shape (2, 2, 2, 2)
        elif module_category_name == "Z2":
            value = np.sum(T[A, B, C_, D, :, :, :, :]).reshape(1, 1, 1, 1)
            # print(value)
        T_list[key] = value

    return T_list


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