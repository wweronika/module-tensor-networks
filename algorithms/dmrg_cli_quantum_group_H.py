#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import algorithms.dmrg as dmrg  # safe to import (no side effects now)

# ---------- helpers ----------
def project_root() -> Path:
    # algorithms/ is this file's parent; repo root is its parent
    return Path(__file__).resolve().parent.parent

def dataset_files(module_name: str):
    d = project_root() / "input" / "mpoHam_A4"
    labels = d / f"{module_name}_ind.txt"
    values = d / f"{module_name}_converted_var.txt"
    size   = d / f"{module_name}_size.txt"
    return labels, values, size

def cscalar(z):
    if isinstance(z, np.generic):
        z = np.asarray(z).item()
    if isinstance(z, complex):
        return {"re": float(z.real), "im": float(z.imag)}
    return float(z)

def clist(seq):
    out = []
    for x in seq:
        if isinstance(x, np.generic):
            x = np.asarray(x).item()
        if isinstance(x, complex):
            out.append([float(x.real), float(x.imag)])
        else:
            out.append([float(x), 0.0])
    return out
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--max_chi_mps", type=int, required=True)
    p.add_argument("--tolerance_in_S", type=float, required=True)
    p.add_argument("--module_name", type=str, required=True)
    p.add_argument("--boundary_modules_left",  type=str, required=True)   # JSON: [[m,m],...]
    p.add_argument("--boundary_modules_right", type=str, required=True)   # JSON: [[m,m],...]
    p.add_argument("--n_sweeps", type=int, required=True)
    p.add_argument("--q", type=float, required=True)
    p.add_argument("--n_irreps", type=int, required=True)
    args = p.parse_args()

    # fixed
    dmrg.opt_path = [(0,3),(0,3),(0,2),(0,1)]

    # bind globals
    dmrg.N = args.N
    dmrg.max_chi_mps = args.max_chi_mps
    dmrg.tolerance_in_S = args.tolerance_in_S

    # parse boundaries
    boundary_left  = [tuple(x) for x in json.loads(args.boundary_modules_left)]
    boundary_right = [tuple(x) for x in json.loads(args.boundary_modules_right)]

    # build problem
    H_local = dmrg.get_XXZ_dmrg(args.module_name, args.q, args.n_irreps)
    dmrg.H = [H_local for i in range(dmrg.N)]
    dmrg.d = dmrg.get_d_from_H(dmrg.H)
    # dmrg.H, dmrg.d = dmrg.get_H_and_d_from_files(dmrg.N, str(labels_file), str(values_file), str(size_file))

    dmrg.allowed_module_pairs, dmrg.allowed_vertical_module_pairs = dmrg.get_allowed_module_pairs_from_H(dmrg.H)
    dmrg.chi_mpo = dmrg.get_chi_mpo(dmrg.allowed_vertical_module_pairs, dmrg.H)
    # print(dmrg.d)
    # print(dmrg.allowed_module_pairs)
    # print(dmrg.allowed_vertical_module_pairs)
    # exit()
    dmrg.modules = {M for pair in dmrg.allowed_module_pairs for M in pair}
    dmrg.modules_sorted = sorted(dmrg.modules)
    dmrg.boundary_modules_left  = boundary_left
    dmrg.boundary_modules_right = boundary_right

    dmrg.mps = dmrg.get_random_mps(dmrg.N, dmrg.d, dmrg.max_chi_mps, dmrg.allowed_module_pairs)
    dmrg.chi = dmrg.get_chi_from_mps(dmrg.N, dmrg.mps)
    dmrg.env_dims_left, dmrg.env_dims_right = dmrg.initialise_env_dims(dmrg.N)
    dmrg.put_mps_into_right_canonical_form(dmrg.mps, dmrg.modules_sorted, dmrg.allowed_module_pairs, dmrg.d, dmrg.chi, dmrg.N)
    _ = dmrg.is_right_canonical_form_overall(dmrg.modules, dmrg.chi, dmrg.mps)

    dmrg.v_left, dmrg.v_right = dmrg.get_boundary_vectors(dmrg.chi_mpo, dmrg.chi_mpo,
        dmrg.allowed_vertical_module_pairs, dmrg.boundary_modules_left, dmrg.boundary_modules_right)
    dmrg.right_envs = dmrg.get_right_environments(dmrg.mps, dmrg.v_right, dmrg.H, dmrg.allowed_vertical_module_pairs, dmrg.N)
    dmrg.left_envs  = [{pair: None for pair in dmrg.allowed_vertical_module_pairs} for _ in range(dmrg.N)]

    # run sweeps
    from datetime import datetime, timedelta
    total_time = timedelta()
    for _ in range(args.n_sweeps):
        t0 = datetime.now()
        for i in range(dmrg.N - 1):
            dmrg.update_left_environment(i)
            dmrg.optimise_site_pair(i, is_moving_right=True, save_entanglement_spectra=False)
        for i in range(dmrg.N - 2, -1, -1):
            dmrg.update_right_environment(i + 1)
            if int(i) == int(dmrg.N / 2):
                dmrg.optimise_site_pair(i, is_moving_right=False, save_entanglement_spectra=True)
            else: 
                dmrg.optimise_site_pair(i, is_moving_right=False, save_entanglement_spectra=False)
        total_time += (datetime.now() - t0)
    input()
    energy_per_site = dmrg.get_energy_density()
    local_energies = [dmrg.get_local_energy_density(i_bond) for i_bond in range(dmrg.N - 1)]
    sum_local = sum(local_energies)

    print(json.dumps({
        "N": dmrg.N,
        "max_chi_mps": dmrg.max_chi_mps,
        "tolerance_in_S": dmrg.tolerance_in_S,
        "module_name": args.module_name,
        "boundary_modules_right": [list(x) for x in boundary_right],
        "boundary_modules_left":  [list(x) for x in boundary_left],
        "local_energies": clist(local_energies),
        "sum_of_local_energies": cscalar(sum_local),
        "average_of_local_energies": cscalar(sum_local / (dmrg.N - 1)),
        "energy_per_site": cscalar(energy_per_site),
        "total_time": str(total_time),
    }))
    
if __name__ == "__main__":
    main()
