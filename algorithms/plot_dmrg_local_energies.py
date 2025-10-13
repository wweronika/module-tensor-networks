#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

def parse_complex(x):
    # Accept [re, im], {"re":..,"im":..}, or plain number
    if isinstance(x, list) and len(x) == 2:
        return complex(float(x[0]), float(x[1]))
    if isinstance(x, dict) and "re" in x:
        return complex(float(x["re"]), float(x.get("im", 0.0)))
    return complex(float(x), 0.0)

def boundary_short(bc):
    # bc is list of [m,m]; return like 00_11 or 00 if single
    if not isinstance(bc, list):
        return "NA"
    parts = []
    for pair in bc:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            parts.append(f"{int(pair[0])}{int(pair[1])}")
    return "_".join(parts) if parts else "NA"

def load_records(path, module=None, N=None, left=None, right=None):
    # left/right filters are strings like '[[0,0]]' if you want exact matches (optional)
    want_left = json.loads(left) if left else None
    want_right = json.loads(right) if right else None

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error"):
                continue
            if module and rec.get("module_name") != module:
                continue
            if N is not None and rec.get("N") != N:
                continue
            if want_left is not None and rec.get("boundary_modules_left") != want_left:
                continue
            if want_right is not None and rec.get("boundary_modules_right") != want_right:
                continue
            yield rec

def plot_one(rec, outdir, plot_imag=False, show=False):
    mod = rec["module_name"]
    N = rec["N"]
    left = rec.get("boundary_modules_left", [])
    right = rec.get("boundary_modules_right", [])
    locals_raw = rec.get("local_energies", [])

    locals_c = [parse_complex(x) for x in locals_raw]
    xs = list(range(len(locals_c)))
    ys_re = [z.real for z in locals_c]
    ys_im = [z.imag for z in locals_c]

    plt.figure()
    plt.plot(xs, ys_re, marker="o", linewidth=1)
    if plot_imag and any(abs(v) > 1e-12 for v in ys_im):
        plt.plot(xs, ys_im, linestyle="--", marker="x", linewidth=1, label="Imag")
        plt.legend()

    plt.xlabel("pair index (bond #)")
    plt.ylabel("local energy density (Re)")
    title = f"{mod} | N={N} | L={left} | R={right}"
    plt.title(title)

    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{mod}_N{N}_L{boundary_short(left)}_R{boundary_short(right)}.png"
    fpath = outdir / fname
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    if show:
        plt.show()
    plt.close()
    return fpath

def main():
    ap = argparse.ArgumentParser(description="Plot local energy density vs pair index from NDJSON.")
    ap.add_argument("--input", required=True, help="NDJSON file from dmrg sweeps")
    ap.add_argument("--outdir", default="plots_local_energy", help="Folder to save PNGs")
    ap.add_argument("--module", default=None, help="Filter: module_name (optional)")
    ap.add_argument("--N", type=int, default=None, help="Filter: N (optional)")
    ap.add_argument("--left", default=None, help='Filter: exact left BC JSON, e.g. "[[0,0]]"')
    ap.add_argument("--right", default=None, help='Filter: exact right BC JSON, e.g. "[[1,1]]"')
    ap.add_argument("--plot-imag", action="store_true", help="Also plot imaginary part if present")
    ap.add_argument("--show", action="store_true", help="Show plots interactively")
    args = ap.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    count = 0
    for rec in load_records(input_path, module=args.module, N=args.N, left=args.left, right=args.right):
        path = plot_one(rec, outdir, plot_imag=args.plot_imag, show=args.show)
        print(f"saved {path}")
        count += 1

    if count == 0:
        print("no matching records found (did you pass the right filters?)")

if __name__ == "__main__":
    main()
