#!/usr/bin/env python3
import argparse, json, os, subprocess, sys
from pathlib import Path

# ----- fixed params you said not to vary -----
MAX_CHI_MPS = 15
TOLERANCE_IN_S = 1e-4
# --------------------------------------------

# module_name -> number of simple objects
NUMBER_OF_SIMPLE_OBJECTS = {
    "Vec": 1,
    "RepPsiA4": 3,
    "RepA4": 4,
    "RepPsiD2": 1,
    "RepD2": 4,
    "RepZ2_1": 2, "RepZ2_2": 2, "RepZ2_3": 2,
    "RepZ3_1": 3, "RepZ3_2": 3, "RepZ3_3": 3, "RepZ3_4": 3,
}
DEFAULT_MODULES = list(NUMBER_OF_SIMPLE_OBJECTS.keys())

def find_project_root(start: Path) -> Path:
    """Return a directory that contains 'algorithms/'."""
    start = start.resolve()
    if (start / "algorithms").is_dir(): return start
    if start.name == "algorithms":      return start.parent
    for p in start.parents:
        if (p / "algorithms").is_dir(): return p
    return start.parent

def run_one(python_bin, env, N, module, left_json, right_json, n_sweeps, out_fh):
    """Run one dmrg_cli invocation and append its JSON (or error JSON) to out_fh."""
    cmd = [
        python_bin, "-m", "algorithms.dmrg_cli",
        "--N", str(N),
        "--max_chi_mps", str(MAX_CHI_MPS),
        "--tolerance_in_S", str(TOLERANCE_IN_S),
        "--module_name", module,
        "--boundary_modules_left", left_json,
        "--boundary_modules_right", right_json,
        "--n_sweeps", str(n_sweeps),
    ]
    print(f"Running N={N} module={module} left={left_json} right={right_json}",
          file=sys.stderr, flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode == 0:
        line = p.stdout.strip()
        try:
            obj = json.loads(line)
            out_fh.write(json.dumps(obj) + "\n"); out_fh.flush()
        except Exception as e:
            out_fh.write(json.dumps({
                "error": True, "reason": "Invalid JSON from dmrg_cli",
                "exception": str(e), "stdout": line, "stderr": p.stderr,
                "params": {
                    "N": N, "module_name": module,
                    "boundary_modules_left": json.loads(left_json),
                    "boundary_modules_right": json.loads(right_json),
                    "n_sweeps": n_sweeps, "max_chi_mps": MAX_CHI_MPS,
                    "tolerance_in_S": TOLERANCE_IN_S,
                }
            }) + "\n"); out_fh.flush()
    else:
        out_fh.write(json.dumps({
            "error": True, "reason": "dmrg_cli returned nonzero exit code",
            "returncode": p.returncode, "stderr": p.stderr, "stdout": p.stdout,
            "params": {
                "N": N, "module_name": module,
                "boundary_modules_left": json.loads(left_json),
                "boundary_modules_right": json.loads(right_json),
                "n_sweeps": n_sweeps, "max_chi_mps": MAX_CHI_MPS,
                "tolerance_in_S": TOLERANCE_IN_S,
            }
        }) + "\n"); out_fh.flush()

def main():
    ap = argparse.ArgumentParser(
        description="Sweep BCs with left=[[0,0]] and right=[[i,i]]; write NDJSON."
    )
    ap.add_argument("--python-bin", default="python", help="Python executable")
    ap.add_argument("--n-min", type=int, default=20, help="Min N (inclusive)")
    ap.add_argument("--n-max", type=int, default=30, help="Max N (inclusive)")
    ap.add_argument("--n-sweeps", type=int, default=1, help="Number of sweeps")
    ap.add_argument("--modules", nargs="*", default=DEFAULT_MODULES,
                    help="Which module names to run (defaults to all)")
    ap.add_argument("--out", default="results_bcsweep.ndjson", help="Output NDJSON path")
    args = ap.parse_args()

    # ensure 'algorithms' is importable for the subprocess
    here = Path(__file__).resolve().parent
    root = find_project_root(here)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (os.pathsep + env.get("PYTHONPATH", "")
                                     if env.get("PYTHONPATH") else "")

    data_dir = root / "input" / "mpoHam_A4"
    Ns = list(range(args.n_min, args.n_max + 1))
    left_json = json.dumps([[0, 0],[1, 1],[2, 2]])  # <- fixed left BC

    with open(args.out, "w", encoding="utf-8") as out_fh:
        for N in Ns:
            for module in args.modules:
                if module not in NUMBER_OF_SIMPLE_OBJECTS:
                    print(f"Skipping unknown module '{module}'",
                          file=sys.stderr, flush=True)
                    continue

                # quick precheck to avoid spam if files are missing
                stem = module
                needed = [data_dir / f"{stem}_ind.txt",
                          data_dir / f"{stem}_converted_var.txt",
                          data_dir / f"{stem}_size.txt"]
                missing = [str(p) for p in needed if not p.exists()]
                if missing:
                    out_fh.write(json.dumps({
                        "error": True, "reason": "missing_data_files",
                        "module_name": module, "missing": missing
                    }) + "\n"); out_fh.flush()
                    continue

                K = NUMBER_OF_SIMPLE_OBJECTS[module]
                for i in range(K):
                    right_json = json.dumps([[i, i]])  # <- sweep right BC singleton
                    run_one(args.python_bin, env, N, module,
                            left_json, right_json, args.n_sweeps, out_fh)

    print(f"Wrote NDJSON to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
