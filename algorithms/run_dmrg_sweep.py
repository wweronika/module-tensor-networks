#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

# ------- fixed parameters -------
MAX_CHI_MPS = 15
TOLERANCE_IN_S = 1e-4
# --------------------------------

NUMBER_OF_SIMPLE_OBJECTS = {
    "Vec": 1,
    "RepPsiA4": 3,
    "RepA4": 4,
    "RepPsiD2": 1,
    "RepD2": 4,
    "Rep_Z2_1": 2,
    "Rep_Z2_2": 2,
    "Rep_Z2_3": 2,
    "Rep_Z3_1": 3,
    "Rep_Z3_2": 3,
    "Rep_Z3_3": 3,
    "Rep_Z3_4": 3,
}
DEFAULT_MODULES = list(NUMBER_OF_SIMPLE_OBJECTS.keys())

def find_project_root_for_algorithms(start: Path) -> Path:
    """
    Return a path that should be on PYTHONPATH such that 'import algorithms' works.
    Tries:
      - if start/algorithms exists -> return start
      - if start is .../algorithms -> return parent
      - else walk up parents to find a dir containing 'algorithms'
    """
    start = start.resolve()
    if (start / "algorithms").is_dir():
        return start
    if start.name == "algorithms":
        return start.parent
    for p in start.parents:
        if (p / "algorithms").is_dir():
            return p
    # Fallback: assume start’s parent (better than nothing)
    return start.parent

def gen_nonempty_boundary_subsets_json(K: int):
    """Yield JSON strings like [[0,0],[2,2]] for all non-empty subsets of [(m,m)] m=0..K-1."""
    items = [(m, m) for m in range(K)]
    for r in range(1, K + 1):
        for combo in itertools.combinations(items, r):
            yield json.dumps(combo)

def run_one(python_bin, env, N, module, left_json, right_json, n_sweeps, out_fh):
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
    print(f"Running N={N} module={module} left={left_json} right={right_json}", file=sys.stderr, flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode == 0:
        line = proc.stdout.strip()
        try:
            obj = json.loads(line)
            out_fh.write(json.dumps(obj) + "\n")
            out_fh.flush()
        except Exception as e:
            err = {
                "error": True, "reason": "Invalid JSON from dmrg_cli", "exception": str(e),
                "stdout": line, "stderr": proc.stderr,
                "params": {
                    "N": N, "module_name": module,
                    "boundary_modules_left": json.loads(left_json),
                    "boundary_modules_right": json.loads(right_json),
                    "n_sweeps": n_sweeps, "max_chi_mps": MAX_CHI_MPS, "tolerance_in_S": TOLERANCE_IN_S,
                },
            }
            out_fh.write(json.dumps(err) + "\n"); out_fh.flush()
    else:
        err = {
            "error": True, "reason": "dmrg_cli returned nonzero exit code",
            "returncode": proc.returncode, "stderr": proc.stderr, "stdout": proc.stdout,
            "params": {
                "N": N, "module_name": module,
                "boundary_modules_left": json.loads(left_json),
                "boundary_modules_right": json.loads(right_json),
                "n_sweeps": n_sweeps, "max_chi_mps": MAX_CHI_MPS, "tolerance_in_S": TOLERANCE_IN_S,
            },
        }
        out_fh.write(json.dumps(err) + "\n"); out_fh.flush()

def main():
    ap = argparse.ArgumentParser(description="Run python -m algorithms.dmrg_cli over a sweep and write NDJSON.")
    ap.add_argument("--python-bin", default="python", help="Python executable")
    ap.add_argument("--n-min", type=int, default=20)
    ap.add_argument("--n-max", type=int, default=30)
    ap.add_argument("--n-sweeps", type=int, default=1)
    ap.add_argument("--modules", nargs="*", default=DEFAULT_MODULES)
    ap.add_argument("--out", default="results.ndjson")
    args = ap.parse_args()

    # Ensure 'algorithms' is importable when we spawn the subprocess:
    here = Path(__file__).resolve().parent
    project_root = find_project_root_for_algorithms(here)
    env = os.environ.copy()
    env["PYTHONPATH"] = (str(project_root)
                         + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env and env["PYTHONPATH"] else ""))

    # Nice sanity tip if package file is missing:
    if not (project_root / "algorithms" / "__init__.py").exists():
        print(
            f"[hint] No __init__.py found in {project_root / 'algorithms'}. "
            f"Add an empty __init__.py so 'algorithms' is a proper package.",
            file=sys.stderr,
        )

    Ns = list(range(args.n_min, args.n_max + 1))
    with open(args.out, "w", encoding="utf-8") as out_fh:
        for N in Ns:
            for module in args.modules:
                K = NUMBER_OF_SIMPLE_OBJECTS.get(module)
                if K is None:
                    print(f"Skipping unknown module '{module}'", file=sys.stderr)
                    continue
                lefts  = list(gen_nonempty_boundary_subsets_json(K))
                rights = list(gen_nonempty_boundary_subsets_json(K))
                for left_json in lefts:
                    for right_json in rights:
                        run_one(args.python_bin, env, N, module, left_json, right_json, args.n_sweeps, out_fh)

    print(f"Wrote NDJSON to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
