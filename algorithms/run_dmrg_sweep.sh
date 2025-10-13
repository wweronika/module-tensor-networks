#!/usr/bin/env bash
set -euo pipefail

# --- config you can tweak ---
# PYTHON_BIN=${PYTHON_BIN:-python3}
# DMRG_CLI=${DMRG_CLI:-./dmrg_cli.py}
N_SWEEPS=${N_SWEEPS:-1}
MAX_CHI_MPS=15
TOL_S=1e-4
# ----------------------------

# Map: module_name -> number_of_simple_objects
declare -A NSO=(
  [Vec]=1
  [RepPsiA4]=3
  [RepA4]=4
  [RepPsiD2]=1
  [RepD2]=4
  [Rep_Z2_1]=2
  [Rep_Z2_2]=2
  [Rep_Z2_3]=2
  [Rep_Z3_1]=3
  [Rep_Z3_2]=3
  [Rep_Z3_3]=3
  [Rep_Z3_4]=3
)

MODULES=(
  Vec RepPsiA4 RepA4 RepPsiD2 RepD2
  Rep_Z2_1 Rep_Z2_2 Rep_Z2_3
  Rep_Z3_1 Rep_Z3_2 Rep_Z3_3 Rep_Z3_4
)

# Generate all non-empty subsets of pairs (m,m) for m=0..K-1
# Output: prints each subset as a single JSON string like [[0,0],[2,2]]
gen_nonempty_boundary_subsets_json() {
  local K="$1"
  local total=$((1<<K))
  local mask m first subset

  for ((mask=1; mask<total; mask++)); do
    subset="["
    first=1
    for ((m=0; m<K; m++)); do
      if (( (mask>>m) & 1 )); then
        if (( first )); then
          subset+="[$m,$m]"
          first=0
        else
          subset+=",[$m,$m]"
        fi
      fi
    done
    subset+="]"
    echo "$subset"
  done
}

# Loop over N, modules, and all boundary subsets (left x right)
for N in $(seq 20 30); do
  for MODULE in "${MODULES[@]}"; do
    K="${NSO[$MODULE]}"

    # precompute arrays of subset JSON strings
    mapfile -t LEFTS  < <(gen_nonempty_boundary_subsets_json "$K")
    mapfile -t RIGHTS < <(gen_nonempty_boundary_subsets_json "$K")

    for LEFT in "${LEFTS[@]}"; do
      for RIGHT in "${RIGHTS[@]}"; do
        # stderr status line (human-readable)
        echo "Running N=$N module=$MODULE left=$LEFT right=$RIGHT" >&2

        # Run and emit NDJSON (one JSON object per line) to stdout
        # "$PYTHON_BIN" "$DMRG_CLI" \
        python -m algorithms.dmrg_cli \
          --N "$N" \
          --max_chi_mps "$MAX_CHI_MPS" \
          --tolerance_in_S "$TOL_S" \
          --module_name "$MODULE" \
          --boundary_modules_left "$LEFT" \
          --boundary_modules_right "$RIGHT" \
          --n_sweeps "$N_SWEEPS"
      done
    done
  done
done
