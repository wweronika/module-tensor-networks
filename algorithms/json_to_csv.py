import json
import csv

input_file = "n_40_to_45_module_RepPsiA4_n_sweeps_3_results_2.json"   # your input file, one JSON per line
output_file = "n_40_to_45_module_RepPsiA4_n_sweeps_3_results_2.csv"  # your desired output CSV

with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    writer = None

    for line in infile:
        if not line.strip():
            continue  # skip empty lines
        obj = json.loads(line)
        # remove large list
        obj.pop("local_energies", None)

        # Flatten nested dicts (e.g., sum_of_local_energies)
        flat = {}
        for k, v in obj.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    flat[f"{k}.{subk}"] = subv
            else:
                flat[k] = v

        # Initialize CSV writer with headers
        if writer is None:
            writer = csv.DictWriter(outfile, fieldnames=list(flat.keys()))
            writer.writeheader()

        writer.writerow(flat)

print(f"✅ Done! Wrote CSV to {output_file}")
