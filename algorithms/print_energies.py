#!/usr/bin/env python3
import json
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Parse JSON lines and print average local energy and energy per site."
    )
    parser.add_argument("--input", "-i", required=True, help="Input file with one JSON per line")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            N = data["N"]
            avg_energy = data["average_of_local_energies"]["re"]
            energy_per_site = data["energy_per_site"]["re"]

            print(f"{N}\t{avg_energy:.10f}\t{energy_per_site:.10f}")

if __name__ == "__main__":
    main()
