"""Print the results table: mean chrF++ (and seeds) for each language / size / method."""
import csv
from collections import defaultdict
from statistics import mean, pstdev

rows = list(csv.DictReader(open("results.csv")))
groups = defaultdict(list)
for r in rows:
    parts = r["name"].split("_")          # e.g. asm_1000_jepa_s1  or  asm_zeroshot
    key = "_".join(parts[:-1]) if parts[-1].startswith("s") and parts[-1][1:].isdigit() else r["name"]
    groups[key].append(float(r["chrf++"]))
print(f"{'run':38s} {'chrF++ mean':>12s} {'sd':>6s}  seeds")
for k in sorted(groups):
    v = groups[k]
    print(f"{k:38s} {mean(v):12.2f} {pstdev(v):6.2f}  {len(v)}")
