"""Inspect dataset fields and a sample example."""

import os
from datasets import load_from_disk

DATASET_PATH = os.getenv("DATASET_PATH", "fuck")

print(f"Loading dataset from: {DATASET_PATH}")
ds = load_from_disk(DATASET_PATH)

print(f"\nDataset size: {len(ds):,}")
print(f"Column names: {ds.column_names}")
print(f"Features: {ds.features}")

print(f"\nFirst example:")
ex = ds[0]
for key, val in ex.items():
    if isinstance(val, list):
        print(f"  {key}: list[{len(val)}] = {val[:5]}{'...' if len(val) > 5 else ''}")
    else:
        print(f"  {key}: {val}")
