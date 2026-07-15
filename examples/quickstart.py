#!/usr/bin/env python
"""Quickstart: predict redox potentials for a few RFB candidate molecules.

    python examples/quickstart.py
"""
from redox_rfb import predict, predict_batch

# single molecule, fast RDKit-only model
print("p-benzoquinone (RDKit-only):", round(predict("O=C1C=CC(=O)C=C1"), 3), "V vs SHE")

# same molecule with the hybrid model (runs xTB — needs xtb on PATH)
try:
    print("p-benzoquinone (hybrid)   :", round(predict("O=C1C=CC(=O)C=C1", model="hybrid"), 3), "V vs SHE")
except Exception as e:
    print("hybrid model skipped:", e)

# batch screen
cands = ["c1ccncc1", "O=C1c2ccccc2C(=O)c2ccccc21", "O=C1C=CC(=O)c2ccccc21"]
print("\nbatch (RDKit-only):")
for smi, v in predict_batch(cands):
    print(f"  {smi:35s} {v:.3f} V" if v is not None else f"  {smi:35s} FAILED")
