#!/usr/bin/env python
"""Compute RDKit descriptors + 1024-bit Morgan fingerprints for all curated molecules.
Output: data/features_rdkit.parquet
"""
import os
import numpy as np, pandas as pd
from rdkit.Chem import MolFromSmiles, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(__file__)
IN = os.path.join(HERE, "..", "data", "reddb_curated.parquet")
OUT = os.path.join(HERE, "..", "data", "features_rdkit.parquet")

DESC_NAMES = ['MolWt','ExactMolWt','HeavyAtomCount','NumHAcceptors','NumHDonors',
 'NumRotatableBonds','NumAromaticRings','NumAliphaticRings','RingCount','FractionCSP3',
 'TPSA','MolLogP','MolMR','LabuteASA','BalabanJ','BertzCT','NumHeteroatoms',
 'NumValenceElectrons','qed','NHOHCount','NOCount','NumSaturatedRings',
 'MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge','MinAbsPartialCharge']

def main():
    df = pd.read_parquet(IN)
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)
    fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
    mols = [MolFromSmiles(s) for s in df["canonical_smiles"]]
    desc = pd.DataFrame([calc.CalcDescriptors(m) for m in mols],
                        columns=[f"rdkit_{n}" for n in DESC_NAMES])
    fp = pd.DataFrame(np.array([fpgen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8),
                      columns=[f"fp_{i}" for i in range(1024)])
    out = pd.concat([df[["canonical_smiles", "redox_potential_V"]].reset_index(drop=True), desc, fp], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=list(desc.columns)).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"RDKit features {out.shape} -> {OUT}")

if __name__ == "__main__":
    main()
