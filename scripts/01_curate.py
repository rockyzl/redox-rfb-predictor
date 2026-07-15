#!/usr/bin/env python
"""Curate RedDB into a clean training table.

Label: E (V vs SHE) = -reaction_energy[Ha] * 27.2114 / n,  n = 2 (2e-/2H+ couple).
Sanitize SMILES with RDKit, deduplicate on canonical SMILES, filter to +/-3 V.
Output: data/reddb_curated.parquet
"""
import os
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MolToSmiles, MolFromSmiles
RDLogger.DisableLog("rdApp.*")

HARTREE_EV = 27.211386
N_ELECTRONS = 2
HERE = os.path.dirname(__file__)
RXN = os.path.join(HERE, "..", "data", "raw", "RedDBv2_reaction.tab")
OUT = os.path.join(HERE, "..", "data", "reddb_curated.parquet")

def canon(s):
    m = MolFromSmiles(s)
    return MolToSmiles(m) if m else None

def main():
    rxn = pd.read_csv(RXN, sep="\t")
    rxn["redox_potential_V"] = -rxn["reaction_energy"] * HARTREE_EV / N_ELECTRONS
    w = rxn[["reactant_smiles", "reaction_energy", "redox_potential_V",
             "reactant_homo", "reactant_lumo", "reactant_solubility"]].copy()
    w["canonical_smiles"] = w["reactant_smiles"].apply(canon)
    w = w.dropna(subset=["canonical_smiles"])
    w["homo_eV"] = w["reactant_homo"] * HARTREE_EV
    w["lumo_eV"] = w["reactant_lumo"] * HARTREE_EV
    w["gap_eV"] = w["lumo_eV"] - w["homo_eV"]
    w = w.rename(columns={"reactant_solubility": "solubility_logS"})
    agg = w.groupby("canonical_smiles").agg(
        redox_potential_V=("redox_potential_V", "mean"),
        redox_potential_std=("redox_potential_V", "std"),
        n_reactions=("redox_potential_V", "size"),
        homo_eV=("homo_eV", "mean"), lumo_eV=("lumo_eV", "mean"), gap_eV=("gap_eV", "mean"),
        solubility_logS=("solubility_logS", "mean")).reset_index()
    agg["redox_potential_std"] = agg["redox_potential_std"].fillna(0.0)
    agg = agg[agg["redox_potential_V"].between(-3, 3)].reset_index(drop=True)
    agg["qc_missing"] = (agg["homo_eV"] == 0) | (agg["lumo_eV"] == 0)
    agg.to_parquet(OUT, index=False)
    print(f"curated {len(agg)} unique molecules -> {OUT}")

if __name__ == "__main__":
    main()
