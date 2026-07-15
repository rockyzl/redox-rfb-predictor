"""
predict_redox.py — SMILES -> aqueous redox potential (V vs SHE) for organic
redox-flow-battery candidates.

Two models are shipped:
  * RDKit-only baseline  (fast, covers full RedDB chemical space; trained on 15,673 molecules)
  * Hybrid RDKit + xTB    (higher physical fidelity; needs an xTB run per molecule)

Usage
-----
    from predict_redox import predict, predict_batch

    predict("O=C1C=CC(=O)C=C1")                     # p-benzoquinone, RDKit-only model
    predict("O=C1C=CC(=O)C=C1", model="hybrid")     # runs xTB on the fly
    predict_batch(["c1ccncc1", "O=C1C=CC(=O)C=C1"])  # list of SMILES

CLI
---
    python predict_redox.py "O=C1C=CC(=O)C=C1"
    python predict_redox.py --model hybrid "O=C1C=CC(=O)C=C1"
    python predict_redox.py --file candidates.smi --out predictions.csv

Requires: rdkit, scikit-learn, joblib, numpy, pandas (+ xtb on PATH for the hybrid model).
"""
import os, sys, re, argparse, tempfile, shutil, subprocess
import numpy as np
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MolToXYZBlock
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: F401 (kept for parity with training)
from rdkit.ML.Descriptors import MoleculeDescriptors
RDLogger.DisableLog("rdApp.*")

HARTREE_EV = 27.211386
# package layout: src/redox_rfb/predictor.py  ->  repo root is two levels up; models/ lives there
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, "..", ".."))
_MODEL_DIR = os.environ.get("REDOX_MODEL_DIR", os.path.join(_REPO_ROOT, "models"))

# --- descriptor set: must match training exactly ---
DESC_NAMES = ['MolWt','ExactMolWt','HeavyAtomCount','NumHAcceptors','NumHDonors',
 'NumRotatableBonds','NumAromaticRings','NumAliphaticRings','RingCount','FractionCSP3',
 'TPSA','MolLogP','MolMR','LabuteASA','BalabanJ','BertzCT',
 'NumHeteroatoms','NumValenceElectrons','qed',
 'NHOHCount','NOCount','NumSaturatedRings','MaxPartialCharge','MinPartialCharge',
 'MaxAbsPartialCharge','MinAbsPartialCharge']
_calc = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)
_fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)

_MODELS = {}

def _load(model):
    if model in _MODELS:
        return _MODELS[model]
    path = os.path.join(_MODEL_DIR, "model_rdkit_baseline.pkl" if model == "rdkit" else "model_hybrid.pkl")
    _MODELS[model] = joblib.load(path)
    return _MODELS[model]


def rdkit_features(mol):
    """26 physicochemical descriptors (+1024-bit Morgan FP for the baseline model)."""
    desc = np.array(_calc.CalcDescriptors(mol), dtype=float)
    fp = _fpgen.GetFingerprintAsNumPy(mol).astype(float)
    return desc, fp


def xtb_features(smiles, threads=2, timeout=150):
    """GFN2-xTB quantum descriptors in ALPB water. Returns dict or None on failure."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = Chem.AddHs(m)
    p = AllChem.ETKDGv3(); p.maxIterations = 200; p.randomSeed = 42
    if AllChem.EmbedMolecule(m, p) != 0:
        p2 = AllChem.ETKDGv3(); p2.useRandomCoords = True; p2.maxIterations = 200
        if AllChem.EmbedMolecule(m, p2) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=300)
    except Exception:
        pass
    d = tempfile.mkdtemp()
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))

    def _sp(xyz, charge, uhf):
        r = subprocess.run(["xtb", xyz, "--gfn", "2", "--alpb", "water",
                            "--chrg", str(charge), "--uhf", str(uhf)],
                           cwd=d, capture_output=True, text=True, timeout=90, env=env)
        mm = re.search(r"TOTAL ENERGY\s+([-\d.]+)\s*Eh", r.stdout)
        return float(mm.group(1)) if mm else None

    try:
        open(os.path.join(d, "mol.xyz"), "w").write(MolToXYZBlock(m))
        r = subprocess.run(["xtb", "mol.xyz", "--gfn", "2", "--opt", "loose", "--alpb", "water"],
                           cwd=d, capture_output=True, text=True, timeout=timeout, env=env)
        out = r.stdout
        f = {}
        g = re.search(r"HOMO-LUMO GAP\s+([-\d.]+)\s*eV", out); f["xtb_gap_eV"] = float(g.group(1)) if g else None
        homo = lumo = None
        for ln in out.splitlines():
            pt = ln.split()
            if "(HOMO)" in ln:
                for i, pp in enumerate(pt):
                    if pp == "(HOMO)": homo = float(pt[i-1])
            if "(LUMO)" in ln:
                for i, pp in enumerate(pt):
                    if pp == "(LUMO)": lumo = float(pt[i-1])
        f["xtb_homo_eV"] = homo; f["xtb_lumo_eV"] = lumo
        e = re.search(r"TOTAL ENERGY\s+([-\d.]+)\s*Eh", out); E0 = float(e.group(1)) if e else None
        f["xtb_energy_neutral_Eh"] = E0
        dp = re.search(r"molecular dipole:[\s\S]*?full:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)", out)
        f["xtb_dipole_D"] = float(dp.group(1)) if dp else None
        optxyz = os.path.join(d, "xtbopt.xyz"); geo = "xtbopt.xyz" if os.path.exists(optxyz) else "mol.xyz"
        try: Ec = _sp(geo, 1, 1)
        except Exception: Ec = None
        try: Ea = _sp(geo, -1, 1)
        except Exception: Ea = None
        if E0 and Ec: f["xtb_vIP_eV"] = (Ec - E0) * HARTREE_EV
        if E0 and Ea: f["xtb_vEA_eV"] = (E0 - Ea) * HARTREE_EV
        return f if homo is not None else None
    except subprocess.TimeoutExpired:
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def predict(smiles, model="rdkit", threads=2):
    """Predict aqueous redox potential (V vs SHE) for one SMILES.

    model='rdkit'  -> fast baseline (descriptors + Morgan FP)
    model='hybrid' -> RDKit descriptors + on-the-fly xTB quantum features
    Returns a float, or raises ValueError on invalid SMILES / xTB failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")

    if model == "rdkit":
        bundle = _load("rdkit")
        desc, fp = rdkit_features(mol)
        X = np.concatenate([desc, fp]).reshape(1, -1)
        return float(bundle["model"].predict(X)[0])

    elif model == "hybrid":
        bundle = _load("hybrid")
        desc, _ = rdkit_features(mol)
        xf = xtb_features(smiles, threads=threads)
        if xf is None:
            raise ValueError(f"xTB featurization failed for {smiles!r}; try model='rdkit'")
        # assemble in the exact training column order
        desc_map = dict(zip([f"rdkit_{n}" for n in DESC_NAMES], desc))
        row = []
        for c in bundle["feature_cols"]:
            row.append(desc_map[c] if c.startswith("rdkit_") else xf.get(c, np.nan))
        X = np.array(row, dtype=float).reshape(1, -1)
        if np.isnan(X).any():
            raise ValueError(f"incomplete xTB features for {smiles!r}")
        return float(bundle["model"].predict(X)[0])

    raise ValueError("model must be 'rdkit' or 'hybrid'")


def predict_batch(smiles_list, model="rdkit", threads=2):
    """Predict for a list of SMILES. Returns list of (smiles, potential_or_None)."""
    out = []
    for s in smiles_list:
        try:
            out.append((s, predict(s, model=model, threads=threads)))
        except Exception as e:
            out.append((s, None))
    return out


def _main():
    ap = argparse.ArgumentParser(description="SMILES -> redox potential (V vs SHE)")
    ap.add_argument("smiles", nargs="*", help="one or more SMILES strings")
    ap.add_argument("--model", choices=["rdkit", "hybrid"], default="rdkit")
    ap.add_argument("--file", help="text file with one SMILES per line")
    ap.add_argument("--out", help="write predictions to this CSV")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()

    smis = list(a.smiles)
    if a.file:
        with open(a.file) as fh:
            smis += [ln.strip().split()[0] for ln in fh if ln.strip() and not ln.startswith("#")]
    if not smis:
        ap.error("provide SMILES on the command line or via --file")

    res = predict_batch(smis, model=a.model, threads=a.threads)
    lines = ["smiles,redox_potential_V,model"]
    for s, v in res:
        val = f"{v:.4f}" if v is not None else "NA"
        lines.append(f"{s},{val},{a.model}")
        print(f"{s}\t{val} V" + ("" if v is not None else "  (failed)"))
    if a.out:
        open(a.out, "w").write("\n".join(lines) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    _main()
