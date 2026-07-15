# redox-rfb-predictor

Hybrid **xTB → machine-learning** prediction of aqueous **redox potential**
(V vs SHE) for organic redox-flow-battery (RFB) molecules — quinones and
aza-aromatics, the two dominant aqueous-RFB chemistries.

Given a SMILES string, the pipeline returns a predicted redox potential. Two
models ship trained and ready:

| Model | Features | Training set | Speed | Use when |
|-------|----------|--------------|-------|----------|
| **RDKit-only baseline** | 26 descriptors + 1024-bit Morgan FP | 15,673 molecules (full RedDB) | ~1 ms/mol | high-throughput screening |
| **Hybrid RDKit + xTB** | 26 RDKit descriptors + 7 GFN2-xTB quantum descriptors | 129 molecules (diverse subset) | ~10–60 s/mol | physically-grounded shortlist ranking |

## Quickstart

```bash
conda env create -f environment.yml && conda activate redox
pip install -e .

python examples/quickstart.py
predict-redox "O=C1C=CC(=O)C=C1"                 # 0.185 V vs SHE (RDKit-only)
predict-redox --model hybrid "O=C1C=CC(=O)C=C1"  # runs xTB
predict-redox --file examples/candidates.smi --out preds.csv
```

### Local web app

After creating the environment and installing the package, start a small local
web interface:

```bash
streamlit run app.py
```

Then open the local address Streamlit prints (normally `http://localhost:8501`).
The page accepts a SMILES string and runs either shipped predictor.

```python
from redox_rfb import predict, predict_batch
predict("O=C1C=CC(=O)C=C1")                  # -> 0.185
predict("O=C1C=CC(=O)C=C1", model="hybrid")  # -> 0.233
predict_batch(["c1ccncc1", "O=C1c2ccccc2C(=O)c2ccccc21"])
```

## Repository layout

```
redox-rfb-predictor/
├── src/redox_rfb/         importable package
│   ├── __init__.py
│   └── predictor.py       predict(), predict_batch(), featurizers, CLI
├── scripts/               end-to-end pipeline (run in order)
│   ├── 00_download_reddb.py    fetch RedDB from Harvard Dataverse
│   ├── 01_curate.py           label + sanitize -> data/reddb_curated.parquet
│   ├── 02_featurize_rdkit.py  RDKit descriptors + Morgan FP
│   ├── 03_featurize_xtb.py    GFN2-xTB quantum descriptors (parallel, checkpointed)
│   ├── 04_train_evaluate.py   train both models + CV comparison
│   └── 05_make_figures.py     evaluation figures
├── models/                trained pickles (xz-compressed)
│   ├── model_rdkit_baseline.pkl
│   └── model_hybrid.pkl
├── data/                  curated tables + feature matrices (parquet)
├── reports/               figures + metric/importance CSVs
├── examples/              quickstart.py, candidates.smi
├── environment.yml  requirements.txt  pyproject.toml  LICENSE
```

## The result: xTB features add real, physically-grounded value

On a common 129-molecule test set (repeated 5-fold × 6 cross-validation):

| Feature set | CV MAE (V) | CV R² |
|-------------|-----------|-------|
| xTB only (7) | 0.203 | 0.29 |
| RDKit descriptors (26) | 0.204 | 0.37 |
| RDKit desc + FP (1050) | 0.194 | 0.43 |
| **RDKit desc + xTB (33)** | **0.173** | **0.50** |
| RDKit desc + FP + xTB | 0.172 | 0.51 |

Adding xTB features to RDKit descriptors **raises R² from 0.37 → 0.50 and cuts
MAE 15 %**. The three most important features in the hybrid model are all xTB
descriptors — vertical electron affinity, HOMO–LUMO gap, LUMO energy — the
electronic quantities that physically govern reduction potential. xTB features
carry **57 %** of total model importance despite being only 7 of 33 inputs.

The RDKit-only baseline reaches R² 0.97 / MAE 0.045 V on the **full** 15,673-molecule
set (an easier test — many molecules per scaffold, wider potential range). See
`reports/evaluation_report.png`.

## How the label is defined

The label is derived from RedDB's DFT reaction energy for the balanced
2-electron / 2-proton redox reaction, referenced to the standard hydrogen
electrode via ΔG = −nFE°:

```
E° (V vs SHE) = − reaction_energy [Hartree] × 27.2114 / n     (n = 2)
```

Yielding 15,673 unique molecules with potentials centered at 0.31 V
(90 % within [−0.46, +1.15] V — the expected aqueous window).

## Reproduce from scratch

```bash
python scripts/00_download_reddb.py     # ~86 MB from Harvard Dataverse
python scripts/01_curate.py
python scripts/02_featurize_rdkit.py
python scripts/03_featurize_xtb.py      # compute-bound; ~1 h on 8 cores for 350 mols
python scripts/04_train_evaluate.py
python scripts/05_make_figures.py
```

## Data

**RedDB** — Sengul et al., *Scientific Data* 2022
([10.1038/s41597-022-01832-2](https://doi.org/10.1038/s41597-022-01832-2);
data DOI [10.7910/DVN/F3QFSQ](https://doi.org/10.7910/DVN/F3QFSQ)), a
computational database of 31,618 electroactive molecules for aqueous RFBs.

## Limitations & scope

- **Computational reference frame.** Predictions are on RedDB's DFT potential
  scale — reliable for *ranking* candidates, but apply an experimental offset
  before comparing to measured half-cell potentials.
- **Chemical domain.** Trained on RedDB's quinone/aza-aromatic scaffolds;
  chemotypes outside this space (viologens, TEMPO radicals, metal complexes)
  are extrapolation.
- **Hybrid model size.** Trained on 129 molecules because xTB is compute-bound
  (~46 % of attempts converged within the per-molecule deadline; open-shell
  cation/anion SCF is the main failure). It demonstrates the quantum-feature
  uplift; for production, featurize a larger subset on a cluster.

## License

MIT (code). RedDB data redistributed under its original CC0 terms.
