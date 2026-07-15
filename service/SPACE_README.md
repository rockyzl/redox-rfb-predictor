---
title: Redox RFB Predictor API
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Redox RFB Predictor API

Public API behind the SciencesLoop redox-potential demo.

- `GET /healthz`
- `POST /v1/predict` with `{"smiles": "O=C1C=CC(=O)C=C1"}`

It runs the fast RDKit-descriptor + Morgan-fingerprint model trained on 15,673
RedDB molecules. The output is a prediction on RedDB's DFT-derived aqueous
redox-potential scale (V vs SHE), **not** an experimental measurement.

The model is mainly applicable to the quinone and aza-aromatic chemistry in
RedDB. Inputs outside that chemical domain are extrapolations.
