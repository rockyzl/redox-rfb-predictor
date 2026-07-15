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
- `POST /v1/resolve-name` with `{"name": "p-benzoquinone"}`
- `POST /v1/preview-structure` with `{"smiles": "O=C1C=CC(=O)C=C1"}`
- `POST /v1/suggest-name` with `{"name": "tempo"}`
- `POST /v1/structure-chat` with a confirmed SMILES and a requested edit

It runs the fast RDKit-descriptor + Morgan-fingerprint model trained on 15,673
RedDB molecules. The output is a prediction on RedDB's DFT-derived aqueous
redox-potential scale (V vs SHE), **not** an experimental measurement.

The model is mainly applicable to the quinone and aza-aromatic chemistry in
RedDB. Inputs outside that chemical domain are extrapolations.

The name resolver first queries PubChem PUG-REST, then validates the returned
SMILES with RDKit. It calls the server-side LLM only as a fallback when PubChem
does not resolve a name, returning up to three labeled candidates and marking
that fallback in the response. Users can select or edit a candidate before
prediction. It is a convenience for entering a molecule, not a substitute for
chemical identity review.

The browser displays an RDKit-rendered 2D structure and canonical SMILES for
confirmation before it requests a prediction.
