# Public service

This directory packages the fast RDKit + Morgan-fingerprint predictor as a
small FastAPI service for the SciencesLoop demo. The public endpoint deliberately
does not expose the xTB hybrid path: xTB takes seconds per request and needs a
separate controlled queue before it is suitable for an anonymous public service.

Contract:

```text
GET  /healthz
POST /v1/predict  {"smiles":"O=C1C=CC(=O)C=C1"}
```

The service accepts only a SMILES string, stores no input, validates it with
RDKit, applies a small in-memory per-client rate limit, and permits browser
requests only from SciencesLoop origins by default.
