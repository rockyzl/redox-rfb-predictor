# Public service

This directory packages the fast RDKit + Morgan-fingerprint predictor as a
small FastAPI service for the SciencesLoop demo. The public endpoint deliberately
does not expose the xTB hybrid path: xTB takes seconds per request and needs a
separate controlled queue before it is suitable for an anonymous public service.

Contract:

```text
GET  /healthz
POST /v1/predict  {"smiles":"O=C1C=CC(=O)C=C1"}
POST /v1/resolve-name  {"name":"p-benzoquinone"}
POST /v1/preview-structure  {"smiles":"O=C1C=CC(=O)C=C1"}
```

The service accepts only a SMILES string, stores no input, validates it with
RDKit, applies a small in-memory per-client rate limit, and permits browser
requests only from SciencesLoop origins by default.

`/v1/resolve-name` uses a server-side LLM key to convert a common or IUPAC name
to a candidate SMILES, then validates that SMILES with RDKit before returning it.
It is rate limited more tightly than prediction, and is an input convenience —
not a source of chemical identity truth.

Both structure routes return RDKit's canonical SMILES and an RDKit-rendered 2D
SVG. The public page requires a user to confirm that rendered structure before
calling the prediction endpoint.
