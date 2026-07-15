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
POST /v1/suggest-name  {"name":"tempo"}
POST /v1/structure-chat  {"current_smiles":"...", "instruction":"make the oxoammonium form"}
```

The service accepts only a SMILES string, stores no input, validates it with
RDKit, applies a small in-memory per-client rate limit, and permits browser
requests only from SciencesLoop origins by default.

`/v1/resolve-name` first queries PubChem PUG-REST by name and validates the
returned SMILES with RDKit. Only when PubChem does not resolve a name does it
use the server-side LLM to propose up to three labeled candidates, which RDKit
then validates. The response says which source was used. Users can select a
candidate or edit its SMILES and re-render it before predicting. This is still
an input convenience, not a substitute for chemical identity review.

Both structure routes return RDKit's canonical SMILES and an RDKit-rendered 2D
SVG. The public page requires a user to confirm that rendered structure before
calling the prediction endpoint.

The structure-chat endpoint can propose up to three edits to an already
confirmed structure. Each proposed SMILES is parsed and re-rendered by RDKit;
the user must select or edit a candidate and confirm it before prediction.
