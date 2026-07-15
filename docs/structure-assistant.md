# Structure assistant: name → structure → prediction

The live demo is intentionally a **structure-confirmation workflow**, not a
black-box “type a name and trust the number” tool. A redox-potential prediction
is only requested after the user sees and confirms a molecular structure.

Live page: <https://sciencesloop.com/agent/redox-rfb-predictor/>

## User flow

```text
chemical name or SMILES
        |
        |-- SMILES: RDKit parses, canonicalizes, and renders it
        |
        `-- name: PubChem suggests possible standardized names
                    |
                    `-- user selects a name
                            |
                            `-- PubChem returns a structure when available
                                (LLM fallback only when PubChem has no match)
                                      |
                                      v
                         RDKit canonical SMILES + 2D drawing
                                      |
                         user may edit SMILES, redraw, or ask for
                         up to three structure-edit candidates
                                      |
                                      v
                            user confirms one structure
                                      |
                                      v
              fixed Random Forest redox-potential predictor
```

## Why the steps are separate

A syntactically valid SMILES does **not** prove that it represents the molecule
the user meant. For example, `TEMPO` should resolve to the nitroxide radical:

```text
CC1(C)CCCC(C)(C)N1[O]
```

RDKit can verify that a SMILES is chemically parseable and can make its 2D
drawing, but it cannot decide whether a guessed structure is the intended
identity. PubChem name lookup is therefore the first choice. The user sees the
resulting structure, can edit its SMILES, and must explicitly confirm it before
the predictor runs.

## What each component does

| Component | Role | Does not do |
| --- | --- | --- |
| PubChem | Suggests names and looks up a matching structure by name | Predict redox potential or guarantee the user's intended identity |
| LLM structure assistant | Suggests spelling/name alternatives when PubChem cannot; proposes up to three requested structural edits | Predict potential, replace PubChem, or override user review |
| RDKit | Parses submitted SMILES, canonicalizes it, and draws the displayed 2D structure | Establish semantic identity from an ambiguous name |
| Random Forest predictor | Estimates redox potential from the confirmed molecule | Run DFT or report an experimental measurement |

All LLM-produced structure candidates are parsed and rendered by RDKit before
they are shown. If RDKit cannot parse one, the service discards it. The user may
always type a SMILES directly and use **Render edited structure** to replace a
candidate.

## API contract

The public service is at
`https://rockyaaos-redox-rfb-predictor.hf.space`.

| Endpoint | Input | Purpose |
| --- | --- | --- |
| `POST /v1/suggest-name` | `{"name":"tempo"}` | Return up to five possible PubChem name matches; use an LLM spelling fallback only when none exist. |
| `POST /v1/resolve-name` | `{"name":"TEMPO"}` | Retrieve a PubChem structure first, returning its RDKit-validated canonical SMILES and SVG. |
| `POST /v1/preview-structure` | `{"smiles":"..."}` | Validate, canonicalize, and draw a user-provided or edited SMILES. |
| `POST /v1/structure-chat` | `{"current_smiles":"...","instruction":"..."}` | Propose up to three RDKit-validated editable structure changes. |
| `POST /v1/predict` | `{"smiles":"..."}` | Estimate potential for the user-confirmed structure. |

The API rate-limits anonymous requests in memory and does not persist submitted
names or SMILES. Browser access is restricted to SciencesLoop origins by
default.

## Prediction model and interpretation

The online prediction path is the fast fixed baseline:

```text
confirmed canonical SMILES
  → 26 RDKit descriptors + 1,024-bit Morgan fingerprint
  → Random Forest (300 trees)
  → RedDB DFT-derived aqueous redox-potential scale, V vs SHE
```

It is not an LLM prediction and does not run DFT during a request. Its training
labels are derived from RedDB calculations, rather than measurements. The
training chemistry is mainly quinones and aza-aromatics. A molecule such as
TEMPO can be correctly represented by the input assistant while still being
outside the prediction model's main applicability domain. That result should be
treated as an extrapolative model output, not an experimental value or a
screening decision by itself.

## Operational rules

1. Never call `/v1/predict` before the user has seen a rendered structure.
2. Prefer PubChem over an LLM for name-to-structure resolution.
3. Label LLM fallback and LLM edit candidates in the UI.
4. Keep SMILES editable; display the RDKit drawing after every change.
5. Show the final canonical SMILES and 2D drawing with the prediction result.
6. Do not claim the returned potential is an experimental measurement.
