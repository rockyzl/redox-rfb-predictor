"""Public, rate-limited API for the fast Redox RFB predictor."""

from __future__ import annotations

import os
import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request as UrlRequest, urlopen
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem, rdBase
from rdkit.Chem.Draw import rdMolDraw2D

from redox_rfb import predict

SCHEMA_VERSION = "redox-rfb-live-v1"
MAX_SMILES_LENGTH = 512
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "12"))
NAME_RATE_LIMIT_PER_MINUTE = int(os.environ.get("NAME_RATE_LIMIT_PER_MINUTE", "6"))
ALLOWED_ORIGINS = tuple(
    value.strip()
    for value in os.environ.get(
        "ALLOWED_ORIGINS",
        "https://sciencesloop.com,https://www.sciencesloop.com",
    ).split(",")
    if value.strip()
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    smiles: Annotated[str, Field(min_length=1, max_length=MAX_SMILES_LENGTH)]


class NameResolutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, Field(min_length=1, max_length=200)]


class NameSuggestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, Field(min_length=1, max_length=200)]


class StructurePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    smiles: Annotated[str, Field(min_length=1, max_length=MAX_SMILES_LENGTH)]


class StructureChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction: Annotated[str, Field(min_length=1, max_length=400)]
    current_smiles: Annotated[str, Field(min_length=1, max_length=MAX_SMILES_LENGTH)]


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self.lock:
            queue = self.events[client]
            while queue and queue[0] <= now - 60:
                queue.popleft()
            if len(queue) >= self.per_minute:
                return False
            queue.append(now)
            return True


def _validated_smiles(raw: str) -> str:
    value = raw.strip()
    with rdBase.BlockLogs():
        if not value or Chem.MolFromSmiles(value, sanitize=True) is None:
            raise ValueError("invalid SMILES")
    return value


def _structure_record(smiles: str) -> dict[str, str]:
    """Return a canonical, RDKit-validated structure plus a server-rendered 2D SVG."""
    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:  # guarded by _validated_smiles; retained for fail-closed behavior
        raise ValueError("invalid SMILES")
    canonical = Chem.MolToSmiles(molecule, canonical=True)
    drawer = rdMolDraw2D.MolDraw2DSVG(480, 280)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    return {"smiles": canonical, "svg": drawer.GetDrawingText()}


def _pubchem_name_lookup(name: str) -> dict[str, str] | None:
    """Look up a name in PubChem before asking an LLM to infer a structure."""
    endpoint = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(name, safe='')}/property/Title,IsomericSMILES,CanonicalSMILES/JSON"
    )
    request = UrlRequest(endpoint, headers={"User-Agent": "redox-rfb-predictor/0.1"})
    try:
        with urlopen(request, timeout=6) as response:  # nosec B310: fixed HTTPS host
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None
    records = payload.get("PropertyTable", {}).get("Properties", [])
    if not records or not isinstance(records[0], dict):
        return None
    record = records[0]
    smiles = record.get("SMILES") or record.get("IsomericSMILES")
    cid = record.get("CID")
    if not isinstance(smiles, str) or not isinstance(cid, int):
        return None
    title = record.get("Title")
    return {"smiles": smiles, "cid": str(cid), "title": title if isinstance(title, str) else name}


def _pubchem_name_suggestions(name: str) -> list[str]:
    endpoint = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/"
        f"{quote(name, safe='')}/JSON?limit=5"
    )
    request = UrlRequest(endpoint, headers={"User-Agent": "redox-rfb-predictor/0.1"})
    try:
        with urlopen(request, timeout=6) as response:  # nosec B310: fixed HTTPS host
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return []
    terms = payload.get("dictionary_terms", {}).get("compound", [])
    return [term for term in terms[:5] if isinstance(term, str) and term.strip()]


def _candidate(
    *, candidate_id: str, label: str, smiles: str, source: str, source_url: str | None = None
) -> dict[str, str]:
    record = _structure_record(_validated_smiles(smiles))
    return {
        "id": candidate_id,
        "label": label[:120],
        "source": source,
        "source_url": source_url or "",
        **record,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once on startup so the first public request is not a cold model load.
    predict("O=C1C=CC(=O)C=C1", model="rdkit")
    app.state.limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)
    app.state.name_limiter = RateLimiter(NAME_RATE_LIMIT_PER_MINUTE)
    yield


app = FastAPI(title="Redox RFB Predictor API", docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "schema_version": SCHEMA_VERSION, "model": "rdkit"}


@app.post("/v1/predict")
def predict_redox(payload: PredictionRequest, request: Request) -> dict[str, object]:
    client = request.client.host if request.client else "unknown"
    if not request.app.state.limiter.allow(client):
        raise HTTPException(status_code=429, detail="rate limit exceeded; retry in one minute")
    try:
        smiles = _validated_smiles(payload.smiles)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        value = predict(smiles, model="rdkit")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="prediction failed") from exc
    return {
        "schema_version": SCHEMA_VERSION,
        "smiles": smiles,
        "model": {
            "id": "rdkit-morgan-baseline",
            "label": "RDKit descriptors + Morgan fingerprint",
            "training_rows": 15673,
        },
        "prediction": {"value": round(value, 6), "unit": "V vs SHE"},
        "target": {
            "name": "aqueous redox potential",
            "reference": "RedDB DFT-derived 2e-/2H+ potential scale",
        },
        "limitations": [
            "This is a prediction on a DFT-derived RedDB label scale, not an experimental measurement.",
            "The model was trained mainly on quinone and aza-aromatic molecules; other chemotypes are extrapolation.",
        ],
    }


@app.post("/v1/preview-structure")
def preview_structure(payload: StructurePreviewRequest, request: Request) -> dict[str, object]:
    """Validate and render a submitted SMILES before a user requests prediction."""
    client = request.client.host if request.client else "unknown"
    if not request.app.state.limiter.allow(client):
        raise HTTPException(status_code=429, detail="rate limit exceeded; retry in one minute")
    try:
        record = _structure_record(_validated_smiles(payload.smiles))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"schema_version": SCHEMA_VERSION, **record, "renderer": "RDKit 2D"}


@app.post("/v1/suggest-name")
def suggest_name(payload: NameSuggestionRequest, request: Request) -> dict[str, object]:
    """Offer name corrections/options before resolving any molecular structure."""
    client_ip = request.client.host if request.client else "unknown"
    if not request.app.state.name_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="name-suggestion limit exceeded; retry in one minute")
    name = payload.name.strip()
    suggestions = _pubchem_name_suggestions(name)
    exact = _pubchem_name_lookup(name)
    if exact and exact["title"] not in suggestions:
        suggestions.insert(0, exact["title"])
    source = "PubChem autocomplete"
    if not suggestions and os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {"suggestions": {"type": "array", "minItems": 0, "maxItems": 5, "items": {"type": "string"}}},
            "required": ["suggestions"],
        }
        try:
            response = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).responses.create(
                model=os.environ.get("NAME_TO_SMILES_MODEL", "gpt-5-mini"),
                input=("Suggest up to five possible corrected chemical names for this input. "
                       "Do not suggest molecular structures or SMILES. Return an empty list if unsure. Input: " + name),
                text={"format": {"type": "json_schema", "name": "chemical_name_suggestions", "strict": True, "schema": schema}},
            )
            generated = json.loads(response.output_text).get("suggestions", [])
            suggestions = [item for item in generated if isinstance(item, str) and item.strip()][:5]
            source = "LLM spelling suggestion"
        except Exception:
            suggestions = []
    return {
        "schema_version": SCHEMA_VERSION,
        "input": name,
        "suggestions": suggestions,
        "source": source if suggestions else "No suggestion available",
    }


@app.post("/v1/structure-chat")
def structure_chat(payload: StructureChatRequest, request: Request) -> dict[str, object]:
    """Use the LLM only to propose *editable* structure changes, then validate them."""
    client_ip = request.client.host if request.client else "unknown"
    if not request.app.state.name_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="structure-assistant limit exceeded; retry in one minute")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="structure assistant is not configured")
    try:
        current = _structure_record(_validated_smiles(payload.current_smiles))["smiles"]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "reply": {"type": "string"},
            "candidates": {
                "type": "array", "minItems": 0, "maxItems": 3,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {"label": {"type": "string"}, "smiles": {"type": "string"}},
                    "required": ["label", "smiles"],
                },
            },
        },
        "required": ["reply", "candidates"],
    }
    prompt = (
        "You are a molecular-structure assistant. A user has already confirmed this current SMILES: "
        f"{current}\nTheir requested edit: {payload.instruction.strip()}\n"
        "Return a short response and up to three proposed, chemically explicit SMILES candidates. "
        "Preserve charges and radicals when requested. If the request is ambiguous or you cannot make a "
        "chemically defensible edit, return no candidates and explain why. Do not predict properties."
    )
    try:
        from openai import OpenAI
        response = OpenAI(api_key=api_key).responses.create(
            model=os.environ.get("NAME_TO_SMILES_MODEL", "gpt-5-mini"), input=prompt,
            text={"format": {"type": "json_schema", "name": "structure_edit", "strict": True, "schema": schema}},
        )
        result = json.loads(response.output_text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="structure assistant failed") from exc
    candidates = []
    seen_smiles = set()
    for index, raw in enumerate(result.get("candidates", [])[:3], start=1):
        if not isinstance(raw, dict):
            continue
        try:
            candidate = _candidate(
                candidate_id=f"chat-{index}",
                label=str(raw.get("label", f"Assistant candidate {index}")),
                smiles=str(raw.get("smiles", "")),
                source="LLM structure-edit candidate",
            )
        except ValueError:
            continue
        if candidate["smiles"] not in seen_smiles:
            candidates.append(candidate)
            seen_smiles.add(candidate["smiles"])
    return {
        "schema_version": SCHEMA_VERSION,
        "reply": str(result.get("reply", "I could not propose a validated edit."))[:600],
        "candidates": candidates,
        "source": "LLM structure assistant; candidates validated by RDKit",
    }


@app.post("/v1/resolve-name")
def resolve_name(payload: NameResolutionRequest, request: Request) -> dict[str, object]:
    """Resolve one chemical name through an LLM, then validate the returned SMILES."""
    client_ip = request.client.host if request.client else "unknown"
    if not request.app.state.name_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="name-resolution limit exceeded; retry in one minute")
    name = payload.name.strip()
    pubchem = _pubchem_name_lookup(name)
    if pubchem:
        try:
            candidate = _candidate(
                candidate_id=f"pubchem-{pubchem['cid']}",
                label=f"PubChem: {pubchem['title']} (CID {pubchem['cid']})",
                smiles=pubchem["smiles"],
                source="PubChem exact name match",
                source_url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem['cid']}",
            )
        except ValueError:
            pubchem = None
        else:
            return {
                "schema_version": SCHEMA_VERSION,
                "name": name,
                "smiles": candidate["smiles"],
                "svg": candidate["svg"],
                "candidates": [candidate],
                "resolver": {
                    "type": "PubChem name lookup then RDKit validation",
                    "source": "PubChem PUG-REST",
                    "cid": int(pubchem["cid"]),
                    "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem['cid']}",
                },
            }

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="name resolution is not configured")

    from openai import OpenAI

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 0,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"label": {"type": "string"}, "smiles": {"type": "string"}},
                    "required": ["label", "smiles"],
                },
            },
        },
        "required": ["candidates"],
    }
    prompt = (
        "Convert this chemical name to candidate molecular structures. Return exactly one candidate when "
        "the name is unambiguous. Return at most three candidates only when the name genuinely permits "
        "different chemical structures or oxidation states; label each distinction clearly. Do not invent "
        "structures. If you are not confident or it is not a molecule, return an empty candidates list. "
        "Write valid SMILES and preserve explicit charges or radicals. Chemical name: " + name
    )
    try:
        response = OpenAI(api_key=api_key).responses.create(
            model=os.environ.get("NAME_TO_SMILES_MODEL", "gpt-5-mini"),
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "chemical_name_to_smiles",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        result = json.loads(response.output_text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="name resolution failed") from exc
    raw_candidates = result.get("candidates")
    if not isinstance(raw_candidates, list):
        raise HTTPException(status_code=422, detail="could not resolve this chemical name")
    candidates = []
    seen_smiles = set()
    for index, raw in enumerate(raw_candidates[:3], start=1):
        if not isinstance(raw, dict):
            continue
        try:
            candidate = _candidate(
                candidate_id=f"llm-{index}",
                label=str(raw.get("label", f"LLM candidate {index}")),
                smiles=str(raw.get("smiles", "")),
                source="LLM fallback candidate",
            )
        except ValueError:
            continue
        if candidate["smiles"] not in seen_smiles:
            candidates.append(candidate)
            seen_smiles.add(candidate["smiles"])
    if not candidates:
        raise HTTPException(status_code=422, detail="could not resolve this chemical name")
    selected = candidates[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "smiles": selected["smiles"],
        "svg": selected["svg"],
        "candidates": candidates,
        "resolver": {
            "type": "LLM fallback then RDKit validation",
            "source": "LLM fallback",
            "model": os.environ.get("NAME_TO_SMILES_MODEL", "gpt-5-mini"),
        },
    }
