"""Public, rate-limited API for the fast Redox RFB predictor."""

from __future__ import annotations

import os
import json
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem, rdBase

from redox_rfb import predict

SCHEMA_VERSION = "redox-rfb-live-v1"
MAX_SMILES_LENGTH = 512
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "12"))
NAME_RATE_LIMIT_PER_MINUTE = int(os.environ.get("NAME_RATE_LIMIT_PER_MINUTE", "3"))
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


@app.post("/v1/resolve-name")
def resolve_name(payload: NameResolutionRequest, request: Request) -> dict[str, object]:
    """Resolve one chemical name through an LLM, then validate the returned SMILES."""
    client_ip = request.client.host if request.client else "unknown"
    if not request.app.state.name_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="name-resolution limit exceeded; retry in one minute")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="name resolution is not configured")

    from openai import OpenAI

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "resolved": {"type": "boolean"},
            "smiles": {"type": "string"},
        },
        "required": ["resolved", "smiles"],
    }
    prompt = (
        "Convert this one chemical name to a single neutral, valid SMILES string. "
        "Use the most conventional structure intended by the name. Do not invent a structure. "
        "If the name is ambiguous, not a molecule, or you are not confident, set resolved=false "
        "and smiles to an empty string. Chemical name: " + payload.name.strip()
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
    if not result.get("resolved"):
        raise HTTPException(status_code=422, detail="could not resolve this chemical name")
    try:
        smiles = _validated_smiles(str(result.get("smiles", "")))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="LLM returned an invalid molecular structure") from exc
    return {
        "schema_version": SCHEMA_VERSION,
        "name": payload.name.strip(),
        "smiles": smiles,
        "resolver": {"type": "LLM then RDKit validation", "model": os.environ.get("NAME_TO_SMILES_MODEL", "gpt-5-mini")},
    }
