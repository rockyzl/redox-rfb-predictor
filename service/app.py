"""Public, rate-limited API for the fast Redox RFB predictor."""

from __future__ import annotations

import os
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
