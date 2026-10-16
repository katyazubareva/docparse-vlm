"""FastAPI service: POST an image, get validated receipt JSON.

    DOCPARSE_ADAPTER=runs/lora-r16 uvicorn docparse.serve:app --port 8000
    curl -F file=@receipt.jpg localhost:8000/extract
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .schema import Receipt

app = FastAPI(title="docparse-vlm")


@lru_cache(maxsize=1)
def get_extractor():
    from .infer import QwenVLExtractor

    return QwenVLExtractor(adapter=os.environ.get("DOCPARSE_ADAPTER"))


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> dict:
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(415, "not a decodable image")
    pred = get_extractor()(image)
    if pred.receipt is None:
        raise HTTPException(422, {"error": "model output failed schema validation", "raw": pred.raw})
    return {"receipt": Receipt.model_validate(pred.receipt).model_dump(), "latency_s": pred.latency_s}
