"""The single instruction prompt and output parser shared by every generative system."""

from __future__ import annotations

import json

from pydantic import ValidationError

from .schema import Receipt

INSTRUCTION = """Extract the receipt into JSON with exactly this structure:
{"items": [{"name": str, "quantity": str, "price": str}], "subtotal": str, "tax": str, "service": str, "discount": str, "total": str, "cash": str, "change": str}
Copy values exactly as printed. Use null for fields that are not on the receipt. Output only the JSON."""


def target_json(receipt: Receipt) -> str:
    """Canonical serialization used as the SFT target (compact to save tokens)."""
    return json.dumps(receipt.model_dump(), ensure_ascii=False, separators=(",", ":"))


def parse_output(text: str) -> Receipt | None:
    """Parse the outermost JSON object out of model text. Returns None when invalid."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return Receipt.model_validate(json.loads(text[start : end + 1]))
    except (json.JSONDecodeError, ValidationError):
        return None
