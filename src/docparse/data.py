"""CORD v2 loading and normalization into the shared `Receipt` schema."""

from __future__ import annotations

import json
from typing import Any, Iterator

from .schema import LineItem, Receipt

CORD_DATASET = "naver-clova-ix/cord-v2"

# CORD key -> our schema field
_SUBTOTAL_MAP = {
    "subtotal_price": "subtotal",
    "tax_price": "tax",
    "service_price": "service",
    "discount_price": "discount",
}
_TOTAL_MAP = {
    "total_price": "total",
    "cashprice": "cash",
    "changeprice": "change",
}


def _text(value: Any) -> str | None:
    """CORD values are usually strings, but repeated keys come back as lists."""
    if value is None:
        return None
    if isinstance(value, list):
        parts = [_text(v) for v in value]
        value = " ".join(p for p in parts if p)
    if isinstance(value, dict):
        return None
    value = str(value).strip()
    return value or None


def _as_list(value: Any) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    return [v for v in value if isinstance(v, dict)]


def cord_to_receipt(gt_parse: dict) -> Receipt:
    items = [
        LineItem(name=_text(m.get("nm")), quantity=_text(m.get("cnt")), price=_text(m.get("price")))
        for m in _as_list(gt_parse.get("menu"))
    ]
    items = [it for it in items if any((it.name, it.quantity, it.price))]

    header: dict[str, str | None] = {}
    for block_key, mapping in (("sub_total", _SUBTOTAL_MAP), ("total", _TOTAL_MAP)):
        for block in _as_list(gt_parse.get(block_key)):
            for src, dst in mapping.items():
                if header.get(dst) is None:
                    header[dst] = _text(block.get(src))
    return Receipt(items=items, **header)


def parse_cord_ground_truth(raw: str) -> Receipt:
    return cord_to_receipt(json.loads(raw)["gt_parse"])


def iter_split(split: str, limit: int | None = None) -> Iterator[tuple[str, Any, Receipt]]:
    """Yield `(doc_id, PIL image, gold Receipt)` for a CORD split ("train" / "validation" / "test")."""
    from datasets import load_dataset

    ds = load_dataset(CORD_DATASET, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    for i, row in enumerate(ds):
        yield f"{split}-{i:04d}", row["image"].convert("RGB"), parse_cord_ground_truth(row["ground_truth"])
