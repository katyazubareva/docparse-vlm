"""Field-level extraction metrics with order-invariant line-item alignment."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schema import AMOUNT_FIELDS, HEADER_FIELDS, ITEM_FIELDS, LineItem, Receipt

_WS = re.compile(r"\s+")
_NON_DIGIT = re.compile(r"\D")


def normalize(field: str, value: str | None) -> str | None:
    """Canonical form used for matching. Amounts and quantities compare by digits only,
    so "Rp 12.000", "12,000" and "12000" all match; names compare case/space-insensitively."""
    if value is None:
        return None
    value = unicodedata.normalize("NFKC", value)
    if field in AMOUNT_FIELDS or field == "quantity":
        value = _NON_DIGIT.sub("", value).lstrip("0") or ("0" if any(c.isdigit() for c in value) else "")
    else:
        value = _WS.sub(" ", value).strip().casefold()
    return value or None


def _item_values(item: LineItem) -> dict[str, str | None]:
    return {f: normalize(f, getattr(item, f)) for f in ITEM_FIELDS}


@dataclass
class DocCounts:
    tp: int = 0
    n_pred: int = 0
    n_gold: int = 0
    total_correct: bool | None = None  # None when gold has no total

    @property
    def exact(self) -> bool:
        return self.tp == self.n_pred == self.n_gold


def score_document(pred: Receipt | None, gold: Receipt) -> DocCounts:
    """Count matched (field, value) leaves. An unparseable prediction (`None`) scores as empty."""
    pred = pred or Receipt()
    c = DocCounts()

    for f in HEADER_FIELDS:
        p, g = normalize(f, getattr(pred, f)), normalize(f, getattr(gold, f))
        c.n_pred += p is not None
        c.n_gold += g is not None
        c.tp += p is not None and p == g

    p_items = [_item_values(i) for i in pred.items]
    g_items = [_item_values(i) for i in gold.items]
    c.n_pred += sum(v is not None for it in p_items for v in it.values())
    c.n_gold += sum(v is not None for it in g_items for v in it.values())
    if p_items and g_items:
        # Hungarian alignment maximizing matched leaves, so item order never matters.
        gain = np.array(
            [[sum(pv[f] is not None and pv[f] == gv[f] for f in ITEM_FIELDS) for gv in g_items] for pv in p_items]
        )
        rows, cols = linear_sum_assignment(gain, maximize=True)
        c.tp += int(gain[rows, cols].sum())

    g_total = normalize("total", gold.total)
    if g_total is not None:
        c.total_correct = normalize("total", pred.total) == g_total
    return c


def _prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@dataclass
class Report:
    n_docs: int
    precision: float
    recall: float
    f1: float
    f1_ci95: tuple[float, float]
    doc_exact: float
    json_valid: float
    total_acc: float

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate(counts: list[DocCounts], valid: list[bool], n_boot: int = 1000, seed: int = 0) -> Report:
    """Micro-averaged P/R/F1 over all leaves, plus a document-level bootstrap CI for F1."""
    tp = np.array([c.tp for c in counts])
    npred = np.array([c.n_pred for c in counts])
    ngold = np.array([c.n_gold for c in counts])
    p, r, f = _prf(int(tp.sum()), int(npred.sum()), int(ngold.sum()))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    boot = [_prf(int(tp[i].sum()), int(npred[i].sum()), int(ngold[i].sum()))[2] for i in idx]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    totals = [c.total_correct for c in counts if c.total_correct is not None]
    return Report(
        n_docs=len(counts),
        precision=p,
        recall=r,
        f1=f,
        f1_ci95=(float(lo), float(hi)),
        doc_exact=float(np.mean([c.exact for c in counts])),
        json_valid=float(np.mean(valid)),
        total_acc=float(np.mean(totals)) if totals else float("nan"),
    )
