"""Run one system over a split and write predictions + metrics.

    python -m docparse.evaluate --system qwen2vl-zeroshot --split test
    python -m docparse.evaluate --system qwen2vl-lora --adapter runs/lora-r16 --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import iter_split
from .metrics import aggregate, score_document

SYSTEMS = ("qwen2vl-zeroshot", "qwen2vl-lora", "oracle", "empty")


def build_system(name: str, adapter: str | None):
    """Returns a callable image -> Prediction. `oracle`/`empty` sanity-check the metric (F1 = 1 / 0)."""
    from .infer import Prediction, QwenVLExtractor

    if name == "qwen2vl-zeroshot":
        return QwenVLExtractor()
    if name == "qwen2vl-lora":
        if not adapter:
            raise SystemExit("--adapter is required for qwen2vl-lora")
        return QwenVLExtractor(adapter=adapter)
    if name == "empty":
        return lambda image, gold: Prediction(None, "", 0.0, 0)
    if name == "oracle":
        return lambda image, gold: Prediction(gold, gold.model_dump_json(), 0.0, 0)
    raise SystemExit(f"unknown system {name!r}; choose from {SYSTEMS}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS, required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    system = build_system(args.system, args.adapter)
    takes_gold = args.system in ("oracle", "empty")
    out_dir = Path(args.out) / f"{args.system}-{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts, valid, latencies = [], [], []
    with open(out_dir / "predictions.jsonl", "w") as f:
        for doc_id, image, gold in iter_split(args.split, args.limit):
            pred = system(image, gold) if takes_gold else system(image)
            c = score_document(pred.receipt, gold)
            counts.append(c)
            valid.append(pred.receipt is not None)
            latencies.append(pred.latency_s)
            f.write(json.dumps({"id": doc_id, "raw": pred.raw, "valid": pred.receipt is not None,
                                "tp": c.tp, "n_pred": c.n_pred, "n_gold": c.n_gold,
                                "latency_s": pred.latency_s, "output_tokens": pred.output_tokens},
                               ensure_ascii=False) + "\n")
            print(f"{doc_id}  tp={c.tp}/{c.n_gold}  valid={pred.receipt is not None}  {pred.latency_s:.2f}s")

    report = aggregate(counts, valid).to_dict()
    report["latency_p50_s"] = float(np.percentile(latencies, 50))
    report["latency_p95_s"] = float(np.percentile(latencies, 95))
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
