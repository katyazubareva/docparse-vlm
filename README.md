# docparse-vlm

**Turning receipt photos into validated, structured JSON with a fine-tuned small vision-language model, and showing what that costs and where it breaks.**

> Status: work in progress. Every number in the tables below is a placeholder until the runs are done.

## Why this project

Extracting fields from documents (receipts, invoices, forms) is one of the most common real-world uses of ML. Teams usually face a choice between three options:

1. **OCR + text LLM**: mature, but OCR errors carry through, and the layout is lost.
2. **A large proprietary VLM, zero-shot**: accurate, but costly per document and sends data off-site.
3. **A small open VLM, fine-tuned**: cheap and private. Is it accurate enough?

This repo answers question 3 with numbers. It compares all three options on the same held-out split, using the same metrics and the same output schema, and reports **accuracy, latency and $/1k documents** side by side.

## Dataset

[CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) has 1,000 Indonesian receipts (800 train / 100 val / 100 test) with nested annotations: menu line items, subtotal, tax and total.
The annotations are normalized into one strict schema ([`src/docparse/schema.py`](src/docparse/schema.py)), so every system is scored on exactly the same targets.

## Systems compared

| ID | System | Notes |
|----|--------|-------|
| B0 | Tesseract OCR → regex rules | floor baseline |
| B1 | OCR → open text LLM (Qwen2.5-7B-Instruct), JSON mode | classic pipeline |
| B2 | Qwen2-VL-2B-Instruct, zero-shot | the "before" for the fine-tune |
| B3 | Large VLM API, zero-shot | quality ceiling / cost reference |
| **M1** | **Qwen2-VL-2B + LoRA SFT** | the main model |
| M2 | M1 + schema-constrained decoding | removes invalid JSON |

## Metrics ([`src/docparse/metrics.py`](src/docparse/metrics.py))

- **Field-level F1**: each leaf is matched as a `(path, normalized value)` pair. Line items are aligned with Hungarian matching, so their order doesn't affect the score.
- **Exact document match**: the share of documents where every field is correct.
- **JSON validity rate**: the share of outputs that parse *and* pass the schema check.
- **Total-amount accuracy**: the field most business uses care about.
- **p50/p95 latency**, measured on a T4 and an A10G, plus **$/1k docs**.
- All headline numbers come with **bootstrap 95% CIs**, because the test split has only 100 receipts.

## Results (placeholder)

| System | Field F1 | Doc exact | JSON valid | Total acc | p50 ms | $/1k docs |
|--------|---------:|----------:|-----------:|----------:|-------:|----------:|
| B0 | – | – | – | – | – | – |
| B1 | – | – | – | – | – | – |
| B2 | – | – | – | – | – | – |
| B3 | – | – | – | – | – | – |
| **M1** | – | – | – | – | – | – |
| M2 | – | – | – | – | – | – |

## Error analysis (planned)

- Error taxonomy: OCR-like character errors, wrong field, missed line item, invented (hallucinated) item, number-format errors.
- Robustness: performance under blur, rotation, JPEG compression and low light (synthetic corruptions).
- Data efficiency: a learning curve at 50 / 100 / 200 / 400 / 800 training documents.
- Ablations: LoRA rank, image resolution (vision token budget), prompt format.

## Repo layout

```
src/docparse/
  schema.py      # pydantic target schema + validation
  data.py        # CORD v2 → schema normalization
  metrics.py     # field F1 with Hungarian item alignment, bootstrap CIs
  prompts.py     # shared instruction prompt
  infer.py       # VLM inference (zero-shot or LoRA adapter)
  train.py       # LoRA SFT for Qwen2-VL
  evaluate.py    # run a system on a split → metrics JSON
  serve.py       # FastAPI service
tests/           # unit tests for normalization + metrics
notebooks/       # Kaggle notebook: zero-shot vs LoRA on a free T4
```

## Run on Kaggle (free GPU)

Import [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) into Kaggle (File → Import Notebook → GitHub). Set Accelerator to **GPU T4 x2** and turn Internet **On**, then Run All.
The notebook checks the metric with oracle/empty runs, evaluates the zero-shot baseline, trains the LoRA adapter, evaluates it, and prints a comparison table and the worst errors. Set `SMOKE = True` for a quick end-to-end check first.

## Quickstart (local)

```bash
uv sync --extra train            # or: pip install -e ".[train]"
python -m docparse.evaluate --system qwen2vl-zeroshot --split test --limit 20
python -m docparse.train --output runs/lora-r16
python -m docparse.evaluate --system qwen2vl-lora --adapter runs/lora-r16 --split test
uvicorn docparse.serve:app --port 8000
```
