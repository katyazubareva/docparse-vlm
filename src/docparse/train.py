"""LoRA supervised fine-tuning of Qwen2-VL on CORD v2.

Fits on a single 16 GB T4 (Kaggle free tier) with the defaults: fp16 base weights,
fp32 adapters, gradient checkpointing, batch 1 x grad-accum 8.
"""

from __future__ import annotations

import argparse
import random

from .data import iter_split
from .infer import DEFAULT_MODEL, MAX_PIXELS, build_messages, load_processor, pick_dtype
from .prompts import target_json

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # LLM only


class Collator:
    """Builds full chat sequences and masks everything except the assistant's JSON answer."""

    def __init__(self, processor):
        self.processor = processor
        tok = processor.tokenizer
        self.assistant_header = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.pad_id = tok.pad_token_id

    def _answer_start(self, ids: list[int]) -> int:
        h = self.assistant_header
        for i in range(len(ids) - len(h), -1, -1):
            if ids[i : i + len(h)] == h:
                return i + len(h)
        raise ValueError("assistant header not found in sequence")

    def __call__(self, batch):
        texts = [self.processor.apply_chat_template(build_messages(target_json(r))) for _, r in batch]
        enc = self.processor(text=texts, images=[img for img, _ in batch], padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        for row, ids in enumerate(enc["input_ids"].tolist()):
            labels[row, : self._answer_start(ids)] = -100
        labels[labels == self.pad_id] = -100
        enc["labels"] = labels
        return enc


def main(argv=None):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2VLForConditionalGeneration, Trainer, TrainingArguments

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-docs", type=int, default=None, help="subsample for the data-efficiency curve")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    random.seed(args.seed)
    train = [(img, r) for _, img, r in iter_split("train")]
    if args.train_docs:
        train = random.sample(train, args.train_docs)
    val = [(img, r) for _, img, r in iter_split("validation")]

    dtype = pick_dtype()
    processor = load_processor(args.model, args.max_pixels)
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=LORA_TARGETS,
                   task_type="CAUSAL_LM"),
    )
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
    )
    Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
            data_collator=Collator(processor)).train()
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)


if __name__ == "__main__":
    main()
