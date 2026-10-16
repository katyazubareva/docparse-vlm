"""Qwen2-VL inference, zero-shot or with a LoRA adapter."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .prompts import INSTRUCTION, parse_output
from .schema import Receipt

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
# Vision token budget: each 28x28 patch is one token. ~1M px keeps receipts legible at ~1.3k tokens.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


def pick_dtype():
    import torch

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def build_messages(target: str | None = None) -> list[dict]:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def load_processor(model_id: str = DEFAULT_MODEL, max_pixels: int = MAX_PIXELS):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_id, min_pixels=MIN_PIXELS, max_pixels=max_pixels)


@dataclass
class Prediction:
    receipt: Receipt | None
    raw: str
    latency_s: float
    output_tokens: int


class QwenVLExtractor:
    def __init__(self, model_id: str = DEFAULT_MODEL, adapter: str | None = None, max_pixels: int = MAX_PIXELS):
        import torch
        from transformers import Qwen2VLForConditionalGeneration

        self.processor = load_processor(model_id, max_pixels)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=pick_dtype(), device_map="auto"
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.eval()
        self._torch = torch

    def __call__(self, image, max_new_tokens: int = 768) -> Prediction:
        prompt = self.processor.apply_chat_template(build_messages(), add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.model.device)

        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        t0 = time.perf_counter()
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        return Prediction(parse_output(text), text, latency, int(new_tokens.numel()))
