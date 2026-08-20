"""Model loading and batch generation — the only GPU-touching module.

Kept separate from `evaluate` on purpose: scoring is pure functions over strings and
runs anywhere, so the whole grading contract stays testable on a laptop. Only this
file needs a GPU.
"""
from __future__ import annotations

import gc
import time

from . import device
from .config import Tier

# The two prompts that define baselines (a) and (b). Baseline (b) has to be a genuine
# effort — deck §17's whole point is that a fine-tune which cannot beat a *well-prompted*
# base model is not worth shipping. Writing a deliberately weak (b) to flatter your
# fine-tune is the main way to cheat this lab, and the rubric checks for it.
NAIVE_PROMPT = "Phân loại ticket sau."

OPTIMIZED_PROMPT = """Bạn là hệ thống phân loại ticket CSKH. Trả về DUY NHẤT một object JSON, không kèm giải thích, không kèm markdown fence.

Schema bắt buộc — đúng 4 khóa:
{"intent": ..., "urgency": ..., "product": ..., "sentiment": ...}

intent    ∈ doi_tra | van_chuyen | hoan_tien | san_pham_loi | hoi_thong_tin
urgency   ∈ cao | trung_binh | thap
sentiment ∈ tieu_cuc | trung_tinh | tich_cuc
product   = tên sản phẩm xuất hiện nguyên văn trong ticket

Ví dụ:
Ticket: "Shop ơi, mình đặt bàn phím cơ mã đơn DH123456. Giao hàng chậm. Đã 3 ngày rồi. Nhờ shop kiểm tra."
JSON: {"intent": "van_chuyen", "urgency": "trung_binh", "product": "bàn phím cơ", "sentiment": "trung_tinh"}"""


def free_memory() -> None:
    """Between runs. Deck §16: not doing this is the most common OOM in a multi-run lab."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:      # pragma: no cover
        pass


def peak_vram_gb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 3
    except ImportError:      # pragma: no cover
        pass
    return None


def load_base(tier: Tier, load_in_4bit: bool = False):
    """Load the base model + tokenizer for `tier`.

    `load_in_4bit` is exposed only so NB4 can *measure* the QLoRA contrast. The default
    is bf16 because the vendor advises against 4-bit on this model family (deck §12).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tier.model_id, trust_remote_code=True)
    # dtype (not torch_dtype — deprecated in transformers 5.x) and NOT hardcoded bf16:
    # the lab's default tier is a T4, which has no bfloat16 (see labkit/device.py).
    kwargs: dict = {"trust_remote_code": True, "dtype": device.torch_dtype(),
                    "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=device.torch_dtype(),
        )
    model = AutoModelForCausalLM.from_pretrained(tier.model_id, **kwargs)
    return model, tok


def generate_batch(
    model,
    tok,
    prompts: list[str],
    system: str | None = None,
    max_new_tokens: int = 160,
    enable_thinking: bool = False,
    batch_size: int = 4,
) -> tuple[list[str], float]:
    """Greedy decode. Returns (completions, mean_latency_ms_per_item).

    Greedy (do_sample=False) is deliberate: this lab compares runs, and sampling noise
    would swamp the differences you are trying to measure.
    """
    import torch

    outs: list[str] = []
    total_ms = 0.0
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        texts = []
        for p in chunk:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": p}]
            kw = {"tokenize": False, "add_generation_prompt": True}
            try:
                texts.append(tok.apply_chat_template(msgs, enable_thinking=enable_thinking, **kw))
            except TypeError:
                texts.append(tok.apply_chat_template(msgs, **kw))

        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        total_ms += (time.perf_counter() - t0) * 1000.0
        for row, src in zip(gen, enc["input_ids"]):
            outs.append(tok.decode(row[len(src):], skip_special_tokens=True).strip())

    return outs, total_ms / max(1, len(prompts))
