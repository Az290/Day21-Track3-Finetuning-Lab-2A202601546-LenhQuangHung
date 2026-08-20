# Simulation findings — Lab 21

Student simulation of the documented path. Two environments:

| Env | Hardware | Python | Purpose |
|---|---|---|---|
| **local** | Apple M4, 16 GB, MPS (no CUDA) | 3.14.3 | fast reproduction + fixes |
| **Colab** | free-tier **Tesla T4 16 GB** | 3.x (Colab) | the documented default path |

Stack resolved on both (from PyPI, 2026-08-21):
`torch 2.13.0 · transformers 5.15.1 · trl 1.10.0 · peft 0.20.0 · accelerate 1.14.0 ·
datasets 5.0.1` — `bitsandbytes` correctly skipped on macOS by the platform marker.

---

## F-01 — NB1 crashed on the lab's own default model — **FIXED**

**Severity: critical.** The lab could not get past its first notebook on `Qwen3.5-4B`.

**Symptom.** `TemplateNotPrefixStable: turn 1: rendering messages[:2] does not extend
messages[:1]`. Raised on *ordinary* prompt→answer data, not an edge case.

**Root cause — subtle and worth reading.** `build_example()` diffed **token lists**
around each assistant turn:

```
generation prompt ends:  ...<|im_start|>assistant\n<think>\n
full render continues:   ...<think>\n\n</think>\n\n{answer}<|im_end|>\n
```

The *strings* are prefix-related (`full.startswith(prefix)` is `True`). The *token
lists* are not: the prefix's trailing `\n` is one token, but in the full render `\n\n`
merges into a single **different** token. Diffing tokens therefore compares
non-comparable sequences.

Two aggravating factors specific to 2026 reasoning models:
* Qwen3.5 emits `<think>\n\n</think>\n\n` **even when the answer contains no
  reasoning**, so every sample crossed this boundary.
* The template also *normalizes* author-supplied `<think>` blocks, so hand-written
  reasoning data renders differently from what you wrote.

**Fix.** Render to text, tokenize once with `return_offsets_mapping=True`, and
supervise tokens whose character span falls inside `[len(prefix), len(upto))`.
Verified: this reproduces `apply_chat_template(tokenize=True)` ids **exactly**, and
special tokens carry real offsets, so `<|im_end|>` — the stop signal — stays
supervised.

**Why the test suite missed it.** The fake tokenizer was a plain ChatML renderer with
character-level tokenization: no think scaffold, no token merging. It could not
express the failure. The fake now reproduces both behaviours, plus 5 regression tests
(58 → 63), including one that asserts the *fixture itself* still produces
non-prefix-related token lists — otherwise the regression test would pass vacuously.

---

## F-02 — `resolve_target_modules` picks up hybrid-attention projections — **BY DESIGN, documented**

On the real `Qwen3.5-4B`, `text-linear` resolves to **12** suffixes, not the 7 a plain
transformer has:

```
down_proj gate_proj up_proj  q_proj k_proj v_proj o_proj
in_proj_a in_proj_b in_proj_qkv in_proj_z out_proj   <- Gated DeltaNet layers
```

The extra five are the **linear-attention** layers (deck §6.4: 24 linear + 8 full
attention, a 3:1 interleave). Adapting them is correct — they are part of the text
decoder — but it changes the arithmetic: matched rank for attention-only is **r≈283**
on the real model versus r≈90 on a plain-transformer shape. `matched_rank()` computes
this at runtime, so the contrast stays fair. Confirmed the vision tower is still
excluded.

---

## F-03 — `AutoModelForCausalLM` accepts the multimodal config — **NO ACTION**

De-risked without downloading weights via `init_empty_weights()` +
`from_config`. `Qwen3_5Config` (architectures: `Qwen3_5ForConditionalGeneration`)
loads as `Qwen3_5ForCausalLM`. `generate.load_base()` works on the default tier.

---

## F-04 — `transformers` does not depend on `jinja2` — **FIXED (pre-Colab)**

`apply_chat_template` raises `ImportError` without it. Would have broken NB1 on cell
one for every student. Pinned in both requirements files.

---

## F-05 — Colab free tier allows only ONE GPU session — **DOCUMENTED**

Opening a second lab notebook while the first holds a runtime gives *"Quá nhiều phiên
đang hoạt động"* and the second silently never starts. Students running NB1 in one tab
and NB3 in another will hit this. Belongs in HARDWARE-GUIDE.md.

---

## F-06 — a "16 GB" T4 gives **14.6 GB** usable — **DOC FIX NEEDED**

Colab reports `VRAM: 14.6 GB`, not 16. The `Qwen3.5-4B` bf16 checkpoint is **9.32 GB**
on the wire, so weights alone take ~62% of the card before any activations, LoRA state
or optimizer moments. HARDWARE-GUIDE.md says "Colab Free T4 16 GB" and budgets ~10 GB
for 4B bf16 LoRA — the headroom is thinner than documented. Whether it actually fits
is what NB3 decides; the number in the guide should say 14.6 GB either way.

---

## F-07 — the lab hardcoded **bf16**, and its default GPU has none — **FIXED**

**Severity: high.** Found by inspection during the Colab run, then fixed pre-emptively.

`sft_config_kwargs()` emitted `bf16=True` unconditionally and `load_base()` used
`dtype=torch.bfloat16`. The lab's **default tier is a free-Colab T4**, which is Turing
(sm_75) — **bfloat16 requires Ampere (8.0+)**. So the recommended path was configured
for hardware the recommended hardware is not.

This is the standard 2026 tutorial bug: every guide is written on an A100, where
`bf16=True` is correct, and the flag gets copied onto cards that cannot do it.

fp16 is not a drop-in swap either: its exponent range is far smaller, so training needs
**gradient scaling** to avoid underflow — which trainers enable only when told
`fp16=True`. The precision decision therefore has to reach the *training arguments*,
not just the model load.

**Fix.** New `labkit/device.py`: `describe()` / `precision()` / `torch_dtype()` /
`banner()` pick bf16 → fp16 → fp32 from the actual device capability, and
`sft_config_kwargs(precision=...)` allows an explicit override. Both flags are always
set, never both true. 7 new tests (63 → 70), including one asserting the flags are
never hardcoded and one that a T4-shaped device produces the explanatory note.

---

## Verified working

| Check | Where | Result |
|---|---|---|
| `git clone` + `pip install` bootstrap | Colab T4 | ✅ `GPU: Tesla T4`, ~30 s |
| GitHub → Colab notebook launch | Colab | ✅ renders, Vietnamese intact |
| Tokenizer download + chat template | both | ✅ |
| `thinking_survives()` on real template | both | ✅ "reasoning preserved" |
| NB1 end-to-end, real 4B tokenizer | local | ✅ 39/188 supervised, both asserts green |
| Unit tests | both | ✅ 63 passed |
| `requirements.txt` resolution | local (py3.14) | ✅ dry-run clean |
| `verify.py` smoke + full | local | ✅ correctly separates done from not-done |
