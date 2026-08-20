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

## F-08 — NB2/NB5 print nothing for tens of minutes — **FIXED**

**Severity: medium (usability, but it makes students kill good runs).**

`score_run()` prints only after a whole baseline finishes. On the free T4 the observed
gap between "Loading weights: 100%" and the first number was **>15 minutes** with zero
output. That is indistinguishable from a hang, and the documented remedy for a hung
Colab cell is to interrupt it.

**Fix.** `generate_batch()` now prints a per-batch line with elapsed time and ETA,
labelled by which pass is running (`(a) base + naive prompt/target`, `ft/regression`,
…). NB2 and NB5 pass labels through.

## F-09 — the published time budget is optimistic — **DOC FIX NEEDED**

README claims NB2 ≈ 10 min and the core ≈ 80 min on a T4. Measured on free Colab:

| Stage | Claimed | Observed |
|---|---|---|
| NB1 | 2 min | **26 s** ✅ |
| model download (first run only) | not mentioned | **~70 s** for 9.32 GB |
| weight load | not mentioned | ~30 s |
| NB2 baseline (a) alone | — | **>15 min** (pre-fp16) |

**Measured after the fp16 fix**, from the new progress output:

```
[(a) base + naive prompt/target] batch 1/13    44s elapsed  ~523s left
[(a) base + naive prompt/target] batch 2/13    88s elapsed  ~483s left
```

**44 s per batch of 4 prompts ≈ 11 s/prompt** at `max_new_tokens=160`, 4B on a T4.
Extrapolating:

| Stage | README claim | Projected from measurement |
|---|---|---|
| NB1 | 2 min | **14–26 s** ✅ |
| NB2 (two baselines × 65 prompts) | 10 min | **≈ 23 min** |
| NB5 (one scoring pass) | 10 min | **≈ 12 min** |
| **core NB1–NB5** | **80 min** | **≈ 95–110 min** |

The structural cause is that the eval set is generated **three times** across the lab
(baseline a, baseline b, fine-tune). That is inherent to the three-baseline design and
is the right trade — but the README must say so, and `EVAL_LIMIT` should be presented
as the normal way to iterate rather than a hidden knob.

---

## F-10 — `assistant_only_loss=True` supervises **ZERO tokens** on the default model — **CRITICAL**

The single most damaging finding, and a **silent** one.

NB3 configured `assistant_only_loss=True` and handed training to TRL. TRL derives that
mask from `{% generation %}` markers in the chat template. **Qwen3.5's template has
none.** Result, measured by `scripts/check_mask_agreement.py`:

```
chat template exposes {% generation %} markers: False
labkit assistant-only : 11/31 tokens (35.5%)   '</think>\n\n{"intent": "doi_tra"}<|im_end|>\n'
TRL  assistant_masks  :  0/31 tokens ( 0.0%)   ''
VERDICT: FAIL — TRL would supervise NOTHING.
```

transformers emits a **warning, not an error**. Training completes. A loss curve is
drawn. The numbers are meaningless.

This is precisely the class of bug the deck spends §13.2 and §16 on — *"no error, a
plausible loss curve, and a broken model"* — reproduced by the lab's own default
configuration. NB1 proves the mask is correct and then NB3 threw that proof away and
trusted a library flag.

**Fix.** Stop trusting the flag. NB3 now trains on the **exact mask NB1 verified**:
`data.to_training_dataset()` pre-tokenizes with `build_example()`, so `input_ids` and
`labels` are the ones the student decoded and asserted on. `assistant_only_loss` is not
set at all.

Consequence, stated honestly on the slide-facing side: pre-tokenized labels are
incompatible with `packing`, so packing is off for this path. Deck §13.3's point
(packing is free only when boundaries are respected) still stands — here the *mask's
correctness* outranks the throughput, and the lab says so rather than quietly keeping a
flag that does nothing.

`scripts/check_mask_agreement.py` ships with the lab so students can run this check
against any base model they swap in.

---

## F-11 — the format scorer was stricter than the target scorer — **FIXED**

`triage_field_accuracy()` recovered a `{...}` block embedded in prose;
`has_required_keys()` accepted only bare or fenced JSON. A model answering
`"Day la ket qua: {...}"` therefore scored on **target** but **0.000 on format** — a
formatting failure that did not happen. Two scorers disagreeing about what counts as
JSON makes both numbers untrustworthy, and `format` is one of the four graded groups.

Both now share `_parse_json_loose()`. +2 tests (76 total).

Surfaced by the first real T4 measurement: `(a) base + naive prompt  target=0.000
format=0.000`. Those particular zeros turned out to be genuine — a naive prompt with no
schema produces prose, not JSON — but checking *why* they were zero exposed the
inconsistency.

## F-12 — observation: the optimized prompt is ~3× faster, not just more accurate

Measured on the T4: baseline (a) ran at **44 s/batch**, baseline (b) at **15 s/batch**.
Same model, same prompts, same decode settings. The optimized prompt tells the model to
emit only JSON, so it emits ~20 tokens and stops; the naive prompt lets it ramble to the
160-token cap.

Worth teaching: prompt engineering bought a **3× latency win before any fine-tuning**,
which sharpens deck §17's point that baseline (b) is a real bar — it is better on the
target metric *and* cheaper to serve.

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
