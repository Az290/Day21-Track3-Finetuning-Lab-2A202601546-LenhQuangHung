#!/usr/bin/env python3
"""Does TRL's `assistant_only_loss` mask the same tokens labkit says it should?

This is the lab's central claim under test. NB1 proves *labkit's* mask is right by
decoding it. NB3 then hands training to TRL with `assistant_only_loss=True` and trusts
it to do the same thing. If TRL masks differently — or silently masks nothing — every
number downstream is measuring a different experiment than the one NB1 verified.

TRL's assistant-only masking depends on the chat template exposing `{% generation %}`
markers. Templates that lack them can fall back to *no masking at all* without raising,
which is the dangerous case: training completes, the loss curve looks plausible, and
the model has been trained on the prompt as well.

    python scripts/check_mask_agreement.py

Fast: tokenizer only, no weights, no GPU.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import data                                    # noqa: E402
from labkit.config import get_tier                         # noqa: E402


def main() -> int:
    from transformers import AutoTokenizer

    tier = get_tier()
    tok = AutoTokenizer.from_pretrained(tier.model_id, trust_remote_code=True)
    print(f"model: {tier.model_id}")

    template = getattr(tok, "chat_template", "") or ""
    has_generation = "{% generation %}" in template or "{%- generation %}" in template
    print(f"chat template exposes {{% generation %}} markers: {has_generation}")
    if not has_generation:
        print("  -> TRL cannot derive an assistant-only mask from this template alone.")

    messages = [
        {"role": "user", "content": "Phan loai ticket nay giup toi."},
        {"role": "assistant", "content": '{"intent": "doi_tra"}'},
    ]

    ours = data.build_example(tok, messages, max_length=1024, mask_mode="assistant-only")
    print(f"\nlabkit assistant-only : {ours.n_supervised}/{ours.n_total} tokens "
          f"({ours.supervised_fraction:.1%})")
    print(f"  supervised text: {data.decode_supervised(tok, ours)!r}")

    # What TRL produces, via the same public API SFTTrainer uses.
    try:
        rendered = tok.apply_chat_template(messages, tokenize=True,
                                           return_assistant_tokens_mask=True,
                                           return_dict=True)
    except Exception as exc:
        print(f"\nTRL path: apply_chat_template(return_assistant_tokens_mask=True) "
              f"raised {type(exc).__name__}: {exc}")
        print("VERDICT: cannot cross-check automatically — see the note below.")
        return _advise(has_generation)

    mask = rendered.get("assistant_masks")
    if mask is None:
        print("\nTRL path: no `assistant_masks` returned.")
        return _advise(has_generation)

    ids = rendered["input_ids"]
    if ids and isinstance(ids[0], list):
        ids, mask = ids[0], mask[0]
    theirs_n = sum(mask)
    theirs_text = tok.decode([t for t, m in zip(ids, mask) if m], skip_special_tokens=False)
    print(f"\nTRL assistant_masks   : {theirs_n}/{len(ids)} tokens "
          f"({theirs_n/max(1,len(ids)):.1%})")
    print(f"  supervised text: {theirs_text!r}")

    if theirs_n == 0:
        print("\nVERDICT: FAIL — TRL would supervise NOTHING. Training would be a no-op "
              "on the loss you care about.")
        return 1
    if theirs_n == len(ids):
        print("\nVERDICT: FAIL — TRL would supervise EVERY token, prompt included "
              "(deck §16: the model learns to rewrite your question).")
        return 1

    answer = messages[-1]["content"][:20]
    agree = answer in theirs_text and answer in data.decode_supervised(tok, ours)
    delta = abs(theirs_n - ours.n_supervised)
    print(f"\nboth supervise the answer: {agree}   |token-count delta| = {delta}")
    if agree and delta <= 4:
        print("VERDICT: PASS — TRL and labkit agree on what the loss covers.")
        return 0
    print("VERDICT: MISMATCH — investigate before trusting NB3's numbers.")
    return 1


def _advise(has_generation: bool) -> int:
    print("\nWhat to do:")
    if not has_generation:
        print("  * This template has no {% generation %} markers, so TRL's")
        print("    assistant_only_loss has nothing to key off. Prefer")
        print("    completion_only_loss (prompt-completion data), or pre-tokenize with")
        print("    labkit.data.build_example and train on input_ids/labels directly.")
    print("  * Either way: print one training batch's labels before a long run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
