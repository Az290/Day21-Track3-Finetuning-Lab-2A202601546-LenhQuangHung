#!/usr/bin/env python3
"""Why does a fine-tune with a healthy loss curve score 0.000 on the task?

Measured on a free-Colab T4: `correct` trained to final_loss 0.0173 and then scored
target=0.000 format=0.000 -- the same numbers as the *untrained* baseline (a) -- while
taking 1.6x as long per sample. A model that learned nothing is fast and wrong; a model
that is off-distribution is slow and wrong. The latency said off-distribution.

The hypothesis this checks: the lab renders a DIFFERENT prompt at training time than at
generation time.

    training   : data._render(..., enable_thinking=None)   -> kwarg omitted, template default
    generation : apply_chat_template(..., enable_thinking=False)  -> passed explicitly

If those two disagree, every adapter in the lab is fine-tuned to continue one scaffold
and then asked to continue another. Prints the two renders, then generates under both
settings plus with the adapter disabled, so the three explanations separate cleanly:

  adapter-off == adapter-on     -> the adapter never loaded (inert)
  thinking=None good, False bad -> template mismatch; the adapter is fine
  both bad                      -> the training itself did not transfer

    python scripts/diagnose_generation.py [adapter_key]      # default: correct
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

N_PROMPTS = 2


def main() -> int:
    key = sys.argv[1] if len(sys.argv) > 1 else "correct"

    from peft import PeftModel

    from labkit import data, device, generate
    from labkit.config import SPECS, get_tier

    tier = get_tier()
    print(device.banner())

    rows = [json.loads(l) for l in (ROOT / "data" / "eval_target.jsonl")
            .open(encoding="utf-8") if l.strip()][:N_PROMPTS]
    train_rows = [json.loads(l) for l in (ROOT / "data" / "split" / "train.jsonl")
                  .open(encoding="utf-8") if l.strip()][:1]

    model, tok = generate.load_base(tier, load_in_4bit=SPECS[key].load_in_4bit)

    # --- 1. the two renders, side by side -----------------------------------------
    msgs = data.to_messages(train_rows[0])
    trained_on = data._render(tok, msgs, False, None)          # what NB3 tokenized
    print("\n" + "=" * 72)
    print("TRAINING render (enable_thinking omitted -> template default)")
    print("=" * 72)
    print(repr(trained_on[:700]))

    ask = [{"role": "system", "content": generate.NAIVE_PROMPT},
           {"role": "user", "content": rows[0]["input"]}]
    for et in (None, False, True):
        try:
            kw = {"tokenize": False, "add_generation_prompt": True}
            if et is not None:
                kw["enable_thinking"] = et
            r = tok.apply_chat_template(ask, **kw)
        except TypeError as exc:
            r = f"<TypeError: {exc}>"
        print(f"\n--- GENERATION render, enable_thinking={et} ---")
        print(repr(r[-320:]))

    # --- 2. generate under each condition ------------------------------------------
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / key))
    model.eval()
    prompts = [r["input"] for r in rows]

    for et in (False, None):
        outs, ms = generate.generate_batch(
            model, tok, prompts, system=generate.NAIVE_PROMPT,
            enable_thinking=et, progress=False, label=f"{key}/et={et}")
        print(f"\n{'=' * 72}\n{key} adapter ON, enable_thinking={et}  ({ms:.0f} ms/sample)")
        for o in outs:
            print("   ", repr(o[:260]))

    with model.disable_adapter():
        outs, ms = generate.generate_batch(
            model, tok, prompts, system=generate.NAIVE_PROMPT,
            enable_thinking=False, progress=False, label="base")
    print(f"\n{'=' * 72}\nadapter DISABLED (this is baseline (a))  ({ms:.0f} ms/sample)")
    for o in outs:
        print("   ", repr(o[:260]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
