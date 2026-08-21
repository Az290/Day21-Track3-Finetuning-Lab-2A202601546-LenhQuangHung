#!/usr/bin/env python3
"""B4 bonus (+3, BONUS-CHALLENGE.md) — a CONTROLLED rank sweep.

Fixes target_modules="text-linear", learning_rate=LORA_LR, and max_steps at whatever
NB3's `correct` run used, and varies ONLY r in {8, 64} (r=16 is already `correct` from
NB3 -- this script does not retrain it). Every other run in the lab either varies
placement (attn_only, matched-rank) or LR (wrong_lr) at a FIXED rank; this is the run
that isolates rank itself, holding everything else including the parameter-budget
consequence of rank constant across comparisons only via the shared target_modules and
step budget (NOT via matched_rank -- that would defeat the point of testing rank).

Usage (after NB3 has produced adapters/correct/ and results/runs.csv):
    python scripts/rank_sweep.py

Writes:
    adapters/rank_r8/, adapters/rank_r64/
    results/runs.csv       -- appends rank_r8, rank_r64 rows (same schema as NB3/NB4)
    results/rank_sweep.json -- target/format/latency for r in {8, 16, 64}, r=16 pulled
                               from the existing `correct` row so it is not retrained
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

from labkit import data, device, evaluate as ev, generate, modeling, report, train
from labkit.config import LORA_LR, LoraSpec, get_tier

ROOT = pathlib.Path.cwd() if (pathlib.Path.cwd() / "data").exists() else pathlib.Path.cwd().parent
TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def existing_correct_row() -> dict:
    import csv
    with open(ROOT / "results" / "runs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["run"] == "correct":
            return r
    raise SystemExit("no `correct` row in results/runs.csv -- run NB3 first")


def train_rank(r: int, max_steps: int, mask_mode: str) -> dict:
    key = f"rank_r{r}"
    out = ROOT / "adapters" / key
    if (out / "adapter_model.safetensors").exists():
        print(f"skip {key}: {out} already trained")
        return {}

    spec = LoraSpec(key=key, r=r, alpha=2 * r, target="text-linear", lr=LORA_LR,
                    load_in_4bit=False,
                    label=f"all-linear · r={r} · LR 10x · 16-bit (B4 rank sweep)",
                    teaches="B4: rank varied, placement/LR/steps held fixed at correct's values.")

    model, tok = generate.load_base(TIER, load_in_4bit=spec.load_in_4bit)
    targets = modeling.resolve_target_modules(model, spec.target)
    trainable = modeling.count_lora_params(model, targets, spec.r)
    print(f"[{key}] placement={spec.target} r={spec.r} trainable≈{trainable/1e6:.2f}M")

    split_dir = ROOT / "data" / "split"
    rows = data.to_training_dataset(
        tok, load_jsonl(split_dir / "train.jsonl"),
        max_length=TIER.max_length, mask_mode=mask_mode)
    train_ds = Dataset.from_list(rows)

    want_sft = train.sft_config_kwargs(
        TIER, spec, output_dir=str(out), mask_mode=mask_mode, max_steps=max_steps)
    sft_kwargs, _ = train.filter_kwargs(SFTConfig, want_sft, label="SFTConfig")
    lora_kwargs, _ = train.filter_kwargs(
        LoraConfig, train.lora_config_kwargs(spec, targets), label="LoraConfig")

    generate.free_memory()
    trainer = SFTTrainer(model=model, args=SFTConfig(**sft_kwargs), train_dataset=train_ds,
                         processing_class=tok, peft_config=LoraConfig(**lora_kwargs))
    fix = train.align_trainable_precision(trainer.model)
    print(f"[{key}] precision fix:", fix)

    t0 = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - t0
    print(f"[{key}] train {elapsed:.0f}s  final loss {result.training_loss:.4f}")

    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[{key}] saved -> {out}")

    row = train.summarize_run(spec, TIER, targets, trainable, elapsed, generate.peak_vram_gb())
    row["final_loss"] = round(result.training_loss, 4)
    row["mask_mode"] = mask_mode
    row["max_steps"] = max_steps
    report.append_row(row, results_dir=ROOT / "results")

    del model, trainer
    generate.free_memory()
    return row


def score_rank(key: str, target: list, load_in_4bit: bool = False) -> dict:
    model, tok = generate.load_base(TIER, load_in_4bit=load_in_4bit)
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / key))
    model.eval()
    preds, lat = generate.generate_batch(
        model, tok, [r["input"] for r in target], system=generate.NAIVE_PROMPT, label=key)
    tgt = sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(preds, target)) / len(target)
    fmt = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds)
    del model
    generate.free_memory()
    return {"target": round(tgt, 4), "format": round(fmt, 4), "latency_ms": round(lat, 1),
           "n": len(target)}


def main() -> None:
    correct_row = existing_correct_row()
    max_steps = int(correct_row["max_steps"])
    mask_mode = correct_row.get("mask_mode") or "assistant-only"
    print(f"holding max_steps={max_steps} mask_mode={mask_mode} target_modules=text-linear "
          f"lr={LORA_LR} fixed; varying r in {{8, 16(existing), 64}}")

    for r in (8, 64):
        train_rank(r, max_steps, mask_mode)

    target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    sweep = {
        "r8": score_rank("rank_r8", target),
        "r64": score_rank("rank_r64", target),
    }

    # r16 == `correct` from NB3/NB5; pull its ALREADY-SCORED target from autopsy.json
    # instead of re-running generation on an adapter we already have a score for.
    autopsy_path = ROOT / "results" / "autopsy.json"
    r16 = None
    if autopsy_path.exists():
        autopsy = json.loads(autopsy_path.read_text(encoding="utf-8"))
        r16 = next((row for row in autopsy if row["run"] == "correct"), None)
    if r16 is None:
        print("WARNING: run NB5 first so results/autopsy.json has `correct`'s target score; "
              "r16_correct left out of the sweep.")
    else:
        sweep["r16_correct"] = {"target": r16["target"], "format": r16["format"],
                                "latency_ms": r16["latency_ms"], "n": r16["n"]}
    sweep = {"r8": sweep["r8"], "r16_correct": sweep.get("r16_correct"), "r64": sweep["r64"]}

    report.write_json(sweep, "rank_sweep.json", results_dir=ROOT / "results")
    print(json.dumps(sweep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
