#!/usr/bin/env python3
"""Where does each parameter's dtype actually come from?

Run this when a training run dies with a dtype error. The one that motivated it:

    NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
                         not implemented for 'BFloat16'

raised from fp16's GradScaler on a T4 -- a GPU with no bfloat16 at all. Something in
the 4-bit path was handing bf16 *gradients* to a scaler that only understands fp16.

The lab's argument is that you measure instead of assume, so this prints the dtypes
rather than reasoning about which library set them:

    python scripts/probe_precision.py            # both paths
    python scripts/probe_precision.py --4bit     # just the QLoRA path
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def counts(params) -> str:
    c = collections.Counter(str(p.dtype).replace("torch.", "") for p in params)
    return "  ".join(f"{k}={v}" for k, v in sorted(c.items())) or "(none)"


def probe(load_in_4bit: bool) -> None:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from labkit import generate, modeling
    from labkit.config import SPECS, get_tier

    tier = get_tier()
    tag = "4-bit (QLoRA contrast)" if load_in_4bit else "16-bit (default path)"
    print(f"\n{'=' * 68}\n{tag} — {tier.model_id}\n{'=' * 68}")

    model, _ = generate.load_base(tier, load_in_4bit=load_in_4bit)
    print("config.dtype        :", getattr(model.config, "dtype", None))
    print("all params          :", counts(model.parameters()))

    # TRL does this itself when you hand SFTTrainer a quantized model + peft_config.
    # Replicated here so the probe sees the same model the Trainer sees.
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("after kbit prep     :", counts(model.parameters()))

    targets = modeling.resolve_target_modules(model, SPECS["correct"].target)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets))

    trainable = [p for p in model.parameters() if p.requires_grad]
    print("TRAINABLE params    :", counts(trainable))

    # This is the exact condition that raises. fp16's GradScaler calls
    # _amp_foreach_non_finite_check_and_unscale_ on every gradient, and that CUDA
    # kernel has no BFloat16 overload.
    import torch

    from labkit import device
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and p.dtype == torch.bfloat16]
    if device.precision() == "fp16" and bad:
        print(f"\n  *** {len(bad)} trainable params are bfloat16 while precision=fp16.")
        print(f"  *** GradScaler.unscale_ will raise on these. e.g. {bad[0]}")
    else:
        print("\n  ok: no bf16 trainable params under precision=" + device.precision())

    del model
    generate.free_memory()


def probe_through_trainer(key: str) -> None:
    """Build the SFTTrainer exactly as NB4 does, then report what it actually holds.

    Hand-replicating the PEFT setup (see `probe`) showed no bf16 anywhere, yet the run
    dies inside fp16's GradScaler complaining about BFloat16 gradients. So the dtype is
    introduced somewhere between "the model we pass in" and "the model TRL trains" --
    and the only way to find out which is to ask the trainer. No training happens here.
    """
    import torch
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    from datasets import Dataset

    from labkit import data, device, generate, modeling, train
    from labkit.config import SPECS, get_tier

    tier, spec = get_tier(), SPECS[key]
    print(f"\n{'=' * 68}\nthrough SFTTrainer: {key} — {spec.label}\n{'=' * 68}")

    model, tok = generate.load_base(tier, load_in_4bit=spec.load_in_4bit)
    rows = [json_line for json_line in _train_rows()]
    ds = Dataset.from_list(data.to_training_dataset(tok, rows, max_length=tier.max_length))

    targets = modeling.resolve_target_modules(model, spec.target)
    if spec.r is None:
        base = modeling.resolve_target_modules(model, "text-linear")
        spec = spec.resolved(modeling.matched_rank(model, base, SPECS["correct"].r, targets))

    want = train.sft_config_kwargs(tier, spec, "/tmp/probe", max_steps=1)
    sft, _ = train.filter_kwargs(SFTConfig, want, label="SFTConfig")
    lora, _ = train.filter_kwargs(LoraConfig, train.lora_config_kwargs(spec, targets),
                                  label="LoraConfig")
    print("fp16 =", sft.get("fp16"), " bf16 =", sft.get("bf16"),
          " precision() =", device.precision())

    trainer = SFTTrainer(model=model, args=SFTConfig(**sft), train_dataset=ds,
                         processing_class=tok, peft_config=LoraConfig(**lora))

    tm = trainer.model
    trainable = [(n, p) for n, p in tm.named_parameters() if p.requires_grad]
    print("TRAINABLE in trainer:", counts(p for _, p in trainable))
    bad = [n for n, p in trainable if p.dtype == torch.bfloat16]
    if bad:
        print(f"\n  *** {len(bad)} of {len(trainable)} trainable params are bfloat16.")
        print("  *** GradScaler.unscale_ has no BFloat16 kernel -> NotImplementedError.")
        for n in bad[:4]:
            print("      ", n)
    else:
        print("\n  no bf16 trainable params — the dtype comes from somewhere else")

    scaler = getattr(getattr(trainer, "accelerator", None), "scaler", None)
    print("accelerator.scaler   :", type(scaler).__name__ if scaler else None)


def _train_rows():
    import json
    p = ROOT / "data" / "split" / "train.jsonl"
    if not p.exists():
        p = ROOT / "data" / "train_seed.jsonl"
    with p.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 32:            # a handful is plenty; nothing is trained
                break
            if line.strip():
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--4bit", dest="only_4bit", action="store_true")
    ap.add_argument("--16bit", dest="only_16bit", action="store_true")
    ap.add_argument("--trainer", metavar="RUN",
                    help="build NB4's SFTTrainer for RUN (e.g. qlora) and dump dtypes")
    args = ap.parse_args()

    from labkit import device
    print(device.banner())

    if args.trainer:
        probe_through_trainer(args.trainer)
        return 0
    if not args.only_4bit:
        probe(False)
    if not args.only_16bit:
        probe(True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
