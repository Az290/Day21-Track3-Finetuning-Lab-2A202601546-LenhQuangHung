"""Thin, version-defensive wrapper around TRL's SFTTrainer.

Why a wrapper at all: the lab this replaces shipped a page of monkey-patches for
`tokenizer=` vs `processing_class=`, `evaluation_strategy` vs `eval_strategy`, and a
`packing=False` workaround. Those were all *pre-1.0 TRL fossils*. Rather than ship a
new set of fossils, this module asks the installed TRL what it accepts and drops what
it does not — so the lab keeps running when TRL moves again, and tells you what it
dropped instead of failing at step 0.

The defaults encode the deck:
  * `target_modules` = text-decoder linear layers (§10.2, corrected for the vision tower)
  * `learning_rate`  = ~10x the full-FT scale (§10.3)
  * effective batch  < 32 (§10.4)
  * `packing` + `padding_free` together (§13.3 — packing is only free when sequence
    boundaries are respected)
  * `loss_type="chunked_nll"` (TRL >= 1.7 default; ~30-50% less VRAM)
"""
from __future__ import annotations

import dataclasses
import inspect
import warnings

from .config import MAX_EFFECTIVE_BATCH, LoraSpec, Tier


def _accepted_fields(cls) -> set[str]:
    """Field names `cls` will accept, whether it is a dataclass or a plain __init__."""
    names: set[str] = set()
    if dataclasses.is_dataclass(cls):
        names |= {f.name for f in dataclasses.fields(cls)}
    try:
        sig = inspect.signature(cls.__init__)
        names |= {p for p in sig.parameters if p != "self"}
    except (TypeError, ValueError):  # pragma: no cover - builtins
        pass
    return names


def filter_kwargs(cls, desired: dict, *, label: str = "config") -> tuple[dict, list[str]]:
    """Keep only the kwargs `cls` accepts. Returns (kept, dropped_names).

    Dropped keys are reported, never swallowed: if your TRL is too old for
    `padding_free`, you want to *know* that packing is now unsafe, not discover it in
    a loss curve.
    """
    ok = _accepted_fields(cls)
    if not ok:                                    # pragma: no cover - defensive
        return dict(desired), []
    kept = {k: v for k, v in desired.items() if k in ok}
    dropped = sorted(set(desired) - set(kept))
    if dropped:
        warnings.warn(
            f"{label}: installed {cls.__name__} does not accept {dropped}. "
            "They were dropped. Check your TRL/PEFT version against requirements.txt.",
            RuntimeWarning,
            stacklevel=2,
        )
    return kept, dropped


def sft_config_kwargs(
    tier: Tier,
    spec: LoraSpec,
    output_dir: str,
    *,
    max_steps: int | None = None,
    num_train_epochs: float = 1.0,
    mask_mode: str = "assistant-only",
    seed: int = 42,
) -> dict:
    """The SFTConfig we *want*. Pass through `filter_kwargs` before constructing."""
    if tier.effective_batch > MAX_EFFECTIVE_BATCH:
        raise ValueError(
            f"effective batch {tier.effective_batch} exceeds {MAX_EFFECTIVE_BATCH} "
            "(deck §10.4: LoRA tolerates large batches worse than full FT, and raising "
            "rank does not fix it). Lower grad_accum for this tier."
        )
    kw = dict(
        output_dir=output_dir,
        max_length=tier.max_length,               # NOT max_seq_length (renamed in TRL v1)
        per_device_train_batch_size=tier.per_device_batch,
        gradient_accumulation_steps=tier.grad_accum,
        learning_rate=spec.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=num_train_epochs,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=seed,
        bf16=True,
        packing=True,
        padding_free=True,                        # §13.3: packing is safe only with this
        loss_type="chunked_nll",                  # TRL >= 1.7 default; big VRAM saving
        gradient_checkpointing=True,
    )
    # Only one of these should ever be set; assistant_only_loss is the multi-turn form.
    if mask_mode == "everything":
        kw["completion_only_loss"] = False
    else:
        kw["assistant_only_loss"] = True
    if max_steps is not None:
        kw["max_steps"] = max_steps
    return kw


def lora_config_kwargs(spec: LoraSpec, target_modules: list[str]) -> dict:
    if spec.r is None or spec.alpha is None:
        raise ValueError(
            f"spec {spec.key!r} has an unresolved rank. Call "
            "`spec.resolved(modeling.matched_rank(...))` first — see NB4."
        )
    return dict(
        r=spec.r,
        lora_alpha=spec.alpha,                    # §9.3 invariant: alpha = 2r
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def summarize_run(spec: LoraSpec, tier: Tier, target_modules: list[str],
                  trainable: int, seconds: float, peak_vram_gb: float | None) -> dict:
    """One row of `results/runs.csv`. Same shape for every run so runs are comparable."""
    return {
        "run": spec.key,
        "label": spec.label,
        "tier": tier.name,
        "model": tier.model_id,
        "placement": spec.target,
        "n_target_modules": len(target_modules),
        "r": spec.r,
        "lora_alpha": spec.alpha,
        "learning_rate": spec.lr,
        "load_in_4bit": spec.load_in_4bit,
        "trainable_params": trainable,
        "train_seconds": round(seconds, 1),
        "peak_vram_gb": None if peak_vram_gb is None else round(peak_vram_gb, 2),
    }
