"""Structural checks — the class of defect the behaviour tests are blind to.

Three findings in this lab (F-18, F-20, F-21) were all the same shape: a *derived or
referenced* artifact drifting from its source with nothing to catch it.

  F-18  colab/*.ipynb drifted from the build_colab.py BOOTSTRAP that generates them
  F-20  the dependency pins existed in three hand-synced copies
  F-21  the README's first instruction named a notebook that has never existed

None of them are behaviour bugs, so 84 behaviour tests said nothing. They cost zero
runtime to check.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _bootstrap() -> str:
    """BOOTSTRAP out of scripts/build_colab.py (not a package — load it by path)."""
    spec = importlib.util.spec_from_file_location(
        "build_colab", ROOT / "scripts" / "build_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BOOTSTRAP


# --- F-21: every path the docs name must exist -------------------------------
PATH_RE = re.compile(r"`([A-Za-z0-9_./-]+\.(?:ipynb|py|md|txt|jsonl|sh))`")

# Paths the docs tell the STUDENT to create. Absent from the repo by design — but listed
# explicitly so a genuine typo in one of them still fails.
STUDENT_CREATED = {"data/CUSTOM_DATASET.md"}


@pytest.mark.parametrize("doc", ["README.md", "HARDWARE-GUIDE.md", "rubric.md"])
def test_documented_paths_exist(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    missing = []
    for ref in set(PATH_RE.findall(text)):
        if "/" not in ref:                 # bare filenames are prose, not paths
            continue
        if ref.startswith(("http", "adapters/", "results/", "submission/", "data/split")):
            continue                        # produced by a run, not shipped
        if ref in STUDENT_CREATED:
            continue
        # `labkit/device.py` is how you refer to the module; it lives under src/
        if not ((ROOT / ref).exists() or (ROOT / "src" / ref).exists()):
            missing.append(ref)
    assert not missing, f"{doc} names files that do not exist: {sorted(missing)}"


# --- F-18: generated notebooks must match their generator --------------------
def test_generated_notebooks_carry_the_current_bootstrap():
    boot = _bootstrap()
    stale = []
    for nb_path in sorted((ROOT / "colab").glob("Lab21_0*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        if "".join(nb["cells"][0]["source"]).strip() != boot.strip():
            stale.append(nb_path.name)
    assert not stale, (
        f"regenerate with scripts/build_colab.py — stale bootstrap in {stale}")


# --- F-20: one source of truth for dependency pins ---------------------------
def test_bootstraps_install_from_requirements_not_a_copied_list():
    run_all = json.loads((ROOT / "colab" / "Lab21_RUN_ALL.ipynb").read_text(encoding="utf-8"))
    sources = {
        "build_colab.BOOTSTRAP": _bootstrap(),
        "Lab21_RUN_ALL cell 1": "".join(run_all["cells"][1]["source"]),
    }
    for name, src in sources.items():
        assert "requirements.txt" in src, f"{name} does not install from requirements.txt"
        # a copied list re-pins packages inline; that is exactly the drift F-20 fixed
        inline = re.findall(r'"(transformers|trl|peft|accelerate|datasets|torchao)>=', src)
        assert not inline, f"{name} re-pins {sorted(set(inline))} inline — drift risk"


# --- F-22: the autopsy must be settled on the task metric, not on train loss ---------

NB4 = ROOT / "notebooks" / "04_misconfig_autopsy.py"
NB5 = ROOT / "notebooks" / "05_evaluate_and_verdict.py"


def test_nb5_scores_every_contrast_adapter():
    """NB4 trains three misconfigured adapters, saves them, and reports `final_loss` --
    a TRAINING loss. The lab names exactly that substitution "Lỗi #3". Until this fix,
    nothing ever scored those adapters on the target task, so NB4's question "did the
    mistake lose?" was answered by the proxy metric the lab warns against."""
    src = NB5.read_text(encoding="utf-8")
    assert "CONTRAST_KEYS" in src, "NB5 must iterate the contrast runs, not just `correct`"
    assert "autopsy.json" in src, "the per-adapter task scores must be written to an artifact"


def test_contrast_adapters_are_scored_at_their_training_precision():
    """`qlora` learned against a 4-bit base. Scoring it on the fp16 base measures a
    base/adapter mismatch and labels the damage 'what QLoRA costs you'."""
    src = NB5.read_text(encoding="utf-8")
    assert "load_in_4bit=SPECS[key].load_in_4bit" in src, (
        "take the 4-bit flag from the run's own spec, never from score_adapter's default")


def test_nb4_does_not_present_train_loss_as_the_verdict():
    """The column can stay -- it is free and it is interesting. It just may not be the
    thing the student ranks the four runs by."""
    src = NB4.read_text(encoding="utf-8")
    assert "final_loss" in src
    assert "LOSS HUẤN LUYỆN" in src, "the column must be labelled as a training loss"
    assert "NB5" in src, "NB4 must point at the notebook that actually settles the ranking"
