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
