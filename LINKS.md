# Lab 21 — Links

**GitHub repo (code, results/, submission/REPORT.md, data/split/):**
https://github.com/Az290/Day21-Track3-Finetuning-Lab-2A202601546-LenhQuangHung

**HuggingFace Hub (adapter weights + results/ + REPORT.md copy, B5 +2 bonus):**
https://huggingface.co/Hung371/lab21-2A202601546-qwen35-triage-vi

## Why the split

`adapters/correct/adapter_model.safetensors` is ~130 MB. GitHub blocks Git LFS uploads
of new objects to a public fork ("can not upload new objects to public fork"), so the
adapter weights live on HuggingFace Hub instead of in this GitHub repo. Everything else
required by `rubric.md` (all of `results/*.json` + `runs.csv`, `data/split/`,
`submission/REPORT.md`, and the adapter's small config/tokenizer files) is committed
directly to this GitHub repo, so `results/` here is what the grader should cross-check
`REPORT.md` against.
