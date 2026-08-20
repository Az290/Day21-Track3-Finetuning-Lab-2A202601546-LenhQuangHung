"""labkit — the shared harness for Day 21's fine-tuning lab.

Import order mirrors the notebooks:
    config    -> which model, which tier, which LoRA spec
    data      -> chat template, loss masking, dataset prep      (NB1, CPU-only)
    evaluate  -> four-group scoring + the regression gate        (NB2, NB5)
    modeling  -> where adapters go, and what they cost           (NB3, NB4)
    train     -> version-defensive TRL configuration             (NB3, NB4)
    report    -> results I/O so runs stay comparable
"""
__all__ = ["config", "data", "evaluate", "modeling", "train", "report"]
__version__ = "1.0.0"
