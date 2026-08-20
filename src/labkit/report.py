"""Results I/O. Every run appends one row with the same columns, so the report table
writes itself and the grader can verify your numbers are internally consistent."""
from __future__ import annotations

import csv
import json
import pathlib
from typing import Iterable

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results"


def append_row(row: dict, filename: str = "runs.csv", results_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Append `row`, creating the header from its keys on first write.

    Rows with new keys are unioned into the header rather than silently dropped —
    losing a column because run 3 measured something run 1 did not is exactly the kind
    of quiet data loss that makes a results table untrustworthy.
    """
    out_dir = results_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    existing: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            existing = list(csv.DictReader(fh))

    rows = existing + [row]
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    return path


def write_json(obj, filename: str, results_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = results_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_rows(filename: str = "runs.csv", results_dir: pathlib.Path | None = None) -> list[dict]:
    path = (results_dir or RESULTS) / filename
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def markdown_table(rows: Iterable[dict], columns: list[str] | None = None) -> str:
    """Paste-ready Markdown for REPORT.md."""
    rows = list(rows)
    if not rows:
        return "_(no rows)_"
    cols = columns or list(rows[0])
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, rule, *body])
