import pathlib

from labkit import report


def test_append_unions_new_columns_instead_of_dropping_them(tmp_path: pathlib.Path):
    report.append_row({"run": "a", "target": 0.5}, results_dir=tmp_path)
    report.append_row({"run": "b", "target": 0.7, "vram": 9.1}, results_dir=tmp_path)
    rows = report.read_rows(results_dir=tmp_path)
    assert len(rows) == 2
    assert rows[0]["run"] == "a" and rows[0]["vram"] == ""
    assert rows[1]["vram"] == "9.1"


def test_markdown_table_shape():
    md = report.markdown_table([{"run": "x", "target": 1}], ["run", "target"])
    assert md.splitlines()[0] == "| run | target |"
    assert md.splitlines()[1] == "|---|---|"


def test_markdown_table_empty():
    assert "no rows" in report.markdown_table([])


def test_write_json_keeps_vietnamese_readable(tmp_path: pathlib.Path):
    p = report.write_json({"tp": "Hà Nội"}, "x.json", results_dir=tmp_path)
    assert "Hà Nội" in p.read_text(encoding="utf-8")
