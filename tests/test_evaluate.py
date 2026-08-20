"""Scoring + regression-gate tests. These encode the lab's grading contract."""
import pytest
from labkit import evaluate as ev


# --- the Vietnamese normalization footgun -----------------------------------

def test_accent_folding_is_symmetric():
    """The bug this course has hit: keyword unaccented, output accented (or vice versa)."""
    assert ev.keyword_recall("Ha Noi la thu do", ["Hà Nội"]) == 1.0
    assert ev.keyword_recall("Hà Nội là thủ đô", ["Ha Noi"]) == 1.0
    assert ev.keyword_recall("Hà Nội là thủ đô", ["Hà Nội"]) == 1.0


def test_accent_folding_can_be_disabled_and_then_it_is_strict():
    assert ev.keyword_recall("Ha Noi", ["Hà Nội"], fold_accents=False) == 0.0


def test_d_stroke_folds():
    assert ev.strip_accents("Đà Nẵng") == "Da Nang"


def test_keyword_recall_is_a_fraction_not_a_boolean():
    assert ev.keyword_recall("chi co mot tu", ["mot", "hai", "ba"]) == pytest.approx(1 / 3)


def test_empty_keywords_do_not_award_free_points_by_accident():
    """Vacuous truth is intentional here, but the rubric must never ship an empty list."""
    assert ev.keyword_recall("bat ky", []) == 1.0


def test_exact_match():
    assert ev.exact_match("  Xin  CHAO ", "xin chao") == 1.0
    assert ev.exact_match("xin chao", "tam biet") == 0.0


# --- format compliance ------------------------------------------------------

def test_json_parses_bare_and_fenced():
    assert ev.json_parses('{"a": 1}') == 1.0
    assert ev.json_parses('```json\n{"a": 1}\n```') == 1.0
    assert ev.json_parses("not json at all") == 0.0


def test_required_keys_partial_credit():
    assert ev.has_required_keys('{"a":1,"b":2}', ["a", "b"]) == 1.0
    assert ev.has_required_keys('{"a":1}', ["a", "b"]) == 0.5
    assert ev.has_required_keys("broken", ["a"]) == 0.0


def test_required_keys_on_a_json_list_scores_zero_not_crash():
    assert ev.has_required_keys("[1,2,3]", ["a"]) == 0.0


# --- reasoning-trace collapse ----------------------------------------------

def test_valid_reasoning_trace_detects_collapse():
    assert ev.valid_reasoning_trace("<think>can nhac ky cang o day</think>4") == 1.0
    assert ev.valid_reasoning_trace("<think></think>4") == 0.0, "empty-think must score 0"
    assert ev.valid_reasoning_trace("4") == 0.0, "no-think must score 0"
    assert ev.valid_reasoning_trace("<think>unclosed 4") == 0.0


# --- the gate ---------------------------------------------------------------

def _g(target, regression=0.80):
    return ev.GroupScores(target=target, regression=regression, format=1.0, n=50)


def test_gate_passes_on_a_real_win():
    v = ev.regression_gate(_g(0.85), _g(0.70))
    assert v.passed and v.target_delta == pytest.approx(0.15)


def test_gate_fails_when_the_prompt_baseline_wins():
    v = ev.regression_gate(_g(0.60), _g(0.75))
    assert not v.passed
    assert any("did not beat" in r for r in v.reasons)


def test_gate_fails_on_a_tie():
    """Beating (b) requires strictly better — a tie means the fine-tune bought nothing."""
    assert not ev.regression_gate(_g(0.70), _g(0.70)).passed


def test_gate_catches_capability_regression_even_when_target_improves():
    """The headline failure: target up, everything else quietly destroyed."""
    v = ev.regression_gate(_g(0.95, regression=0.40), _g(0.70, regression=0.80))
    assert not v.passed
    assert any("regressed" in r for r in v.reasons)


def test_small_regression_inside_tolerance_still_passes():
    v = ev.regression_gate(_g(0.90, regression=0.789), _g(0.70, regression=0.80))
    assert v.passed


def test_comparison_table_row_shape():
    rows = ev.comparison_table({"base+naive": _g(0.4), "base+tuned-prompt": _g(0.7), "lora": _g(0.85)})
    assert [r["run"] for r in rows] == ["base+naive", "base+tuned-prompt", "lora"]
    assert set(rows[0]) == {"run", "target", "regression", "format", "latency_ms", "n"}


def test_timer_measures_something():
    with ev.Timer() as t:
        sum(range(10000))
    assert t.ms >= 0.0


# --- target-task scorer -----------------------------------------------------

LABEL = {"intent": "doi_tra", "urgency": "cao", "product": "tai nghe bluetooth",
         "sentiment": "tieu_cuc"}


def test_triage_perfect_and_zero():
    assert ev.triage_field_accuracy(
        '{"intent":"doi_tra","urgency":"cao","product":"tai nghe bluetooth","sentiment":"tieu_cuc"}',
        LABEL) == 1.0
    assert ev.triage_field_accuracy("khong phai json", LABEL) == 0.0


def test_triage_partial_credit():
    got = ev.triage_field_accuracy(
        '{"intent":"doi_tra","urgency":"thap","product":"sai","sentiment":"tieu_cuc"}', LABEL)
    assert got == pytest.approx(0.5)


def test_triage_accepts_fenced_json_and_surrounding_prose():
    fenced = 'Day la ket qua:\n```json\n' + \
        '{"intent":"doi_tra","urgency":"cao","product":"tai nghe bluetooth","sentiment":"tieu_cuc"}' + \
        '\n```\nCam on!'
    assert ev.triage_field_accuracy(fenced, LABEL) == 1.0


def test_triage_product_folds_accents_but_categories_do_not():
    """product is copied from free text; the closed-vocabulary fields must match exactly."""
    assert ev.triage_field_accuracy(
        '{"intent":"doi_tra","urgency":"cao","product":"TAI NGHE BLUETOOTH","sentiment":"tieu_cuc"}',
        LABEL) == 1.0
    near_miss = ev.triage_field_accuracy(
        '{"intent":"doi tra","urgency":"cao","product":"tai nghe bluetooth","sentiment":"tieu_cuc"}',
        LABEL)
    assert near_miss == pytest.approx(0.75), "'doi tra' != 'doi_tra' — closed vocab is strict"


def test_triage_missing_keys_score_zero_for_those_fields():
    assert ev.triage_field_accuracy('{"intent":"doi_tra"}', LABEL) == pytest.approx(0.25)


def test_format_and_target_scorers_agree_on_what_json_is():
    """Regression: has_required_keys used a stricter parser than triage_field_accuracy,
    so prose-wrapped JSON scored on target but 0 on format."""
    prose = 'Day la ket qua phan loai: {"intent":"doi_tra","urgency":"cao",' \
            '"product":"tai nghe bluetooth","sentiment":"tieu_cuc"} - hy vong giup duoc ban.'
    assert ev.triage_field_accuracy(prose, LABEL) == 1.0
    assert ev.has_required_keys(prose, ev.TRIAGE_KEYS) == 1.0, \
        "format scorer must recognise the same JSON the target scorer parsed"


def test_format_scorer_still_rejects_non_json():
    assert ev.has_required_keys("khong co json o day", ev.TRIAGE_KEYS) == 0.0
