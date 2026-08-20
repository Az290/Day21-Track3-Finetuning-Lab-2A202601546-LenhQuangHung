"""The loss-mask tests. If these fail, the lab teaches the wrong thing."""
import pytest
from fake_tokenizer import FakeTokenizer
from labkit import data


ANSWER = "Ha Noi la thu do Viet Nam."
QUESTION = "Thu do Viet Nam la gi?"
MSGS = [
    {"role": "user", "content": QUESTION},
    {"role": "assistant", "content": ANSWER},
]
THINK_MSGS = [
    {"role": "user", "content": "2+2?"},
    {"role": "assistant", "content": "<think>cong hai so</think>4"},
]


def test_assistant_only_supervises_the_answer_not_the_question():
    tok = FakeTokenizer()
    ex = data.build_example(tok, MSGS, mask_mode="assistant-only")
    supervised = data.decode_supervised(tok, ex)
    assert ANSWER in supervised
    assert QUESTION not in supervised, "the prompt leaked into the loss"
    assert 0 < ex.n_supervised < ex.n_total


def test_everything_mode_is_the_classic_bug():
    """`everything` supervises the prompt too — the §16 'model writes your question back'."""
    tok = FakeTokenizer()
    ex = data.build_example(tok, MSGS, mask_mode="everything")
    supervised = data.decode_supervised(tok, ex)
    assert QUESTION in supervised and ANSWER in supervised
    assert ex.n_supervised == ex.n_total
    assert ex.supervised_fraction == 1.0


def test_masked_think_excludes_the_reasoning_block():
    tok = FakeTokenizer()
    full = data.build_example(tok, THINK_MSGS, mask_mode="assistant-only")
    masked = data.build_example(tok, THINK_MSGS, mask_mode="masked-think")
    assert "cong hai so" in data.decode_supervised(tok, full)
    assert "cong hai so" not in data.decode_supervised(tok, masked)
    assert "4" in data.decode_supervised(tok, masked)
    assert masked.n_supervised < full.n_supervised


def test_response_only_also_drops_reasoning():
    tok = FakeTokenizer()
    ex = data.build_example(tok, THINK_MSGS, mask_mode="response-only")
    sup = data.decode_supervised(tok, ex)
    assert "cong hai so" not in sup and "4" in sup


def test_masking_is_a_partition():
    """Every token is either supervised or masked — never both, never neither."""
    tok = FakeTokenizer()
    ex = data.build_example(tok, MSGS, mask_mode="assistant-only")
    kept = sum(1 for l in ex.labels if l != data.IGNORE_INDEX)
    dropped = sum(1 for l in ex.labels if l == data.IGNORE_INDEX)
    assert kept + dropped == len(ex.input_ids) == ex.n_total
    assert len(ex.labels) == len(ex.input_ids)


def test_supervised_labels_equal_their_input_ids():
    tok = FakeTokenizer()
    ex = data.build_example(tok, MSGS, mask_mode="assistant-only")
    for tid, lab in zip(ex.input_ids, ex.labels):
        assert lab in (data.IGNORE_INDEX, tid), "label must be -100 or the token itself"


def test_multi_turn_supervises_every_assistant_turn():
    tok = FakeTokenizer()
    msgs = [
        {"role": "user", "content": "cau mot"},
        {"role": "assistant", "content": "dap mot"},
        {"role": "user", "content": "cau hai"},
        {"role": "assistant", "content": "dap hai"},
    ]
    ex = data.build_example(tok, msgs, mask_mode="assistant-only")
    sup = data.decode_supervised(tok, ex)
    assert "dap mot" in sup and "dap hai" in sup
    assert "cau mot" not in sup and "cau hai" not in sup


def test_truncation_keeps_labels_aligned():
    tok = FakeTokenizer()
    ex = data.build_example(tok, MSGS, max_length=20, mask_mode="assistant-only")
    assert len(ex.input_ids) == len(ex.labels) == 20
    assert ex.n_total == 20


def test_thinking_survives_detects_a_stripping_template():
    good = data.thinking_survives(FakeTokenizer(strip_thinking=False))
    bad = data.thinking_survives(FakeTokenizer(strip_thinking=True))
    assert good["ok"] is True and good["body_present"] is True
    assert bad["ok"] is False, "a template that deletes <think> must be flagged"
    assert "STRIPS" in bad["verdict"]


def test_unstable_template_raises_instead_of_masking_wrong():
    tok = FakeTokenizer(prefix_unstable=True)
    with pytest.raises(data.TemplateNotPrefixStable):
        data.build_example(tok, MSGS, mask_mode="assistant-only")


def test_bad_mask_mode_rejected():
    with pytest.raises(ValueError):
        data.build_example(FakeTokenizer(), MSGS, mask_mode="all-of-it")


def test_token_stats_percentiles_and_pow2():
    st = data.token_stats(list(range(1, 101)))
    assert st["n"] == 100 and st["p50"] == 50 and st["max"] == 100
    assert st["p95"] >= st["p50"]
    assert st["suggested_max_length"] in (256, 512, 1024, 2048)


def test_token_stats_empty():
    assert data.token_stats([]) == {"n": 0}


def test_to_messages_normalizes_alpaca_and_chat():
    m = data.to_messages({"instruction": "dich cau nay", "input": "hello", "output": "xin chao"})
    assert m[0]["role"] == "user" and "dich cau nay" in m[0]["content"] and "hello" in m[0]["content"]
    assert m[1] == {"role": "assistant", "content": "xin chao"}
    passthrough = [{"role": "user", "content": "x"}]
    assert data.to_messages({"messages": passthrough}) == passthrough


def test_split_is_deterministic_and_disjoint():
    recs = list(range(100))
    a1, b1 = data.split(recs, seed=42)
    a2, b2 = data.split(recs, seed=42)
    assert a1 == a2 and b1 == b2
    assert len(a1) == 90 and len(b1) == 10
    assert not (set(a1) & set(b1))
    assert sorted(a1 + b1) == recs
