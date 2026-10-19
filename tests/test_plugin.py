"""No-GPU checks for plugin.py's validation helpers, renderer resolution, and adapter-name
resolution. Does NOT touch vLLM or a model -- the actual engine call
(`_answer_question`/`_fetch_label_logprobs`) needs a running vLLM server and is covered by
integration testing against a real container instead.

  python3 -m pytest tests/test_plugin.py -v -s --tb=short
"""

import asyncio
import json as _json
import math

import pytest
from pydantic import ValidationError as PydanticValidationError

from granite_switch_systemone import plugin as sp

# --------------------------------------------------------------------------------------------
# Question / SystemOneRequest schema validation (jev's 422s)
# --------------------------------------------------------------------------------------------


def test_question_choice_valid():
    q = sp.Question(type="choice", instructions="Which?", criteria={"a": "A", "b": "B"})
    assert q.criteria == {"a": "A", "b": "B"}


def test_question_choice_single_option_accepted():
    """jev's real OpenAPI spec (ChoiceQuestion.criteria) says minProperties=1, not 2."""
    q = sp.Question(type="choice", instructions="Which?", criteria={"a": "A"})
    assert q.criteria == {"a": "A"}


def test_question_choice_zero_options_rejected():
    with pytest.raises(Exception):
        sp.Question(type="choice", instructions="Which?", criteria={})


def test_question_choice_too_many_options_rejected():
    criteria = {f"k{i}": f"d{i}" for i in range(256)}
    with pytest.raises(Exception):
        sp.Question(type="choice", instructions="Which?", criteria=criteria)


def test_question_score_bounds():
    """jev's real OpenAPI spec (ScoreQuestion.criteria) says minItems=1, not 2."""
    sp.Question(type="score", instructions="How much?", criteria=["lo", "hi"])
    sp.Question(type="score", instructions="How much?", criteria=["only one"])
    with pytest.raises(Exception):
        sp.Question(type="score", instructions="How much?", criteria=[])
    with pytest.raises(Exception):
        sp.Question(type="score", instructions="How much?", criteria=[str(i) for i in range(11)])


def test_question_instructions_optional():
    """jev's real OpenAPI spec: `instructions` is NOT in any question type's
    `required` list -- a question can rely on well-named criteria alone."""
    q = sp.Question(type="choice", criteria={"a": "A", "b": "B"})
    assert q.instructions is None


def test_question_noul_criteria_optional():
    q = sp.Question(type="noul", instructions="Is it?")
    assert q.criteria is None
    q2 = sp.Question(type="noul", instructions="Is it?", criteria={"true": "Yes"})
    assert q2.criteria == {"true": "Yes"}


def test_system_one_request_requires_nonempty_questions():
    q = sp.Question(type="noul", instructions="Is it?")
    sp.SystemOneRequest(state="s", model="m", questions={"q1": q})
    with pytest.raises(Exception):
        sp.SystemOneRequest(state="s", model="m", questions={})


def test_question_instructions_accepts_object_and_array():
    q1 = sp.Question(type="noul", instructions={"question": "Is it urgent?", "ref": "ticket #42"})
    assert q1.instructions["question"] == "Is it urgent?"
    q2 = sp.Question(type="noul", instructions=["Is it urgent?", "context line"])
    assert q2.instructions == ["Is it urgent?", "context line"]


def test_question_choice_criteria_values_accept_object_array_and_null():
    q = sp.Question(
        type="choice",
        instructions="Which?",
        criteria={"a": "plain string", "b": {"desc": "an object"}, "c": ["an", "array"], "d": None},
    )
    assert q.criteria["d"] is None


def test_question_score_criteria_entries_accept_object_and_array():
    q = sp.Question(
        type="score",
        instructions="How much?",
        criteria=["plain string", {"desc": "an object"}, ["an", "array"]],
    )
    assert q.criteria[1] == {"desc": "an object"}


def test_question_score_criteria_entry_cannot_be_null():
    """Unlike choice, jev does not document score levels as nullable."""
    with pytest.raises(Exception):
        sp.Question(type="score", instructions="How much?", criteria=["lo", None])


def test_validation_error_response_matches_jev_shape():
    """jev's real OpenAPI spec documents the 422 body as
    `{"detail": [{"loc", "msg", "type", ...}]}` -- pydantic's own
    ValidationError.errors() shape already."""
    try:
        sp.SystemOneRequest(state="s", model="m", questions={})
    except PydanticValidationError as e:
        resp = sp._validation_error_response(e)
    else:
        raise AssertionError("expected a validation error")
    assert resp.status_code == 422
    body = _json.loads(resp.body)
    assert "detail" in body and isinstance(body["detail"], list) and body["detail"]
    item = body["detail"][0]
    assert {"loc", "msg", "type"} <= item.keys()
    assert isinstance(item["loc"], list)


# --------------------------------------------------------------------------------------------
# Body-level validation: nesting depth, finite numbers, BOM, Content-Encoding/Length
# --------------------------------------------------------------------------------------------


def test_check_json_value_nesting_depth_ok_at_limit():
    value = {}
    cur = value
    for _ in range(sp.MAX_NESTING_DEPTH - 1):
        cur["x"] = {}
        cur = cur["x"]
    sp._check_json_value(value)  # must not raise


def test_check_json_value_rejects_excess_nesting():
    value = {}
    cur = value
    for _ in range(sp.MAX_NESTING_DEPTH + 5):
        cur["x"] = {}
        cur = cur["x"]
    with pytest.raises(ValueError, match="nesting"):
        sp._check_json_value(value)


def test_check_json_value_rejects_non_finite_floats():
    with pytest.raises(ValueError, match="finite"):
        sp._check_json_value({"a": [1.0, float("nan")]})
    with pytest.raises(ValueError, match="finite"):
        sp._check_json_value({"a": float("inf")})


def test_check_json_value_accepts_finite_numbers():
    sp._check_json_value({"a": [1, 2.5, -3, {"b": 0.0}]})  # must not raise


def test_parse_strict_json_rejects_bom():
    with pytest.raises(ValueError, match="BOM"):
        sp._parse_strict_json(b"\xef\xbb\xbf{}")


def test_parse_strict_json_rejects_non_finite_literals():
    with pytest.raises(ValueError, match="non-finite"):
        sp._parse_strict_json(b'{"a": Infinity}')
    with pytest.raises(ValueError, match="non-finite"):
        sp._parse_strict_json(b'{"a": NaN}')


def test_parse_strict_json_rejects_overflowing_literal():
    with pytest.raises(ValueError, match="finite"):
        sp._parse_strict_json(b'{"a": 1e400}')


def test_parse_strict_json_accepts_valid_body():
    data = sp._parse_strict_json(b'{"state": "s", "model": "m", "questions": {}}')
    assert data == {"state": "s", "model": "m", "questions": {}}


def test_check_content_headers_rejects_encoding():
    result = sp._check_content_headers({"content-encoding": "gzip"})
    assert result is not None and result[0] == 415


def test_check_content_headers_rejects_oversized_length():
    result = sp._check_content_headers({"content-length": str(sp.MAX_BODY_BYTES + 1)})
    assert result is not None and result[0] == 413


def test_check_content_headers_accepts_plain_request():
    assert sp._check_content_headers({"content-length": "10"}) is None
    assert sp._check_content_headers({}) is None


# --------------------------------------------------------------------------------------------
# softmax / confidence (jev's own formulas, not our invention)
# --------------------------------------------------------------------------------------------


def test_softmax_sums_to_one_and_prefers_higher_logprob():
    probs = sp._softmax([math.log(0.1), math.log(0.9)])
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[1] > probs[0]


@pytest.mark.parametrize(
    "fn_name,probs,expected",
    [
        ("_score_confidence", [0.2] * 5, 0.0),
        ("_score_confidence", [0.04] * 5, 0.0),
        ("_score_confidence", [0.01, 0.02, 0.07, 0.3, 0.6], 0.55),
        ("_choice_confidence", [0.5, 0.5], 0.0),
        ("_choice_confidence", [0.2, 0.2], 0.0),
        ("_choice_confidence", [0.82, 0.18], 0.64),
        ("_score_confidence", [1.0], 1.0),
        ("_choice_confidence", [1.0], 1.0),
    ],
)
def test_confidence_matches_official_test_vectors(fn_name, probs, expected):
    """Pinned against the official typesafe-ai/system-one-adapter-python's own test
    vectors (tests/utils/test_confidence_metrics.py), not values we made up."""
    fn = getattr(sp, fn_name)
    assert fn(probs) == pytest.approx(expected)


# --------------------------------------------------------------------------------------------
# _chunk_label_ids: pure list math for a choice with more than MAX_LABEL_OPTIONS options.
# --------------------------------------------------------------------------------------------


def test_chunk_label_ids_single_chunk_under_the_cap():
    ids = list(range(50))
    assert sp._chunk_label_ids(ids) == [ids]


def test_chunk_label_ids_exactly_at_the_cap():
    ids = list(range(sp.MAX_LABEL_OPTIONS))
    assert sp._chunk_label_ids(ids) == [ids]


def test_chunk_label_ids_splits_above_the_cap():
    ids = list(range(200))
    chunks = sp._chunk_label_ids(ids)
    assert len(chunks) == 2
    assert chunks[0] == ids[:128]
    assert chunks[1] == ids[128:]
    assert [i for c in chunks for i in c] == ids


def test_chunk_label_ids_at_jevs_max_choice_options():
    """jev's own ceiling for choice: 255 options -> ceil(255/128) = 2 chunks."""
    ids = list(range(255))
    chunks = sp._chunk_label_ids(ids)
    assert len(chunks) == 2
    assert sum(len(c) for c in chunks) == 255


# --------------------------------------------------------------------------------------------
# _cancel_pending: pure asyncio, no vLLM needed.
# --------------------------------------------------------------------------------------------


def test_cancel_pending_cancels_unfinished_tasks_only():
    async def scenario():
        async def never_finishes():
            await asyncio.sleep(1000)

        async def finishes_immediately():
            return "done"

        running = asyncio.ensure_future(never_finishes())
        already_done = asyncio.ensure_future(finishes_immediately())
        await asyncio.sleep(0)
        assert already_done.done()
        assert not running.done()

        await sp._cancel_pending([running, already_done])

        assert running.cancelled()
        assert already_done.done() and not already_done.cancelled()

    asyncio.run(scenario())


def test_cancel_pending_is_a_noop_on_already_finished_tasks():
    async def scenario():
        async def finishes_immediately():
            return "done"

        tasks = [asyncio.ensure_future(finishes_immediately()) for _ in range(3)]
        await asyncio.gather(*tasks)
        await sp._cancel_pending(tasks)
        assert all(t.done() and not t.cancelled() for t in tasks)

    asyncio.run(scenario())


# --------------------------------------------------------------------------------------------
# _resolve_adapter_name: cross-checks a candidate decision-adapter name against the loaded
# checkpoint's REAL GraniteSwitchConfig.adapter_names, never a hardcoded name list.
# --------------------------------------------------------------------------------------------


class _FakeHfConfig:
    def __init__(self, adapter_names):
        self.adapter_names = adapter_names


class _FakeModelConfig:
    def __init__(self, adapter_names):
        self.hf_config = _FakeHfConfig(adapter_names)


class _FakeChat:
    def __init__(self, adapter_names):
        self.model_config = _FakeModelConfig(adapter_names)


def test_resolve_adapter_name_auto_resolves_single_adapter():
    chat = _FakeChat(["decision"])
    assert sp._resolve_adapter_name(chat) == "decision"


def test_resolve_adapter_name_requires_env_var_with_multiple_adapters():
    chat = _FakeChat(["guardian-core", "decision"])
    with pytest.raises(RuntimeError, match="SYSTEMONE_ADAPTER_NAME"):
        sp._resolve_adapter_name(chat)


def test_resolve_adapter_name_requires_env_var_with_zero_adapters():
    chat = _FakeChat([])
    with pytest.raises(RuntimeError, match="SYSTEMONE_ADAPTER_NAME"):
        sp._resolve_adapter_name(chat)


def test_resolve_adapter_name_honors_env_override(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_ADAPTER_NAME", "decision")
    chat = _FakeChat(["guardian-core", "decision"])
    assert sp._resolve_adapter_name(chat) == "decision"


def test_resolve_adapter_name_env_override_still_validated(monkeypatch):
    """An override naming an adapter the checkpoint doesn't have must still fail
    loudly, not silently fall back to anything."""
    monkeypatch.setenv("SYSTEMONE_ADAPTER_NAME", "typo-adapter-name")
    chat = _FakeChat(["decision"])
    with pytest.raises(RuntimeError, match="typo-adapter-name"):
        sp._resolve_adapter_name(chat)


# --------------------------------------------------------------------------------------------
# _load_renderer: dynamic "module:function" resolution -- this is the actual decoupling
# mechanism (plugin.py never imports a renderer module directly).
# --------------------------------------------------------------------------------------------


def test_load_renderer_default_resolves_to_this_packages_own_renderer():
    import os as _os

    _os.environ.pop(sp._RENDERER_ENV, None)
    fn = sp._load_renderer()
    from granite_switch_systemone.renderer import render

    assert fn is render


def test_load_renderer_rejects_a_path_without_a_colon(monkeypatch):
    monkeypatch.setenv(sp._RENDERER_ENV, "not_a_valid_path")
    with pytest.raises(RuntimeError, match="module:function"):
        sp._load_renderer()


def test_load_renderer_rejects_a_missing_attribute(monkeypatch):
    monkeypatch.setenv(sp._RENDERER_ENV, "granite_switch_systemone.renderer:nonexistent_fn")
    with pytest.raises(RuntimeError, match="nonexistent_fn"):
        sp._load_renderer()


def test_load_renderer_resolves_a_custom_path(monkeypatch):
    monkeypatch.setenv(sp._RENDERER_ENV, "math:sqrt")
    fn = sp._load_renderer()
    import math as _math

    assert fn is _math.sqrt


# --------------------------------------------------------------------------------------------
# _enforce_context_limits: its ValueError messages must contain a phrase from a third-party
# benchmark harness's own capacity-rejection marker list, or a legitimately-oversized request
# gets scored as a hard "error" instead of "unsupported" by that harness.
# --------------------------------------------------------------------------------------------

_CAPACITY_MARKERS = (
    "maximum context length",
    "maximum model length",
    "longer than the maximum model length",
    "context window",
    "too many tokens",
)


class _FakeTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_enforce_context_limits_per_question_message_matches_a_capacity_marker():
    tok = _FakeTok()
    q = sp.Question(type="noul", instructions="word " * 100_000)
    with pytest.raises(ValueError) as exc_info:
        sp._enforce_context_limits(tok, 32768, False, "short state", {"q": q})
    assert any(marker in str(exc_info.value) for marker in _CAPACITY_MARKERS)


def test_enforce_context_limits_total_message_matches_a_capacity_marker():
    tok = _FakeTok()
    questions = {f"q{i}": sp.Question(type="noul", instructions="word " * 5000) for i in range(10)}
    with pytest.raises(ValueError) as exc_info:
        sp._enforce_context_limits(tok, 32768, False, "short state", questions)
    assert any(marker in str(exc_info.value) for marker in _CAPACITY_MARKERS)


def test_enforce_context_limits_accepts_small_request():
    tok = _FakeTok()
    q = sp.Question(type="noul", instructions="a short question")
    sp._enforce_context_limits(tok, 32768, False, "a short state", {"q": q})  # must not raise


# --------------------------------------------------------------------------------------------
# _answer_question's usage accounting, with create_chat_completion's responses faked.
# --------------------------------------------------------------------------------------------


def _fake_chat_response(label_ids, prompt_tokens, cached_tokens, created_cache_tokens):
    from types import SimpleNamespace as NS

    top = [NS(token=f"token_id:{tid}", logprob=-float(i)) for i, tid in enumerate(label_ids)]
    details = (
        None if cached_tokens is None else NS(cached_tokens=cached_tokens, created_cache_tokens=created_cache_tokens)
    )
    return NS(
        choices=[NS(logprobs=NS(content=[NS(top_logprobs=top)]))],
        usage=NS(prompt_tokens=prompt_tokens, completion_tokens=1, prompt_tokens_details=details),
    )


@pytest.mark.parametrize(
    ("first_chunk_cache", "expected_cached", "expected_written"),
    [((40, 60), 40, 60), ((None, None), 0, 0)],
    ids=["details reported", "details disabled"],
)
def test_answer_question_counts_a_chunked_prompt_and_its_cache_once(
    monkeypatch, first_chunk_cache, expected_cached, expected_written
):
    """A 200-option choice is two engine calls over one prompt: input and cache counts
    come from the first chunk only (the second chunk's cache hit is the reuse chunking
    relies on, not more input); output tokens are one per call."""
    calls = []

    async def fake_fetch(chat, served_model, adapter_name, msgs, label_ids_chunk):
        calls.append(label_ids_chunk)
        cached, written = first_chunk_cache if len(calls) == 1 else (100, 0)
        return _fake_chat_response(label_ids_chunk, 100, cached, written)

    monkeypatch.setattr(sp, "_fetch_label_logprobs", fake_fetch)
    criteria = {f"o{i}": f"option {i}" for i in range(200)}
    q = sp.Question(type="choice", instructions="Which?", criteria=criteria)
    renderer = lambda tok, state, instructions, descs: ([], list(range(1000, 1000 + len(descs))))  # noqa: E731

    _, answer, usage = asyncio.run(sp._answer_question("q", q, "state", None, None, "m", "decision", renderer))

    assert len(calls) == 2
    assert answer["type"] == "choice"
    assert usage == sp.TokenUsage(
        input_tokens=100, output_tokens=2, cached_tokens=expected_cached, cache_write_tokens=expected_written
    )
