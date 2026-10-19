"""Conformance of /v1/decisions to OpenAI's published OpenAPI spec, checked by a JSON
Schema validator against the spec itself rather than against bounds copied from it.

tests/spec/openai_decisions.json is the `POST /decisions` request and response schemas,
with everything they reference, extracted verbatim from openai/openai-openapi (commit
and licence recorded in the file). To refresh it, re-extract from a newer commit and
update the `source` field; any new disagreement then shows up here.

Every spec-valid request must answer 200 and every 200 body must validate against
`DecisionResponse`, except the deliberate limits listed in `UNSUPPORTED`, which must
answer 400 with OpenAI's error shape. Spec-invalid requests must answer 400, except the
leniencies in `ACCEPTED_LENIENTLY`.

  python3 -m pytest tests/test_openai_spec.py -v
"""

import json
from pathlib import Path

import pytest

from granite_switch_systemone import plugin as sp

jsonschema = pytest.importorskip("jsonschema")

SPEC = json.loads((Path(__file__).parent / "spec" / "openai_decisions.json").read_text())


def _validator(name: str):
    defs = json.loads(json.dumps(SPEC["schemas"]).replace("#/components/schemas/", "#/$defs/"))
    return jsonschema.Draft202012Validator({"$ref": f"#/$defs/{name}", "$defs": defs})


REQUEST = _validator("DecisionRequest")
RESPONSE = _validator("DecisionResponse")

PREDICATE = {"type": "predicate", "instructions": "Is it urgent?"}
CHOICE = {
    "type": "choice",
    "name": "queue",
    "instructions": "Which team?",
    "choices": [{"value": "a", "description": "Team A"}, {"value": "b"}],
}
SCORE = {
    "type": "score",
    "name": "severity",
    "instructions": "How urgent?",
    "levels": [{"label": "low"}, {"label": "high", "description": "Same day"}],
}


def _body(**fields):
    return {"model": "m", "input": "The export failed.", "questions": [PREDICATE], **fields}


def _message(role, content, **fields):
    return {"role": role, "content": content, **fields}


VALID = {
    "string input": _body(),
    "empty string input": _body(input=""),
    "empty message list": _body(input=[]),
    "all three question types": _body(questions=[PREDICATE, CHOICE, SCORE]),
    "200 questions": _body(questions=[PREDICATE] * 200),
    "empty instructions": _body(questions=[{**PREDICATE, "instructions": ""}]),
    "named predicate": _body(questions=[{**PREDICATE, "name": "urgent"}]),
    "boolean choice values": _body(questions=[{**CHOICE, "choices": [{"value": True}, {"value": False}]}]),
    "255 choices": _body(questions=[{**CHOICE, "choices": [{"value": f"v{i}"} for i in range(255)]}]),
    "10 score levels": _body(questions=[{**SCORE, "levels": [{"label": str(i)} for i in range(10)]}]),
    "user message, string content": _body(input=[_message("user", "hi")]),
    "user message, explicit type": _body(input=[_message("user", "hi", type="message")]),
    "user message, input_text parts": _body(
        input=[_message("user", [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}])]
    ),
    "input_text with a cache breakpoint": _body(
        input=[
            _message("user", [{"type": "input_text", "text": "a", "prompt_cache_breakpoint": {"mode": "explicit"}}])
        ]
    ),
    "prompt cache and safety fields": _body(
        prompt_cache_key="k", prompt_cache_options={"ttl": "30m", "mode": "implicit"}, safety_identifier="u1"
    ),
    "null optional fields": _body(prompt_cache_key=None, prompt_cache_options=None, safety_identifier=None),
}

# Spec-valid requests this text decision adapter rejects on purpose (README, "Deliberate
# scope limits on this route"). Each must fail with a 400 in OpenAI's error shape.
UNSUPPORTED = {
    "image part": _body(input=[_message("user", [{"type": "input_image", "image_url": "https://x.test/y.png"}])]),
    "system message": _body(input=[_message("system", "Be terse."), _message("user", "hi")]),
    "developer message": _body(input=[_message("developer", "Be terse."), _message("user", "hi")]),
    "assistant message": _body(input=[_message("user", "hi"), _message("assistant", "hello")]),
    "function call items": _body(
        input=[
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "42"},
        ]
    ),
    "custom tool call items": _body(
        input=[
            {"type": "custom_tool_call", "call_id": "c1", "name": "t", "input": "x"},
            {"type": "custom_tool_call_output", "call_id": "c1", "output": "y"},
        ]
    ),
}

INVALID = {
    "missing model": {"input": "x", "questions": [PREDICATE]},
    "missing input": {"model": "m", "questions": [PREDICATE]},
    "no questions": _body(questions=[]),
    "201 questions": _body(questions=[PREDICATE] * 201),
    "1 choice": _body(questions=[{**CHOICE, "choices": [{"value": "a"}]}]),
    "256 choices": _body(questions=[{**CHOICE, "choices": [{"value": f"v{i}"} for i in range(256)]}]),
    "1 score level": _body(questions=[{**SCORE, "levels": [{"label": "a"}]}]),
    "11 score levels": _body(questions=[{**SCORE, "levels": [{"label": str(i)} for i in range(11)]}]),
    "predicate without instructions": _body(questions=[{"type": "predicate"}]),
    "numeric choice values": _body(questions=[{**CHOICE, "choices": [{"value": 1}, {"value": 2}]}]),
    "level without label": _body(questions=[{**SCORE, "levels": [{"description": "a"}, {"description": "b"}]}]),
    "unknown top-level field": _body(temperature=0),
    "unknown question field": _body(questions=[{**PREDICATE, "extra": 1}]),
    "unknown message role": _body(input=[_message("tool", "x")]),
    "unknown cache breakpoint mode": _body(
        input=[_message("user", [{"type": "input_text", "text": "a", "prompt_cache_breakpoint": {"mode": "x"}}])]
    ),
}

# Spec-invalid requests the plugin answers anyway: the fields are accepted and ignored,
# or a missing name is treated as one, so tightening them would only break clients.
ACCEPTED_LENIENTLY = {
    "null question name": _body(questions=[{**PREDICATE, "name": None}]),
    "unsupported cache ttl": _body(prompt_cache_options={"ttl": "1h"}),
    "cache key over 64 characters": _body(prompt_cache_key="k" * 65),
}


def _errors(validator, instance):
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in validator.iter_errors(instance)]


@pytest.fixture()
def client(monkeypatch):
    async def fake_answer(name, q, state, chat, tok, served_model, adapter_name, renderer):
        if q.type == "choice":
            keys = list(q.criteria)
            answer = {
                "type": "choice",
                "choice": keys[0],
                "probabilities": {k: 1.0 / len(keys) for k in keys},
                "confidence": 0.5,
            }
        elif q.type == "noul":
            answer = {"type": "noul", "noul": 0.9}
        else:
            n = len(q.criteria)
            answer = {
                "type": "score",
                "score": 1.0,
                "legend": {str(i): d for i, d in enumerate(q.criteria)},
                "probabilities": {str(i): 1.0 / n for i in range(n)},
                "confidence": 0.3,
            }
        return name, answer, sp.TokenUsage(input_tokens=10, output_tokens=1, cached_tokens=4)

    monkeypatch.setattr(sp, "_answer_question", fake_answer)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(sp.router)
    app.state.systemone_chat = object()
    app.state.systemone_tokenizer = lambda text, add_special_tokens=False: {"input_ids": [0] * len(str(text).split())}
    app.state.systemone_max_model_len = 8192
    app.state.systemone_served_model = "granite-switch"
    app.state.systemone_adapter_name = "decision"
    app.state.systemone_renderer = lambda *args: ([], [])
    return TestClient(app)


def test_vendored_spec_is_the_published_decisions_operation():
    assert SPEC["operation"]["operationId"] == "createDecision"
    assert SPEC["source"].startswith("https://github.com/openai/openai-openapi/blob/")


@pytest.mark.parametrize("name", VALID)
def test_spec_valid_request_is_answered_with_a_spec_valid_response(client, name):
    body = VALID[name]
    assert not _errors(REQUEST, body), "test case is not spec-valid"
    resp = client.post("/v1/decisions", json=body)
    assert resp.status_code == 200, resp.text
    assert not _errors(RESPONSE, resp.json())


@pytest.mark.parametrize("name", UNSUPPORTED)
def test_unsupported_input_is_rejected_with_openai_error_shape(client, name):
    body = UNSUPPORTED[name]
    assert not _errors(REQUEST, body), "test case is not spec-valid"
    resp = client.post("/v1/decisions", json=body)
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("name", INVALID)
def test_spec_invalid_request_is_rejected(client, name):
    body = INVALID[name]
    assert _errors(REQUEST, body), "test case is not spec-invalid"
    resp = client.post("/v1/decisions", json=body)
    assert resp.status_code == 400, resp.text
    assert set(resp.json()["error"]) == {"message", "type", "param", "code"}


@pytest.mark.parametrize("name", ACCEPTED_LENIENTLY)
def test_lenient_acceptance_still_returns_a_spec_valid_response(client, name):
    body = ACCEPTED_LENIENTLY[name]
    assert _errors(REQUEST, body), "test case is not spec-invalid"
    resp = client.post("/v1/decisions", json=body)
    assert resp.status_code == 200, resp.text
    assert not _errors(RESPONSE, resp.json())


def test_usage_reports_cached_tokens_summed_over_questions(client):
    resp = client.post("/v1/decisions", json=_body(questions=[PREDICATE, CHOICE, SCORE]))
    usage = resp.json()["usage"]
    assert usage["input_tokens_details"] == {"cached_tokens": 12, "cache_write_tokens": 0}
    assert usage["output_tokens_details"] == {"reasoning_tokens": 0}
