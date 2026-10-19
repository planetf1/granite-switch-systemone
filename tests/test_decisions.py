"""No-GPU checks for the /v1/decisions wire format: schema bounds from OpenAI's published
spec, input flattening, answer mapping -- and, the load-bearing ones, PARITY with
/v1/systemone: the translation must converge on exactly the internal (state, Question)
the systemone route builds for the equivalent request. Identical internals is what makes
both routes render byte-identical prompts and read the same label logits, so the engine
behaviour of /v1/decisions is the engine behaviour of /v1/systemone by construction. The
engine call itself needs a running server (tests/test_decisions_e2e.py, env-gated).

  python3 -m pytest tests/test_decisions.py -v -s --tb=short
"""

import pytest
from pydantic import ValidationError

from granite_switch_systemone import plugin as sp


def _predicate(**kw):
    base = {"type": "predicate", "instructions": "Is it urgent?"}
    base.update(kw)
    return base


def _choice(**kw):
    base = {
        "type": "choice",
        "instructions": "Which team?",
        "choices": [
            {"value": "payments", "description": "Payout failures"},
            {"value": "account", "description": "Login issues"},
        ],
    }
    base.update(kw)
    return base


def _score(**kw):
    base = {
        "type": "score",
        "instructions": "How urgent?",
        "levels": [
            {"label": "can wait", "description": "No rush"},
            {"label": "today", "description": "Same day"},
        ],
    }
    base.update(kw)
    return base


def _request(questions, **kw):
    base = {"model": "granite-switch", "input": "a ticket", "questions": questions}
    base.update(kw)
    return base


# --------------------------------------------------------------------------------------------
# Schema bounds (OpenAI's published spec)
# --------------------------------------------------------------------------------------------


def test_predicate_rejects_choices_and_levels():
    with pytest.raises(ValidationError, match="predicate"):
        sp.DecisionQuestion.model_validate(_predicate(choices=[{"value": "a"}, {"value": "b"}]))
    with pytest.raises(ValidationError, match="predicate"):
        sp.DecisionQuestion.model_validate(_predicate(levels=[{"label": "a"}, {"label": "b"}]))


def test_choice_bounds():
    sp.DecisionQuestion.model_validate(_choice())  # 2 is valid
    sp.DecisionQuestion.model_validate(_choice(choices=[{"value": str(i)} for i in range(255)]))  # 255 is valid
    with pytest.raises(ValidationError, match="2-255"):
        sp.DecisionQuestion.model_validate(_choice(choices=[{"value": "only"}]))
    with pytest.raises(ValidationError, match="2-255"):
        sp.DecisionQuestion.model_validate(_choice(choices=[{"value": str(i)} for i in range(256)]))


def test_choice_values_must_be_unique():
    with pytest.raises(ValidationError, match="unique"):
        sp.DecisionQuestion.model_validate(_choice(choices=[{"value": "a"}, {"value": "a"}]))


def test_boolean_string_collision_rejected():
    with pytest.raises(ValidationError, match="indistinguishable"):
        sp.DecisionQuestion.model_validate(_choice(choices=[{"value": True}, {"value": "true"}]))


def test_boolean_choice_values_accepted():
    q = sp.DecisionQuestion.model_validate(_choice(choices=[{"value": True}, {"value": False}]))
    assert [c.value for c in q.choices] == [True, False]


def test_score_bounds():
    """OpenAPI spec: levels minItems 2, maxItems 10."""
    sp.DecisionQuestion.model_validate(_score())  # 2 levels valid
    sp.DecisionQuestion.model_validate(_score(levels=[{"label": str(i)} for i in range(10)]))  # 10 valid
    with pytest.raises(ValidationError, match="2-10"):
        sp.DecisionQuestion.model_validate(_score(levels=[{"label": "only"}]))
    with pytest.raises(ValidationError, match="2-10"):
        sp.DecisionQuestion.model_validate(_score(levels=[]))
    with pytest.raises(ValidationError, match="2-10"):
        sp.DecisionQuestion.model_validate(_score(levels=[{"label": str(i)} for i in range(11)]))


def test_request_question_bounds():
    sp.DecisionRequest.model_validate(_request([_predicate()]))
    sp.DecisionRequest.model_validate(_request([_predicate()] * 200))
    with pytest.raises(ValidationError, match="1-200"):
        sp.DecisionRequest.model_validate(_request([]))
    with pytest.raises(ValidationError, match="1-200"):
        sp.DecisionRequest.model_validate(_request([_predicate()] * 201))


def test_request_ignores_sdk_optional_fields():
    req = sp.DecisionRequest.model_validate(_request([_predicate()], prompt_cache_key="k", safety_identifier="u"))
    assert req.prompt_cache_key == "k"


def test_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        sp.DecisionRequest.model_validate(_request([_predicate()], totally_unknown=1))


# --------------------------------------------------------------------------------------------
# Input flattening
# --------------------------------------------------------------------------------------------


def test_input_string_passes_through():
    assert sp._decisions_input_to_state("hello") == "hello"


def test_input_messages_flatten_with_blank_lines():
    data = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": [{"type": "input_text", "text": "second"}]},
    ]
    parsed = [sp.DecisionInputMessage.model_validate(m) for m in data]
    assert sp._decisions_input_to_state(parsed) == "first\n\nsecond"


def test_input_image_rejected():
    data = [{"role": "user", "content": [{"type": "input_image", "image_url": "http://x"}]}]
    parsed = [sp.DecisionInputMessage.model_validate(m) for m in data]
    with pytest.raises(ValueError, match="image input is not supported"):
        sp._decisions_input_to_state(parsed)


def test_input_non_user_role_rejected():
    with pytest.raises(ValidationError):
        sp.DecisionInputMessage.model_validate({"role": "assistant", "content": "x"})


# --------------------------------------------------------------------------------------------
# PARITY: /v1/decisions translation must converge on /v1/systemone's internals
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decisions_question,systemone_question",
    [
        (_predicate(), {"type": "noul", "instructions": "Is it urgent?"}),
        (
            _choice(),
            {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"payments": "Payout failures", "account": "Login issues"},
            },
        ),
        (
            _score(),
            {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["No rush", "Same day"],
            },
        ),
    ],
)
def test_translation_matches_systemone_internals(decisions_question, systemone_question):
    """The load-bearing parity check: for the equivalent request, the decisions
    translation and the systemone route's own validation must produce the same state
    string and the same internal Question. Everything downstream (_answer_question,
    the renderer, the engine call) is shared code, so identical internals means
    byte-identical prompts and identical answers."""
    decisions_req = sp.DecisionRequest.model_validate(_request([decisions_question], input="the same state"))
    systemone_req = sp.SystemOneRequest.model_validate(
        {"model": "m", "state": "the same state", "questions": {"0": systemone_question}}
    )
    state_text = sp._decisions_input_to_state(decisions_req.input)
    assert state_text == "the same state"

    key, internal, _ = sp._decisions_question_to_internal(decisions_req.questions[0], 0)
    assert key == "0"
    assert internal == systemone_req.questions["0"]


def test_translation_internal_keys_are_positional():
    """Two questions with the same (or absent) names must not collide: the internal key
    is the index, and the echoed name comes from the original question."""
    req = sp.DecisionRequest.model_validate(_request([_predicate(name="same"), _predicate(name="same")]))
    translated = [sp._decisions_question_to_internal(q, i) for i, q in enumerate(req.questions)]
    assert [key for key, _, _ in translated] == ["0", "1"]


def test_score_level_without_description_uses_label():
    q = sp.DecisionQuestion.model_validate(
        _score(levels=[{"label": "low"}, {"label": "high", "description": "urgent"}])
    )
    _, internal, extras = sp._decisions_question_to_internal(q, 0)
    assert internal.criteria == ["low", "urgent"]
    assert extras["labels"] == ["low", "high"]


# --------------------------------------------------------------------------------------------
# Answer mapping (internal jev-shaped answer -> OpenAI Decisions answer)
# --------------------------------------------------------------------------------------------


def test_predicate_answer_mapping():
    q = sp.DecisionQuestion.model_validate(_predicate(name="escalate"))
    answer = sp._internal_answer_to_decisions(q, {}, {"type": "noul", "noul": 0.89})
    assert answer == {"type": "predicate", "name": "escalate", "probability": 0.89}


def test_choice_answer_maps_typed_values():
    q = sp.DecisionQuestion.model_validate(_choice(name="ok", choices=[{"value": True}, {"value": False}]))
    _, _, extras = sp._decisions_question_to_internal(q, 0)
    internal_answer = {
        "type": "choice",
        "choice": "true",
        "probabilities": {"true": 0.8, "false": 0.2},
        "confidence": 0.6,
    }
    answer = sp._internal_answer_to_decisions(q, extras, internal_answer)
    assert answer["choice"] is True  # the ORIGINAL typed value, not the string key
    assert answer["probabilities"] == [
        {"value": True, "probability": 0.8},
        {"value": False, "probability": 0.2},
    ]
    assert answer["confidence"] == 0.6
    assert answer["name"] == "ok"


def test_choice_answer_without_name_echoes_null():
    q = sp.DecisionQuestion.model_validate(_choice())
    answer = sp._internal_answer_to_decisions(
        q,
        {"value_map": {"payments": "payments", "account": "account"}},
        {
            "type": "choice",
            "choice": "payments",
            "probabilities": {"payments": 0.9, "account": 0.1},
            "confidence": 0.8,
        },
    )
    assert answer["name"] is None
    assert answer["choice"] == "payments"


def test_score_answer_carries_original_labels():
    q = sp.DecisionQuestion.model_validate(_score(name="sev"))
    _, _, extras = sp._decisions_question_to_internal(q, 0)
    internal_answer = {
        "type": "score",
        "score": 1.2,
        "legend": {"0": "No rush", "1": "Same day"},
        "probabilities": {"0": 0.3, "1": 0.7},
        "confidence": 0.4,
    }
    answer = sp._internal_answer_to_decisions(q, extras, internal_answer)
    assert answer["score"] == 1.2
    assert answer["probabilities"] == [
        {"value": 0, "label": "can wait", "probability": 0.3},
        {"value": 1, "label": "today", "probability": 0.7},
    ]
    assert answer["confidence"] == 0.4


# --------------------------------------------------------------------------------------------
# Route-level wiring: the full /v1/decisions handler flow (registration, validation,
# translation, fan-out, answer mapping, usage) with the engine call stubbed. The
# engine path itself is covered by tests/test_decisions_e2e.py against a real server.
# --------------------------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    async def fake_answer(name, q, state, chat, tok, served_model, adapter_name, renderer):
        if q.type == "choice":
            keys = list(q.criteria)
            probs = [0.8, 0.2] + [0.0] * (len(keys) - 2)
            answer = {
                "type": "choice",
                "choice": keys[0],
                "probabilities": dict(zip(keys, probs)),
                "confidence": 0.6,
            }
        elif q.type == "noul":
            answer = {"type": "noul", "noul": 0.9}
        else:
            count = len(q.criteria)
            answer = {
                "type": "score",
                "score": 1.0,
                "legend": {str(i): d for i, d in enumerate(q.criteria)},
                "probabilities": {str(i): 1.0 / count for i in range(count)},
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


def test_decisions_route_end_to_end_with_stubbed_engine(client):
    resp = client.post(
        "/v1/decisions",
        json={
            "model": "granite-switch",
            "input": "a ticket",
            "questions": [
                {
                    "type": "choice",
                    "name": "queue",
                    "instructions": "Which team?",
                    "choices": [
                        {"value": "payments", "description": "Payouts"},
                        {"value": "account", "description": "Logins"},
                    ],
                },
                {"type": "predicate", "name": "escalate", "instructions": "Urgent?"},
                {
                    "type": "score",
                    "name": "sev",
                    "instructions": "How bad?",
                    "levels": [{"label": "low"}, {"label": "high"}],
                },
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "granite-switch"
    assert [a["name"] for a in body["answers"]] == ["queue", "escalate", "sev"]
    assert body["answers"][0]["choice"] == "payments"
    assert body["answers"][1] == {
        "type": "predicate",
        "name": "escalate",
        "probability": 0.9,
    }
    assert body["answers"][2]["probabilities"][1]["label"] == "high"
    # one stubbed engine call per question: 10 in (4 from cache), 1 out, each
    assert body["usage"] == {
        "input_tokens": 30,
        "input_tokens_details": {"cached_tokens": 12, "cache_write_tokens": 0},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 33,
    }


def test_decisions_route_validation_error_uses_openai_shape(client):
    resp = client.post(
        "/v1/decisions",
        json={"model": "granite-switch", "input": "x", "questions": []},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert "1-200" in resp.json()["error"]["message"]


def test_decisions_route_image_input_rejected(client):
    resp = client.post(
        "/v1/decisions",
        json={
            "model": "granite-switch",
            "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "u"}]}],
            "questions": [{"type": "predicate", "instructions": "x?"}],
        },
    )
    assert resp.status_code == 400
    assert "image input is not supported" in resp.json()["error"]["message"]


def test_systemone_route_still_works_alongside(client):
    """The new route must not disturb the existing one: same stubbed engine, jev shapes."""
    resp = client.post(
        "/v1/systemone",
        json={
            "model": "granite-switch",
            "state": "a ticket",
            "questions": {"escalate": {"type": "noul", "instructions": "Urgent?"}},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["answers"]["escalate"] == {"type": "noul", "noul": 0.9}


# --------------------------------------------------------------------------------------------
# Error body shape (OpenAI's, not jev's, on this route)
# --------------------------------------------------------------------------------------------


def test_openai_error_shape():
    resp = sp._openai_error(__import__("http").HTTPStatus.BAD_REQUEST, "boom", param="questions")
    import json

    body = json.loads(resp.body)
    assert body == {
        "error": {
            "message": "boom",
            "type": "invalid_request_error",
            "param": "questions",
            "code": None,
        }
    }
    assert resp.status_code == 400


def test_engine_error_maps_to_openai_error_shape(client, monkeypatch):
    """An engine-side failure (vLLM ErrorResponse) must not surface as a bare 500."""

    async def failing(*args):
        raise sp.EngineError("This model's maximum context length is 8192 tokens.", 400)

    monkeypatch.setattr(sp, "_answer_question", failing)
    resp = client.post(
        "/v1/decisions", json={"model": "m", "input": "x", "questions": [{"type": "predicate", "instructions": "q?"}]}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert "maximum context length" in resp.json()["error"]["message"]


def test_engine_error_on_systemone_route_uses_its_error_shape(client, monkeypatch):
    async def failing(*args):
        raise sp.EngineError("engine overloaded", 503)

    monkeypatch.setattr(sp, "_answer_question", failing)
    resp = client.post("/v1/systemone", json={"model": "m", "state": "x", "questions": {"q": {"type": "noul"}}})
    assert resp.status_code == 503
    assert resp.json() == {"error": {"message": "engine overloaded", "code": 503}}
