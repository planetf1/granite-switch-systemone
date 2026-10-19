"""Env-gated end-to-end parity: against a RUNNING server (any deployment of this plugin),
the same semantic questions through /v1/decisions and /v1/systemone must return the same
probabilities. Both routes share the engine path by construction (see
tests/test_decisions.py's parity tests); this checks the wiring end to end, including the
routes' registration on the real app.

  SYSTEMONE_BASE_URL=http://host:8000 python3 -m pytest tests/test_decisions_e2e.py -v
"""

import json
import os
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("SYSTEMONE_BASE_URL")

pytestmark = pytest.mark.skipif(not BASE, reason="SYSTEMONE_BASE_URL not set")

STATE = "My payouts have failed three times. The bank says everything is fine."


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise AssertionError(f"{path} returned {e.code}: {e.read().decode()}") from e


def test_decisions_matches_systemone():
    decisions = _post(
        "/v1/decisions",
        {
            "model": "granite-switch",
            "input": STATE,
            "questions": [
                {
                    "type": "choice",
                    "name": "queue",
                    "instructions": "Which team should handle this ticket?",
                    "choices": [
                        {"value": "payments", "description": "Payout failures"},
                        {"value": "account", "description": "Login issues"},
                        {"value": "other", "description": "Something else"},
                    ],
                },
                {
                    "type": "predicate",
                    "name": "escalate",
                    "instructions": "Does this need urgent attention?",
                },
                {
                    "type": "score",
                    "name": "sev",
                    "instructions": "How urgent is this?",
                    "levels": [
                        {"label": "can wait", "description": "No rush"},
                        {"label": "today", "description": "Same day"},
                    ],
                },
            ],
        },
    )
    systemone = _post(
        "/v1/systemone",
        {
            "model": "granite-switch",
            "state": STATE,
            "questions": {
                "queue": {
                    "type": "choice",
                    "instructions": "Which team should handle this ticket?",
                    "criteria": {
                        "payments": "Payout failures",
                        "account": "Login issues",
                        "other": "Something else",
                    },
                },
                "escalate": {
                    "type": "noul",
                    "instructions": "Does this need urgent attention?",
                },
                "sev": {
                    "type": "score",
                    "instructions": "How urgent is this?",
                    "criteria": ["No rush", "Same day"],
                },
            },
        },
    )

    by_name = {a["name"]: a for a in decisions["answers"]}
    assert [a["name"] for a in decisions["answers"]] == ["queue", "escalate", "sev"]

    # choice: same pick, same per-option probabilities
    d_choice = by_name["queue"]
    s_choice = systemone["answers"]["queue"]
    assert d_choice["type"] == "choice"
    assert d_choice["choice"] == s_choice["choice"]
    d_probs = {p["value"]: p["probability"] for p in d_choice["probabilities"]}
    for key, prob in s_choice["probabilities"].items():
        assert d_probs[key] == pytest.approx(prob, abs=1e-9), key

    # predicate: probability == noul (P(true))
    assert by_name["escalate"]["probability"] == pytest.approx(systemone["answers"]["escalate"]["noul"], abs=1e-9)

    # score: same expected level and per-level probabilities
    d_score = by_name["sev"]
    s_score = systemone["answers"]["sev"]
    assert d_score["score"] == pytest.approx(s_score["score"], abs=1e-9)
    d_probs = {p["value"]: p["probability"] for p in d_score["probabilities"]}
    for key, prob in s_score["probabilities"].items():
        assert d_probs[int(key)] == pytest.approx(prob, abs=1e-9), key


def test_decisions_input_messages_flatten_to_same_state():
    """A message-list input must answer identically to the plain-string form."""
    string_form = _post(
        "/v1/decisions",
        {
            "model": "granite-switch",
            "input": STATE,
            "questions": [{"type": "predicate", "name": "q", "instructions": "Is this about payouts?"}],
        },
    )
    message_form = _post(
        "/v1/decisions",
        {
            "model": "granite-switch",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": STATE}],
                }
            ],
            "questions": [{"type": "predicate", "name": "q", "instructions": "Is this about payouts?"}],
        },
    )
    assert string_form["answers"][0]["probability"] == pytest.approx(
        message_form["answers"][0]["probability"], abs=1e-9
    )
