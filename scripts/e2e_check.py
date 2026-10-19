#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end check for a live granite-switch-systemone server.

Runs the full wire-format battery against any deployment of this plugin --
local, container, or remote -- using only the Python standard library, so it
runs anywhere Python 3 runs. Deployment is out of scope by design: whatever
brings the server up (a local container, a cluster job, anything else) only
needs to hand this script the server's base URL.

  python3 scripts/e2e_check.py --base-url http://localhost:8000

Exit code 0 means every check passed. Checks:

  health              GET /health answers
  models              GET /v1/models lists the served model
  systemone_shapes    one noul/choice/score question each, jev answer shapes
  decisions_shapes    one predicate/choice/score question each, OpenAI shapes
  parity              the same questions through both routes agree: same
                      choice pick, same per-option/level probabilities, same
                      predicate probability, same score
  input_flattening    a message-list input answers identically to the string
  boolean_values      boolean choice values round-trip as booleans
  usage               usage is present and non-zero
  errors              400s with OpenAI's error shape (empty questions, image
                      input, 11 score levels) and jev's 422 shape on the
                      systemone route
  chat                /v1/chat/completions still works on the shared server
                      (skip with --skip-chat)
  prefix_cache        vLLM's prefix-cache hit counter (GET /metrics) for a
                      decision after a chat or a decision about the same
                      state: hits when the chat's first turn is the
                      `State:` turn and for a second decision, none when the
                      chat sends the raw state (skip with --skip-prefix-cache;
                      also skipped with --skip-chat). The counters are
                      server-wide, so run it on a server with no other
                      traffic.

Parity tolerance defaults to 1e-6: the two routes render byte-identical
prompts, but GPU batching can shift low-order bits between separate requests,
so exact equality is not required.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

STATE = "My payouts have failed three times. The bank says everything is fine."


class Check:
    def __init__(self, name: str) -> None:
        self.name = name
        self.failures: list[str] = []
        self.notes: list[str] = []

    def expect(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)

    def expect_equal(self, actual: object, expected: object, label: str) -> None:
        self.expect(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")

    def expect_close(self, actual: float, expected: float, label: str, tol: float) -> None:
        self.expect(abs(actual - expected) <= tol, f"{label}: expected ~{expected!r}, got {actual!r}")


class Client:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        return self.request("POST", path, body)

    def get(self, path: str) -> tuple[int, dict]:
        return self.request("GET", path)

    def get_text(self, path: str) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=self.timeout) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()


DECISIONS_BODY = {
    "model": "PLACEHOLDER",
    "input": STATE,
    "questions": [
        {
            "type": "choice",
            "name": "queue",
            "instructions": "Which team should handle this ticket?",
            "choices": [
                {"value": "payments", "description": "Payout failures and refunds"},
                {"value": "account", "description": "Login and account access"},
                {"value": "other", "description": "Something else"},
            ],
        },
        {"type": "predicate", "name": "escalate", "instructions": "Does this need urgent attention?"},
        {
            "type": "score",
            "name": "severity",
            "instructions": "How urgent is this?",
            "levels": [
                {"label": "can wait", "description": "No rush"},
                {"label": "this week", "description": "Within the week"},
                {"label": "today", "description": "Same day"},
            ],
        },
    ],
}

SYSTEMONE_BODY = {
    "model": "PLACEHOLDER",
    "state": STATE,
    "questions": {
        "queue": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "payments": "Payout failures and refunds",
                "account": "Login and account access",
                "other": "Something else",
            },
        },
        "escalate": {"type": "noul", "instructions": "Does this need urgent attention?"},
        "severity": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["No rush", "Within the week", "Same day"],
        },
    },
}


def check_health(client: Client, model: str) -> Check:
    check = Check("health")
    status, _ = client.get("/health")
    check.expect_equal(status, 200, "GET /health status")
    return check


def check_models(client: Client, model: str) -> Check:
    check = Check("models")
    status, body = client.get("/v1/models")
    check.expect_equal(status, 200, "GET /v1/models status")
    ids = [m.get("id") for m in body.get("data", [])]
    check.expect(model in ids, f"served model {model!r} not in /v1/models: {ids}")
    return check


def check_systemone_shapes(client: Client, model: str) -> Check:
    check = Check("systemone_shapes")
    status, body = client.post("/v1/systemone", {**SYSTEMONE_BODY, "model": model})
    check.expect_equal(status, 200, "status")
    answers = body.get("answers", {})
    check.expect(set(answers) == {"queue", "escalate", "severity"}, "answer keys")
    queue = answers.get("queue", {})
    check.expect_equal(queue.get("type"), "choice", "queue type")
    check.expect(queue.get("choice") in {"payments", "account", "other"}, "queue pick")
    check.expect(
        isinstance(queue.get("probabilities"), dict) and len(queue["probabilities"]) == 3,
        "queue probabilities shape",
    )
    check.expect(isinstance(queue.get("confidence"), (int, float)), "queue confidence")
    escalate = answers.get("escalate", {})
    check.expect_equal(escalate.get("type"), "noul", "escalate type")
    check.expect(isinstance(escalate.get("noul"), (int, float)), "escalate probability")
    severity = answers.get("severity", {})
    check.expect_equal(severity.get("type"), "score", "severity type")
    check.expect(isinstance(severity.get("score"), (int, float)), "severity score")
    check.expect_equal(len(severity.get("probabilities", {})), 3, "severity probabilities")
    return check


def check_decisions_shapes(client: Client, model: str) -> Check:
    check = Check("decisions_shapes")
    status, body = client.post("/v1/decisions", {**DECISIONS_BODY, "model": model})
    check.expect_equal(status, 200, "status")
    answers = body.get("answers", [])
    check.expect_equal(len(answers), 3, "answer count")
    check.expect_equal([a.get("name") for a in answers], ["queue", "escalate", "severity"], "names")
    queue, escalate, severity = answers
    check.expect_equal(queue.get("type"), "choice", "queue type")
    check.expect(queue.get("choice") in {"payments", "account", "other"}, "queue pick")
    check.expect(
        isinstance(queue.get("probabilities"), list) and len(queue["probabilities"]) == 3,
        "queue probabilities shape",
    )
    check.expect(isinstance(queue.get("confidence"), (int, float)), "queue confidence")
    check.expect_equal(escalate.get("type"), "predicate", "escalate type")
    check.expect(isinstance(escalate.get("probability"), (int, float)), "escalate probability")
    check.expect_equal(severity.get("type"), "score", "severity type")
    check.expect(isinstance(severity.get("score"), (int, float)), "severity score")
    check.expect_equal(len(severity.get("probabilities", [])), 3, "severity probabilities")
    check.expect_equal([p.get("value") for p in severity.get("probabilities", [])], [0, 1, 2], "level indices")
    return check


def check_parity(client: Client, model: str, tol: float) -> Check:
    check = Check("parity")
    _, decisions = client.post("/v1/decisions", {**DECISIONS_BODY, "model": model})
    _, systemone = client.post("/v1/systemone", {**SYSTEMONE_BODY, "model": model})
    d_queue, d_escalate, d_severity = decisions["answers"]
    s_queue = systemone["answers"]["queue"]
    s_escalate = systemone["answers"]["escalate"]
    s_severity = systemone["answers"]["severity"]

    check.expect_equal(d_queue["choice"], s_queue["choice"], "choice pick")
    d_probs = {p["value"]: p["probability"] for p in d_queue["probabilities"]}
    for key, prob in s_queue["probabilities"].items():
        check.expect_close(d_probs[key], prob, f"choice probability[{key}]", tol)

    check.expect_close(d_escalate["probability"], s_escalate["noul"], "predicate probability", tol)
    check.expect_close(d_severity["score"], s_severity["score"], "score", tol)
    d_probs = {p["value"]: p["probability"] for p in d_severity["probabilities"]}
    for key, prob in s_severity["probabilities"].items():
        check.expect_close(d_probs[int(key)], prob, f"level probability[{key}]", tol)
    return check


def check_input_flattening(client: Client, model: str, tol: float) -> Check:
    check = Check("input_flattening")
    question = {"type": "predicate", "name": "q", "instructions": "Is this about payouts?"}
    _, string_form = client.post("/v1/decisions", {"model": model, "input": STATE, "questions": [question]})
    _, message_form = client.post(
        "/v1/decisions",
        {
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": STATE}]}],
            "questions": [question],
        },
    )
    check.expect_close(
        string_form["answers"][0]["probability"],
        message_form["answers"][0]["probability"],
        "string vs message-list probability",
        tol,
    )
    return check


def check_boolean_values(client: Client, model: str) -> Check:
    check = Check("boolean_values")
    status, body = client.post(
        "/v1/decisions",
        {
            "model": model,
            "input": "The export finished successfully.",
            "questions": [
                {
                    "type": "choice",
                    "name": "ok",
                    "instructions": "Did the export succeed?",
                    "choices": [{"value": True}, {"value": False}],
                }
            ],
        },
    )
    check.expect_equal(status, 200, "status")
    answer = body["answers"][0]
    check.expect(answer["choice"] in (True, False), "typed boolean pick")
    values = [p["value"] for p in answer["probabilities"]]
    check.expect_equal(sorted(values, key=str), [False, True], "typed boolean probabilities")
    return check


def check_usage(client: Client, model: str) -> Check:
    check = Check("usage")
    _, body = client.post("/v1/decisions", {**DECISIONS_BODY, "model": model})
    usage = body.get("usage", {})
    check.expect(usage.get("input_tokens", 0) > 0, "input_tokens > 0")
    check.expect_equal(
        usage.get("total_tokens"),
        usage.get("input_tokens", -1) + usage.get("output_tokens", 0),
        "total = input + output",
    )
    return check


def check_errors(client: Client, model: str) -> Check:
    check = Check("errors")
    status, body = client.post("/v1/decisions", {"model": model, "input": "x", "questions": []})
    check.expect_equal(status, 400, "empty questions status")
    check.expect_equal(body.get("error", {}).get("type"), "invalid_request_error", "error shape")

    status, body = client.post(
        "/v1/decisions",
        {
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "u"}]}],
            "questions": [{"type": "predicate", "instructions": "x?"}],
        },
    )
    check.expect_equal(status, 400, "image input status")

    status, body = client.post(
        "/v1/decisions",
        {
            "model": model,
            "input": "x",
            "questions": [
                {
                    "type": "score",
                    "instructions": "x?",
                    "levels": [{"label": str(i)} for i in range(11)],
                }
            ],
        },
    )
    check.expect_equal(status, 400, "11 score levels status")

    status, body = client.post(
        "/v1/systemone",
        {"model": model, "state": "x", "questions": {"q": {"type": "choice", "criteria": {}}}},
    )
    check.expect_equal(status, 422, "systemone bad criteria status")
    check.expect("detail" in body, "jev HTTPValidationError shape")
    return check


def check_chat(client: Client, model: str) -> Check:
    check = Check("chat")
    status, body = client.post(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say ok."}],
            "max_tokens": 5,
        },
    )
    check.expect_equal(status, 200, "status")
    check.expect(bool(body.get("choices")), "choices present")
    return check


# vLLM's prefix-cache counters, in tokens; prometheus_client appends `_total` to counters.
PREFIX_CACHE_COUNTERS = {"vllm:prefix_cache_queries_total": "queries", "vllm:prefix_cache_hits_total": "hits"}
METRIC_LINE = re.compile(r"^(?P<name>[^{\s]+)(?:\{.*\})?\s+(?P<value>\S+)")
TICKET = (
    "My payouts to the business account have failed three times this month. The bank says nothing "
    "is wrong on their side and the account details are correct. Each failure arrived as an email "
    "with no reason code, and support has not replied. I have payroll due on Friday. "
)


def prefix_cache_counters(client: Client) -> dict[str, float] | None:
    """Prefix-cache queries and hits summed over engines, or None when /metrics does not
    expose them (stats disabled with --disable-log-stats, or not a vLLM server)."""
    status, text = client.get_text("/metrics")
    if status != 200:
        return None
    totals = {"queries": 0.0, "hits": 0.0}
    found = False
    for line in text.splitlines():
        match = METRIC_LINE.match(line)
        if match and match["name"] in PREFIX_CACHE_COUNTERS:
            totals[PREFIX_CACHE_COUNTERS[match["name"]]] += float(match["value"])
            found = True
    return totals if found else None


def cache_hits(client: Client, path: str, body: dict) -> tuple[float | None, dict]:
    """POST one request and return the prefix-cache hit tokens it added, with its body.
    The engine reports stats on its own loop, which can trail the HTTP response, so poll
    until the query counter moves. None means it never moved: prefix caching is off."""
    before = prefix_cache_counters(client)
    _, response = client.post(path, body)
    for _ in range(50):
        after = prefix_cache_counters(client)
        if before is not None and after is not None and after["queries"] > before["queries"]:
            return after["hits"] - before["hits"], response
        time.sleep(0.1)
    return None, response


def check_prefix_cache(client: Client, model: str) -> Check:
    check = Check("prefix_cache")
    if prefix_cache_counters(client) is None:
        check.expect(False, "GET /metrics has no vllm:prefix_cache_* counters (log stats disabled?)")
        return check

    def state() -> str:
        # A fresh state per probe, unique from its first block, so no earlier request
        # (this run's or a previous one's) can supply a hit.
        return f"Ticket {uuid.uuid4().hex}. " + TICKET * 6

    def decision(text: str, question: str) -> dict:
        return {
            "model": model,
            "input": text,
            "questions": [{"type": "predicate", "name": "q", "instructions": question}],
        }

    def chat(messages: list[dict]) -> dict:
        return {"model": model, "messages": messages, "max_tokens": 1}

    def probe(label: str, setup: tuple[str, dict], text: str, expect_hits: bool) -> None:
        cache_hits(client, *setup)
        hits, response = cache_hits(client, "/v1/decisions", decision(text, "Does this need urgent attention?"))
        if hits is None:
            check.expect(False, f"{label}: prefix-cache query counter never moved (prefix caching disabled?)")
            return
        prompt = response.get("usage", {}).get("input_tokens", 0)
        check.notes.append(f"{label}: {hits:.0f} of {prompt} decision prompt tokens from cache")
        if expect_hits:
            # The state turn is most of the decision prompt; the decision turn is the rest.
            check.expect(hits >= prompt / 2, f"{label}: expected at least half the prompt from cache")
        else:
            check.expect(hits == 0, f"{label}: expected no full block in common")

    text = state()
    probe(
        "after a decision about the same state",
        ("/v1/decisions", decision(text, "Is the customer angry?")),
        text,
        True,
    )
    text = state()
    shared_layout = [
        {"role": "user", "content": "State:\n" + text},
        {"role": "user", "content": "Summarise this ticket in one sentence."},
    ]
    probe(
        "after a chat whose first turn is `State:` + state", ("/v1/chat/completions", chat(shared_layout)), text, True
    )
    text = state()
    raw = [{"role": "user", "content": text}]
    probe("after a chat sending the raw state", ("/v1/chat/completions", chat(raw)), text, False)
    return check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--base-url", required=True, help="server base URL, e.g. http://host:8000")
    parser.add_argument("--model", default=None, help="model name (default: first /v1/models entry)")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="parity tolerance")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout seconds")
    parser.add_argument("--skip-chat", action="store_true", help="skip the chat-completions check")
    parser.add_argument("--skip-prefix-cache", action="store_true", help="skip the prefix-cache check")
    args = parser.parse_args()

    client = Client(args.base_url, args.timeout)
    model = args.model
    if model is None:
        status, body = client.get("/v1/models")
        if status != 200 or not body.get("data"):
            print("FAIL models: cannot discover the served model; pass --model")
            return 1
        model = body["data"][0]["id"]
    print(f"server {args.base_url} model {model}")

    checks = [
        check_health(client, model),
        check_models(client, model),
        check_systemone_shapes(client, model),
        check_decisions_shapes(client, model),
        check_parity(client, model, args.tolerance),
        check_input_flattening(client, model, args.tolerance),
        check_boolean_values(client, model),
        check_usage(client, model),
        check_errors(client, model),
    ]
    if not args.skip_chat:
        checks.append(check_chat(client, model))
        if not args.skip_prefix_cache:
            checks.append(check_prefix_cache(client, model))

    failed = 0
    for check in checks:
        if check.failures:
            failed += 1
            print(f"FAIL {check.name}")
            for failure in check.failures:
                print(f"     {failure}")
        else:
            print(f"PASS {check.name}")
        for note in check.notes:
            print(f"     {note}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
