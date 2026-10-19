# granite-switch-systemone

A [jev system-one API](https://docs.typesafe.ai/api) (`POST /v1/systemone`) and
[OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions)
(`POST /v1/decisions`) bridge for
[granite-switch](https://github.com/generative-computing/granite-switch)'s vLLM backend,
mounted on the **same** server that already serves `/v1/chat/completions` — one engine, one
GPU KV cache, one container.

jev's system-one API takes an unstructured `state` plus typed `questions`
(`noul`/`choice`/`score`) and returns typed `answers` with calibrated probabilities. A
granite-switch decision adapter can already do exactly this through
`/v1/chat/completions` — a caller just has to hand-build the request themselves (the
adapter's trained invocation marker, a single-token label pool, a softmax over
`top_logprobs`). This package does that translation server-side, as a
[`vllm.endpoint_plugins`](https://github.com/vllm-project/vllm) entry point, so a
jev-shaped caller gets a jev-shaped answer without needing to know any of that. The same
bridge also speaks OpenAI's Decisions wire format on `/v1/decisions`, so callers written
against either industry shape — the official OpenAI SDKs included — work by changing
`base_url` only; see [examples/decisions.md](examples/decisions.md) for calls in both
formats.

## Why this is its own repo, not part of granite-switch or mellea

- **granite-switch** should not carry adapter-specific inference logic — every other
  granite adapter function's prompt-building and output-reading lives in
  [**mellea**](https://github.com/generative-computing/mellea)'s own inference path
  (`mellea/backends/adapters/`, `mellea/formatters/granite/intrinsics/`), not in
  granite-switch itself, and `granite-switch` and `mellea` must not depend on each other
  in a way that creates a cycle.
- **mellea** does not yet have an equivalent primitive for a single-token
  label-logit-softmax decision (every one of its existing `IOContract` shapes parses
  structured JSON text from the model, not a softmax over per-label logprobs) — see
  `src/granite_switch_systemone/renderer.py`'s docstring for exactly what's missing and
  where it belongs once it exists.

So this package depends on `granite-switch` (never the reverse), carries today's one
placeholder implementation of the decision-adapter rendering/readout logic clearly marked
as such, and is built so that implementation can be swapped out for a real call into
mellea's own machinery later with no code change to the generic bridge itself (see
"Architecture" below).

## Architecture

- **`plugin.py`** — the generic request/response bridge, carrying both wire formats
  (jev's `/v1/systemone` and OpenAI's `/v1/decisions`) over one translation to the same
  internals: body validation matching jev's own `HTTPValidationError` schema (and
  OpenAI's error shape on the Decisions route), context-length guarding, the
  >128-option `choice` chunking jev's ceiling needs that vLLM's own
  `logprob_token_ids` cap doesn't support in one call, jev's own `confidence` formulas,
  and the actual engine call (`openai_serving_chat.create_chat_completion`, the exact
  method `/v1/chat/completions`'s own route uses — no new engine access path).
  **Has no knowledge of any specific adapter's trained prompt convention.**
- **`renderer.py`** — the one adapter-specific piece: turns `(state, instructions,
  option descriptions)` into chat messages and single-token label ids, for the specific
  two-user-turn, aLoRA-marker-activated convention a granite-switch decision adapter is
  trained with. Resolved dynamically by `plugin.py` via `SYSTEMONE_RENDERER` (a
  `"module:function"` dotted path, defaulting to this module) — `plugin.py` never
  imports it directly, so swapping in a different renderer (a different adapter's
  convention, or eventually mellea's own equivalent primitive) needs no change to
  `plugin.py`.

## Why there's no system prompt in the rendered request

Dropping the system prompt lets a decision share vLLM's prefix cache with a
`/v1/chat/completions` request, but only when that chat's tokens match from the very
start. The cache reuses a block only when every token before it matches, so the chat
must open with no system turn and its first user turn must be literally
`State:\n<state>`. A chat that has its own system prompt, or that sends the state
without the `State:` prefix, diverges within the first few tokens and shares nothing.
Decisions about the same state always share the state turn with each other, which is
what makes multi-question requests and chunked large `choice`s cheap.

`tests/test_prefix_sharing.py` pins these cases against the real tokenizer and chat
template, and `scripts/e2e_check.py`'s `prefix_cache` check measures the hits on a live
server.

## Configuration

These configure the shared engine path — adapter activation, prompt rendering, plugin
loading — so they govern both `/v1/systemone` and `/v1/decisions`; there is nothing
to configure per route.

| Env var | Required? | What it does |
|---|---|---|
| `SYSTEMONE_ADAPTER_NAME` | Only if the checkpoint has more than one adapter | Which `adapter_name` activates the decision adapter. Auto-resolved if the checkpoint has exactly one adapter; cross-checked against the checkpoint's real `GraniteSwitchConfig.adapter_names` either way, raising at server startup (not silently) on a mismatch. |
| `SYSTEMONE_INVOCATION_MARKER` | **Always**, for the default renderer | The literal text that activates the decision aLoRA in its invocation turn (e.g. `"### DECISION"`). No safe default: granite-switch's chat template silently falls back to a *different* activation placement when the marker text isn't found, rather than raising — see `renderer.py`'s docstring. |
| `SYSTEMONE_RENDERER` | No (defaults to this package's own renderer) | A `"module:function"` dotted path to a different rendering implementation, for a differently-trained adapter or a future mellea-backed one. |
| `VLLM_PLUGINS` | Yes (set in the Dockerfile) | Must include both `register_granite_switch` and `granite_switch_systemone` — see the Dockerfile's own comment for why. |

## Usage

```bash
docker build --platform linux/amd64 -t granite-switch-systemone:latest .

docker run -d --gpus all \
    -v /path/to/composed/checkpoint:/repository:ro \
    -p 8000:8000 \
    -e SYSTEMONE_INVOCATION_MARKER="### DECISION" \
    granite-switch-systemone:latest
```

```python
import json, urllib.request

body = {
    "model": "granite-switch",
    "state": "My payouts have failed three times. The bank says everything is fine.",
    "questions": {
        "queue": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {"payments": "Payout failures", "account": "Login issues", "other": "..."},
        },
        "escalate": {"type": "noul", "instructions": "Does this need urgent attention?"},
    },
}
req = urllib.request.Request(
    "http://localhost:8000/v1/systemone",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.loads(urllib.request.urlopen(req).read())
# resp["answers"]["queue"]    -> {"type": "choice", "choice": "payments", "probabilities": {...}, "confidence": 0.97}
# resp["answers"]["escalate"] -> {"type": "noul", "noul": 0.89}
```

## `/v1/decisions` (OpenAI's Decisions format)

The same bridge also serves OpenAI's Decisions wire format (`POST /v1/decisions`, public
beta) next to `/v1/systemone` — one engine, one adapter, both industry shapes, so a
caller written against OpenAI's Decisions API (official SDKs, pydantic-ai, ...) works by
changing `base_url` only. A request through either route renders a byte-identical prompt
and reads the same label logits: the `/v1/decisions` translation converges on exactly the
internal state + question objects the systemone route builds for the equivalent request,
which is what `tests/test_decisions.py`'s parity tests assert. Full example calls in both
formats (curl, the OpenAI Python SDK, and the equivalent systemone request) are in
[examples/decisions.md](examples/decisions.md).

Mapping: `input` (a string, or user messages with text parts) flattens to the shared
`state`; `predicate` maps onto the trained true/false readout (`probability` = P(true));
`choice` maps onto the same criteria rendering — values typed `string | boolean` are
coerced to their text form, and a request mixing boolean `true` with the string `"true"`
is rejected, because the trained convention's option keys are plain text and the model
cannot distinguish the two; `score` maps onto the same level readout (`score` = expected
level, per-level `probabilities` carrying the original labels). Answers come back in
question order, echoing each question's `name` or null.

```bash
curl http://localhost:8000/v1/decisions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "granite-switch",
      "input": "My payouts have failed three times. The bank says everything is fine.",
      "questions": [
        {"type": "choice", "name": "queue",
         "instructions": "Which team should handle this ticket?",
         "choices": [
           {"value": "payments", "description": "Payout failures"},
           {"value": "account", "description": "Login issues"},
           {"value": "other", "description": "Something else"}]},
        {"type": "predicate", "name": "escalate",
         "instructions": "Does this need urgent attention?"}
      ]
    }'
```

Response (abridged): `{"model": ..., "answers": [{"type": "choice", "name": "queue",
"choice": "payments", "probabilities": [...], "confidence": 0.97}, {"type": "predicate",
"name": "escalate", "probability": 0.89}], "usage": {...}}` — answers in question order.

Deliberate scope limits on this route:

- Image `input` parts are rejected with a 400 — text decision adapter, the same stance as
  jev's `box`.
- Per-question `refusal` answers are never produced: this adapter fail-closes with an
  error rather than refusing.
- `input` messages must have role `user`. System, developer and assistant messages and
  tool-call items are valid in OpenAI's spec but rejected with a 400: the trained layout
  is one `State:` turn and has no place to render them.
- Error bodies follow OpenAI's `{"error": {"message", "type", "param", "code"}}` shape (the
  SDKs parse it on this route), not jev's `HTTPValidationError`.
- `usage.input_tokens_details` carries vLLM's prefix-cache counts: `cached_tokens` read
  from the cache, `cache_write_tokens` written to it. vLLM reports them only with
  `--enable-prompt-tokens-details` (set in `serve.sh`); both are 0 without it.
  `prompt_cache_key`, `prompt_cache_options` and per-part `prompt_cache_breakpoint` are
  accepted and ignored, since vLLM reuses every full block of a matching prefix on its
  own.

`tests/test_openai_spec.py` validates requests and responses against OpenAI's published
OpenAPI spec (`tests/spec/openai_decisions.json`, extracted verbatim with its source
commit), with each limit above listed as a deliberate exception.
- On vLLM releases that ship a native `/v1/decisions` (main after v0.32), this plugin's
  route shadows it under vLLM's documented endpoint-plugin precedence — strictly better
  here, since the native route rejects GraniteSwitch architectures.

## Testing against a live server

`scripts/e2e_check.py` runs the full wire-format battery against any deployment of this
container — both routes' answer shapes, cross-route parity, input flattening, boolean
value round-tripping, error shapes, usage, that `/v1/chat/completions` still works
on the shared server, and prefix-cache hits between chat and decisions (read from
vLLM's `/metrics`; run it on a server with no other traffic). It uses only the Python standard library, so it runs wherever
Python does:

```bash
python3 scripts/e2e_check.py --base-url http://localhost:8000
```

Pass `--model` to skip model discovery, `--skip-chat` on servers without chat,
`--skip-prefix-cache` to skip the cache check, and
`--tolerance` to loosen the cross-route parity bound (default 1e-6; the two routes
render byte-identical prompts, but GPU batching can shift low-order bits between
separate requests). Exit code 0 means every check passed.

Deployment is deliberately out of scope: whatever brings the server up — a local
container, a cluster job — only needs to hand the script its base URL, so any
site-specific deployment automation can live outside this repo.

## Deliberate scope limits (not a proxy to typesafe.ai)

- `model` is required on the request (schema compliance) but never validated — the
  response always echoes the actually-served model name.
- Context limits (64k tokens/request, 32k/question by default, 262k under
  `TypeSafe-Danger-Zone-Long-Context: true`) are additionally capped by this server's
  own `--max-model-len`, since a running engine can't be resized per request.
- A `choice` with more than 128 options (vLLM's own `SamplingParams.logprob_token_ids`
  limit per request, below jev's published 255-option ceiling) is handled by chunking
  across more than one `create_chat_completion` call for the identical prompt, rather
  than rejected.
- `401`/`429`/`529` are not implemented — access control and rate limiting belong to
  whatever is hosting this container.
- jev's fourth question type (`"box"`, bounding-box-on-an-image) is not supported — no
  vision capability on a text decision adapter.

## Known upstream limitation

`generative-computing/granite-switch`'s composer discards each adapter's original
`adapter_config.json` (which, for an aLoRA adapter, carries `alora_invocation_tokens` —
PEFT's own source of truth for the invocation marker `SYSTEMONE_INVOCATION_MARKER` has
to be told explicitly here) once compose finishes. See
[generative-computing/granite-switch#TODO](https://github.com/generative-computing/granite-switch/issues)
for the follow-up filed to retain it.

## License

Apache-2.0.
