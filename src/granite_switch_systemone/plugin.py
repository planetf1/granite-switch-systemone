# SPDX-License-Identifier: Apache-2.0
"""jev system-one API (`POST /v1/systemone`) and OpenAI Decisions API
(`POST /v1/decisions`) as `vllm.endpoint_plugins` entry points, mounted on the SAME vLLM
OpenAI server that serves `/v1/chat/completions` -- one engine, one KV cache, one
container. Both routes are wire-format facades over one engine path: a request made
through either renders a byte-identical prompt and reads the same label logits (the
parity tests in `tests/test_decisions.py` assert the translation converges on exactly
what the systemone route builds for the equivalent request).

What this is: typesafe.ai's system-one API takes an unstructured `state` and typed
`questions` (`noul`/`choice`/`score`) and returns typed `answers` with calibrated
probabilities. A granite-switch decision adapter can already do exactly this through
`/v1/chat/completions` -- callers just have to hand-build the request themselves (an
adapter-specific invocation marker, a label pool, softmax over `top_logprobs`). This
plugin does that translation server-side so a jev-shaped caller gets a jev-shaped
answer.

ADAPTER-AGNOSTIC BY DESIGN: this module has no knowledge of any specific decision
adapter's trained prompt convention. The one piece that does -- turning
`(state, instructions, option descriptions)` into chat messages and label token ids
-- is resolved at import time via `SYSTEMONE_RENDERER`, a required
`"module:function"` dotted path (default: `granite_switch_systemone.renderer:render`,
this package's own placeholder implementation for the one adapter convention this
repo was built against -- see that module's docstring for why it is a placeholder,
not a permanent design). This package never imports a renderer module directly; it
only `importlib`s whatever path is configured, so it carries no dependency (and no
risk of a dependency cycle) on wherever a renderer's real implementation ends up
living.

Why no system prompt (the renderer's job, not this module's, but relevant to this
module's own prefix-cache design): without it, a decision's tokens can match a
`/v1/chat/completions` request from the start, and vLLM's prefix cache reuses blocks
only for a match from the start -- the two "endpoints" are two pieces of Python issuing
requests to the same AsyncLLM. That holds only for a chat with no system turn whose
first user turn is literally `State:\\n<state>`; any other chat diverges within a few
tokens (`tests/test_prefix_sharing.py`).

How a request reaches the engine: `attach_router`/`init_state` are vLLM 0.26+'s own
`vllm.endpoint_plugins` contract (`vllm/plugins/endpoint_plugins/interface.py`).
`init_state` runs after vLLM's own `init_generate_state` has set
`app.state.openai_serving_chat` (`vllm/entrypoints/openai/api_server.py::init_app_state`),
so we reach the engine the sanctioned way:
`openai_serving_chat.create_chat_completion(ChatCompletionRequest(...))`, the exact
method `/v1/chat/completions`'s own route calls
(`vllm/entrypoints/openai/chat_completion/api_router.py`). No new engine access path
is opened.

Label logprobs: `ChatCompletionRequest.logprob_token_ids` asks for exactly our label
token ids' logprobs (no `top_logprobs`-guessing/missing-label fallback needed), and
`return_tokens_as_token_ids=True` makes each returned token the exact string
`"token_id:{id}"`, so matching a label back to its logprob is exact integer
comparison, not decoded-string comparison. One sub-ULP constraint this inherits from
vLLM itself (not a choice we made): `SamplingParams` caps `logprob_token_ids` at
`MAX_LOGPROB_TOKEN_IDS` (128, `vllm/sampling_params.py`) -- below jev's own 255-option
ceiling for `choice`. A `choice` with more than 128 options is handled by splitting
its label ids into `MAX_LABEL_OPTIONS`-sized chunks and issuing one
`create_chat_completion` call per chunk, concurrently, all for the IDENTICAL prompt
(only `logprob_token_ids` differs per chunk) -- vLLM's own prefix cache is what makes
the second-and-later chunks cheap. This is not opening a new engine access path
either: every chunk still goes through the one sanctioned `create_chat_completion`
call, just more than once per question. jev's own 255-option ceiling bounds this to
at most `ceil(255/128) = 2` calls, never unbounded.

Deliberate scope limits for this self-hosted mirror (not a proxy to typesafe.ai):
`model` is required on the request (schema compliance) but never validated against
anything -- the response always echoes the actually-served model name. `401`/`429`/
`529` are not implemented (access control and rate limiting belong to whatever is
hosting this container, not duplicated here). jev's fourth question type (`"box"`,
bounding-box-on-an-image, gated by `TypeSafe-Danger-Zone-Images: true`) is not
supported -- no vision capability on a decision adapter.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import ValidationError as PydanticValidationError

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State
    from vllm.engine.protocol import EngineClient

_RENDERER_ENV = "SYSTEMONE_RENDERER"
_DEFAULT_RENDERER = "granite_switch_systemone.renderer:render"
_ADAPTER_NAME_ENV = "SYSTEMONE_ADAPTER_NAME"

# jev: request body <= 32 MiB.
MAX_BODY_BYTES = 32 * 1024 * 1024
# jev: nesting <= 64 levels anywhere in the request body.
MAX_NESTING_DEPTH = 64
# jev's documented context limits (tokens); the `TypeSafe-Danger-Zone-Long-Context: true`
# header raises both to DANGER_ZONE_TOKEN_LIMIT. We additionally cap by the engine's own
# --max-model-len, since we can't resize a running engine per request.
DEFAULT_PER_QUESTION_TOKEN_LIMIT = 32_000
DEFAULT_PER_REQUEST_TOKEN_LIMIT = 64_000
DANGER_ZONE_TOKEN_LIMIT = 262_144
DANGER_ZONE_HEADER = "typesafe-danger-zone-long-context"
# vLLM's own cap on `SamplingParams.logprob_token_ids` per request
# (vllm/sampling_params.py, MAX_LOGPROB_TOKEN_IDS) -- below jev's 255-option ceiling for
# `choice`. Also used as our chunk size: a question needing more single-token labels than
# this issues ceil(n / MAX_LABEL_OPTIONS) chunked requests instead of one (see
# _fetch_label_logprobs), never silently truncated or rejected.
MAX_LABEL_OPTIONS = 128


def _load_renderer():
    """Resolve `SYSTEMONE_RENDERER` (default: this package's own placeholder) to a
    callable at plugin-init time -- never imported directly by this module, so this
    package has zero import-time dependency on whatever module the configured path
    names (see module docstring)."""
    path = os.environ.get(_RENDERER_ENV, _DEFAULT_RENDERER)
    if ":" not in path:
        raise RuntimeError(f"{_RENDERER_ENV}={path!r} is not a valid 'module:function' path")
    module_name, func_name = path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, func_name)
    except AttributeError as e:
        raise RuntimeError(f"{_RENDERER_ENV}={path!r}: module {module_name!r} has no attribute {func_name!r}") from e


def _resolve_adapter_name(chat) -> str:
    """Which `adapter_name` activates the decision adapter on THIS loaded checkpoint.
    Cross-checked against the real `GraniteSwitchConfig.adapter_names` vLLM already
    has in memory for this request (`chat.model_config.hf_config.adapter_names`) --
    never assumed, never guessed from a hardcoded name list. `SYSTEMONE_ADAPTER_NAME`
    always wins if set, still validated against the same list, so a typo'd override
    fails loudly at startup. If unset and the checkpoint has EXACTLY one adapter, that
    one is used automatically (the common single-decision-adapter deployment); with
    zero or more than one adapter, the env var is required -- there is no safe guess.

    Getting this wrong is NOT a cosmetic bug: when `chat_template_kwargs.adapter_name`
    names an adapter absent from the checkpoint's chat-template `adapter_map`,
    granite-switch's Jinja template silently activates NO adapter at all -- no error,
    no control token, just a request that runs on plain base Granite with zero
    decision-adapter calibration."""
    adapter_names = getattr(chat.model_config.hf_config, "adapter_names", None) or []
    override = os.environ.get(_ADAPTER_NAME_ENV)
    if override:
        if override in adapter_names:
            return override
        raise RuntimeError(
            f"{_ADAPTER_NAME_ENV}={override!r} is not among this checkpoint's adapters {adapter_names!r}"
        )
    if len(adapter_names) == 1:
        return adapter_names[0]
    raise RuntimeError(
        f"{_ADAPTER_NAME_ENV} is not set, and this checkpoint has "
        f"{len(adapter_names)} adapters {adapter_names!r} (auto-resolution only "
        f"applies with exactly one) -- set {_ADAPTER_NAME_ENV} to the decision "
        "adapter's name."
    )


# --------------------------------------------------------------------------------------------
# Request/response shapes (jev's system-one API, as published at docs.typesafe.ai /
# https://api.typesafe.ai/openapi.json)
# --------------------------------------------------------------------------------------------


class Question(BaseModel):
    """Mirrors jev's real OpenAPI spec (NoulQuestion/ChoiceQuestion/ScoreQuestion),
    not just its prose docs -- the two disagree in places (e.g. the prose never
    mentions that `instructions` is optional for every type, or that choice/score
    each only require >=1 entry, not >=2; the real schema is authoritative). `box`
    (bounding-box-on-an-image questions, gated by `TypeSafe-Danger-Zone-Images: true`
    + an `assets` field) is a fourth jev question type this mirror does not support
    -- no vision capability on a decision adapter -- so it is deliberately absent
    from `type` and `assets` is deliberately absent from `SystemOneRequest` below; a
    client that sends either gets a clear 422, not a silent partial answer."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul", "choice", "score"]
    # jev: string | object | array | null, and NOT required, for every question
    # type's instructions.
    instructions: Any = None
    criteria: Any = None

    @model_validator(mode="after")
    def _check_criteria(self) -> Question:
        if self.type == "choice":
            # jev: ChoiceQuestion.criteria minProperties=1, maxProperties=255.
            if not isinstance(self.criteria, dict) or not (1 <= len(self.criteria) <= 255):
                raise ValueError("a `choice` question needs `criteria` as an object with 1..255 options")
            # jev: a choice criteria value is string | object | array | null.
            if not all(v is None or isinstance(v, (str, dict, list)) for v in self.criteria.values()):
                raise ValueError(
                    "a `choice` question's `criteria` values must each be a string, object, array, or null"
                )
        elif self.type == "score":
            # jev: ScoreQuestion.criteria minItems=1, maxItems=10.
            if not isinstance(self.criteria, list) or not (1 <= len(self.criteria) <= 10):
                raise ValueError("a `score` question needs `criteria` as an array of 1..10 level descriptions")
            # jev: a score level description is string | object | array (unlike choice,
            # not documented as nullable -- every level needs an actual description).
            if not all(isinstance(v, (str, dict, list)) for v in self.criteria):
                raise ValueError("a `score` question's `criteria` entries must each be a string, object, or array")
        elif self.criteria is not None and not isinstance(self.criteria, dict):
            raise ValueError("a `noul` question's `criteria`, if given, must be an object")
        return self


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Any
    model: str
    questions: dict[str, Question]

    @model_validator(mode="after")
    def _nonempty(self) -> SystemOneRequest:
        # jev: SystemOneRequest.questions has minProperties=1 (confirmed against the
        # real OpenAPI spec; the prose docs never state this, but the schema does).
        if not self.questions:
            raise ValueError("`questions` must have at least one entry")
        return self


# --------------------------------------------------------------------------------------------
# Request/response shapes (OpenAI's Decisions API, `POST /v1/decisions`, public beta:
# developers.openai.com/api/docs/guides/decisions and the openai-openai spec). A
# wire-format facade over the same internals as /v1/systemone, so callers written
# against OpenAI's Decisions API (official SDKs, pydantic-ai, ...) work by changing
# base_url only. Deliberate scope limits, mirroring the systemone route's:
# image `input` parts are rejected with a 400 (text decision adapter, the same stance
# as jev's `box`); per-question `refusal` answers are never produced (this adapter
# fail-closes with an error rather than refusing); error bodies follow OpenAI's
# `{"error": {...}}` shape (the SDKs parse it on this route) instead of jev's
# HTTPValidationError; `usage.input_tokens_details` carries vLLM's prefix-cache read
# and write counts (0 unless the server runs with `--enable-prompt-tokens-details`);
# only `user` messages are accepted in `input` (no system/developer/assistant turns, no
# tool-call items), since the trained layout has no place to render them.
# --------------------------------------------------------------------------------------------


def _choice_key(value: str | bool) -> str:
    """The option key a choice value renders as: strings pass through, booleans coerce
    to "true"/"false". OpenAI's spec types choice values as `string | boolean` and says
    "a string and a boolean with the same text are distinct", but the trained
    convention's option keys are plain label text, where `true` and "true" are
    indistinguishable to the model -- so a request mixing the two is rejected (see
    DecisionQuestion._check_type_fields) rather than answered with probabilities the
    model cannot actually separate."""
    return value if isinstance(value, str) else ("true" if value else "false")


class DecisionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | bool
    description: str | None = None


class DecisionLevel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    description: str | None = None


class DecisionQuestion(BaseModel):
    """One OpenAI Decisions question (`predicate`/`choice`/`score`), with the field
    constraints of OpenAI's published spec: `instructions` required, 2-255 unique
    choices, at most 10 score levels. `name` is optional -- answers are positional, so
    duplicate or missing names cannot collide (the response echoes the name or null)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["predicate", "choice", "score"]
    instructions: str
    name: str | None = None
    choices: list[DecisionChoice] | None = None
    levels: list[DecisionLevel] | None = None

    @model_validator(mode="after")
    def _check_type_fields(self) -> DecisionQuestion:
        if self.type == "choice":
            if self.levels is not None:
                raise ValueError("a `choice` question takes no `levels`")
            if not self.choices or not (2 <= len(self.choices) <= 255):
                raise ValueError("a `choice` question needs `choices` with 2-255 entries")
            values = [c.value for c in self.choices]
            if len({repr(v) for v in values}) != len(values):
                raise ValueError("choice values must be unique")
            if len({_choice_key(v) for v in values}) != len(values):
                raise ValueError(
                    "boolean and string choice values with the same text are distinct in "
                    "OpenAI's spec but indistinguishable to the model; use one or the other"
                )
        elif self.type == "score":
            if self.choices is not None:
                raise ValueError("a `score` question takes no `choices`")
            if not self.levels or not (2 <= len(self.levels) <= 10):
                raise ValueError("a `score` question needs `levels` with 2-10 entries")
        elif self.choices is not None or self.levels is not None:
            raise ValueError("a `predicate` question takes no `choices` or `levels`")
        return self


class DecisionCacheBreakpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["explicit"] = "explicit"


class DecisionInputPart(BaseModel):
    """`prompt_cache_breakpoint` is accepted and ignored: vLLM's prefix cache reuses
    every full block of a matching prefix on its own, with no breakpoints to place."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["input_text", "input_image"]
    text: str | None = None
    image_url: str | None = None
    prompt_cache_breakpoint: DecisionCacheBreakpoint | None = None


class DecisionInputMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["message"] = "message"
    role: Literal["user"]
    content: str | list[DecisionInputPart]


class DecisionRequest(BaseModel):
    """OpenAI's Decisions request body. `prompt_cache_key`/`prompt_cache_options`/
    `safety_identifier` are accepted (the official SDKs send them) and ignored -- the
    engine's prefix cache is shared automatically, and access control belongs to
    whatever hosts this container, the same stance as the systemone route's 401/429."""

    model_config = ConfigDict(extra="forbid")

    model: str
    input: str | list[DecisionInputMessage]
    questions: list[DecisionQuestion]

    prompt_cache_key: str | None = None
    prompt_cache_options: Any = None
    safety_identifier: str | None = None

    @model_validator(mode="after")
    def _check_questions(self) -> DecisionRequest:
        # OpenAI: "Ask 1-200 questions about the input. ... Answers come back in the
        # same order."
        if not (1 <= len(self.questions) <= 200):
            raise ValueError("`questions` must contain 1-200 entries")
        return self


# --------------------------------------------------------------------------------------------
# Body-level validation: jev's 400s (bad UTF-8/nesting/non-finite), 413, 415 -- none of these
# are free in a plain FastAPI route (`request: SystemOneRequest` would 400 a schema violation
# too, not 422 as jev's own table wants; and `json.loads` accepts NaN/Infinity by default), so
# we read+parse the body ourselves before touching pydantic.
# --------------------------------------------------------------------------------------------


def _check_content_headers(headers) -> tuple[int, str] | None:
    """Content-Encoding (415) and a Content-Length-header fast path for 413. Returns
    `(status_code, message)` on violation, else None. `headers` only needs `.get`, so
    this is testable with a plain dict as well as Starlette's real `Headers`."""
    if headers.get("content-encoding"):
        return (HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Encoding is not supported")
    content_length = headers.get("content-length")
    if content_length is not None:
        try:
            n = int(content_length)
        except ValueError:
            n = None
        if n is not None and n > MAX_BODY_BYTES:
            return (
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"request body exceeds {MAX_BODY_BYTES} bytes",
            )
    return None


def _reject_json_constant(token: str) -> None:
    """`parse_constant` callback: json.loads's one hook for the literal tokens `NaN`,
    `Infinity`, `-Infinity`. Overflowing numeric literals like `1e400` parse to `inf`
    without going through this hook at all, which is why `_check_json_value` below
    ALSO does a recursive `math.isfinite` scan over every parsed number."""
    raise ValueError(f"non-finite number literal {token!r} is not allowed")


def _check_json_value(value: Any, depth: int = 0) -> None:
    """Recursively enforce jev's nesting (<=64 levels anywhere in the body) and
    finite-number rules on already-parsed JSON. Raises ValueError on either
    violation."""
    if depth > MAX_NESTING_DEPTH:
        raise ValueError(f"request body nesting exceeds {MAX_NESTING_DEPTH} levels")
    if isinstance(value, dict):
        for v in value.values():
            _check_json_value(v, depth + 1)
    elif isinstance(value, list):
        for v in value:
            _check_json_value(v, depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"number {value!r} is not finite")


def _parse_strict_json(body: bytes) -> Any:
    """UTF-8 (no BOM) + JSON, rejecting non-finite numbers and excess nesting. Raises
    ValueError on any violation -- the route maps that to HTTP 400."""
    if body[:3] == b"\xef\xbb\xbf":
        raise ValueError("request body must not start with a UTF-8 BOM")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"request body is not valid UTF-8: {e}") from e
    try:
        data = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as e:
        raise ValueError(f"request body is not valid JSON: {e}") from e
    _check_json_value(data)
    return data


def _validation_error_response(e: PydanticValidationError) -> JSONResponse:
    """jev's documented 422 body (`HTTPValidationError`, from the real OpenAPI spec --
    its prose docs never show this shape, only the schema does):
    `{"detail": [{"loc", "msg", "type", "input"?, "ctx"?}, ...]}`. That is pydantic's
    own `ValidationError.errors()` shape already, apart from the extra `url` key
    pydantic adds that jev doesn't document, so we only pass through the keys jev's
    `ValidationError` schema actually has."""
    detail = []
    for err in e.errors():
        # jev's own loc examples are rooted at "body" (e.g. ["body", "state"]) --
        # the FastAPI convention of naming the request-body parameter as the first
        # path segment. We parse the body ourselves rather than through a FastAPI
        # Body(...) parameter, so pydantic's own loc starts one level in; prepend
        # "body" to match.
        item = {"loc": ["body", *err["loc"]], "msg": err["msg"], "type": err["type"]}
        if "input" in err:
            item["input"] = err["input"]
        if "ctx" in err:
            # For a custom @model_validator's raised ValueError, pydantic puts the
            # exception object itself (not a string) at ctx["error"] -- stringify it
            # so the body stays JSON-serializable; everything else in ctx is already
            # plain data.
            item["ctx"] = {k: (str(v) if isinstance(v, BaseException) else v) for k, v in err["ctx"].items()}
        detail.append(item)
    return JSONResponse(content={"detail": detail}, status_code=int(HTTPStatus.UNPROCESSABLE_ENTITY))


def _error(status: HTTPStatus, message: str) -> JSONResponse:
    """For the status codes jev's OpenAPI spec does NOT give a documented body shape
    for (400/413/415 -- only 422's HTTPValidationError is in the spec); see
    `_validation_error_response` for 422 itself."""
    return JSONResponse(
        content={"error": {"message": message, "code": int(status)}},
        status_code=int(status),
    )


def _openai_error(status: HTTPStatus, message: str, param: str | None = None, code: str | None = None) -> JSONResponse:
    """OpenAI's documented error body for the /v1/decisions route
    (`{"error": {"message", "type", "param", "code"}}`): the official SDKs parse
    this shape on this route, which is the point of speaking it."""
    return JSONResponse(
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": code,
            }
        },
        status_code=int(status),
    )


# Context-limit guard: counts state, instructions and options, so it rejects oversized
# requests before any engine work. Template boilerplate is not counted; vLLM's own
# prompt-length check is the backstop.


def _text_of(x: Any) -> str:
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return "" if x is None else str(x)


def _count_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _enforce_context_limits(
    tok,
    max_model_len: int,
    danger_zone: bool,
    state: Any,
    questions: dict[str, Question],
) -> None:
    per_question_cap = min(
        DANGER_ZONE_TOKEN_LIMIT if danger_zone else DEFAULT_PER_QUESTION_TOKEN_LIMIT,
        max_model_len,
    )
    total_cap = min(
        DANGER_ZONE_TOKEN_LIMIT if danger_zone else DEFAULT_PER_REQUEST_TOKEN_LIMIT,
        max_model_len,
    )
    state_text = _text_of(state) if state not in ("", None, {}, []) else "(empty)"
    state_tokens = _count_tokens(tok, state_text)
    total = 0
    for name, q in questions.items():
        options = q.criteria.items() if isinstance(q.criteria, dict) else enumerate(q.criteria or [])
        option_text = "\n".join(f"{key} {_text_of(desc)}" for key, desc in options)
        n = state_tokens + _count_tokens(tok, f"{_text_of(q.instructions)}\n{option_text}")
        if n > per_question_cap:
            raise ValueError(
                f"question {name!r} is approximately {n} tokens, exceeding the "
                f"{per_question_cap}-token maximum context length per question "
                f"(max_tokens_exceeded)"
            )
        total += n
    if total > total_cap:
        raise ValueError(
            f"request is approximately {total} tokens total, exceeding the "
            f"{total_cap}-token maximum context length per request "
            f"(max_tokens_exceeded)"
        )


# --------------------------------------------------------------------------------------------
# jev's own confidence formulas (vendored from the official
# typesafe-ai/system-one-adapter-python `_utils/confidence_metrics.py`, not our own
# invention) -- generic to jev's wire format, not specific to any adapter.
# --------------------------------------------------------------------------------------------


def _softmax(logps: list[float]) -> list[float]:
    m = max(logps)
    exps = [math.exp(x - m) for x in logps]
    s = sum(exps)
    return [v / s for v in exps]


def _normalize_probs(probs: list[float]) -> list[float]:
    """Shared normalization step both confidence formulas below start from --
    uniform if the total is zero."""
    total = sum(probs)
    if total == 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]


def _choice_confidence(probs: list[float]) -> float:
    """jev's actual `confidence` formula for a choice answer -- NOT our own
    invention, and NOT top-1 probability: the top probability rescaled so a uniform
    distribution gives 0 and all mass on one option gives 1,
    `(max(p) - 1/n) / (1 - 1/n)`."""
    if len(probs) == 1:
        return 1.0
    normalized = _normalize_probs(probs)
    uniform = 1.0 / len(normalized)
    return (max(normalized) - uniform) / (1.0 - uniform)


def _score_confidence(probs: list[float]) -> float:
    """jev's actual `confidence` formula for a score answer: 1 minus the expected
    distance from the modal level, divided by the mean distance every level has from
    the scale's midpoint (a uniform distribution's own expected distance). With two
    levels this reduces to the same formula as `_choice_confidence`."""
    if len(probs) == 1:
        return 1.0
    normalized = _normalize_probs(probs)
    mode = max(range(len(normalized)), key=normalized.__getitem__)
    distance_from_mode = sum(p * abs(i - mode) for i, p in enumerate(normalized))
    center = (len(normalized) - 1) / 2
    uniform_mean_distance = sum(abs(i - center) for i in range(len(normalized))) / len(normalized)
    return max(0.0, 1.0 - distance_from_mode / uniform_mean_distance)


# --------------------------------------------------------------------------------------------
# One question -> one jev-shaped answer, over `/v1/chat/completions`'s own serving object.
# --------------------------------------------------------------------------------------------


def _chunk_label_ids(label_ids: list[int]) -> list[list[int]]:
    """Split label ids into `MAX_LABEL_OPTIONS`-sized groups -- one `logprob_token_ids`
    request per group. Pure list math, no engine/vLLM dependency, so it's unit-testable
    without a GPU. The common case (<= MAX_LABEL_OPTIONS labels) returns a single chunk,
    so callers don't need a separate code path for it."""
    return [label_ids[i : i + MAX_LABEL_OPTIONS] for i in range(0, len(label_ids), MAX_LABEL_OPTIONS)]


class TokenUsage(NamedTuple):
    """One question's token counts. `cached_tokens` and `cache_write_tokens` are vLLM's
    prefix-cache read and write counts for the prompt; vLLM reports them only when the
    server runs with `--enable-prompt-tokens-details` (serve.sh sets it), and they are
    0 otherwise."""

    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    cache_write_tokens: int = 0


def _sum_usage(usages: list[TokenUsage]) -> TokenUsage:
    return TokenUsage(*(sum(column) for column in zip(*usages))) if usages else TokenUsage(0, 0)


def _chunk_usage(usage) -> TokenUsage:
    details = usage.prompt_tokens_details
    return TokenUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens or 0,
        cached_tokens=(details.cached_tokens or 0) if details else 0,
        cache_write_tokens=(getattr(details, "created_cache_tokens", None) or 0) if details else 0,
    )


def _error_response_type() -> type:
    """vLLM's ErrorResponse, which moved from `entrypoints/openai/engine/protocol` (0.26-0.28)
    to `entrypoints/serve/engine/protocol` (0.29+); granite-switch 0.1.1 supports 0.26-0.30."""
    try:
        from vllm.entrypoints.serve.engine.protocol import ErrorResponse
    except ImportError:
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
    return ErrorResponse


async def _fetch_label_logprobs(
    chat,
    served_model: str,
    adapter_name: str,
    msgs: list[dict[str, str]],
    label_ids_chunk: list[int],
):
    """One `create_chat_completion` call asking for exactly `label_ids_chunk`'s
    logprobs, for the given (already fully rendered) `msgs`. Returns the raw
    `ChatCompletionResponse`; callers combine multiple chunks' responses when a
    question needs more than `MAX_LABEL_OPTIONS` labels."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
    )

    ErrorResponse = _error_response_type()
    req = ChatCompletionRequest(
        model=served_model,
        messages=msgs,
        max_completion_tokens=1,
        temperature=0.0,
        logprobs=True,
        top_logprobs=1,
        logprob_token_ids=label_ids_chunk,
        return_tokens_as_token_ids=True,
        chat_template_kwargs={"adapter_name": adapter_name},
    )
    resp = await chat.create_chat_completion(req)
    if isinstance(resp, ErrorResponse):
        raise RuntimeError(resp.error.message)
    assert isinstance(resp, ChatCompletionResponse)
    return resp


async def _answer_question(
    name: str,
    q: Question,
    state: Any,
    chat,
    tok,
    served_model: str,
    adapter_name: str,
    renderer,
) -> tuple[str, dict[str, Any], tuple[int, int]]:
    """Returns (name, jev-shaped answer dict, this question's TokenUsage)."""
    if q.type == "choice":
        keys = list(q.criteria)
        # An option without a description is described by its key, as vLLM's native strategy does.
        descs = [k if q.criteria[k] in (None, "") else q.criteria[k] for k in keys]
    elif q.type == "noul":
        crit = q.criteria if isinstance(q.criteria, dict) else {}
        keys = ["false", "true"]
        descs = [crit.get("false") or "No / false", crit.get("true") or "Yes / true"]
    else:  # "score"
        descs = list(q.criteria)
        keys = [str(i) for i in range(len(descs))]

    msgs, label_ids = renderer(tok, state, q.instructions, descs)

    # A choice with more than MAX_LABEL_OPTIONS options (jev allows up to 255) chunks
    # into multiple create_chat_completion calls over the IDENTICAL prompt -- vLLM's own
    # prefix cache, not anything this plugin does, is what makes the second-and-later
    # chunks cheap. Fired concurrently; vLLM's scheduler decides each chunk's cache
    # hit/miss on its own.
    responses = await asyncio.gather(
        *(
            _fetch_label_logprobs(chat, served_model, adapter_name, msgs, chunk)
            for chunk in _chunk_label_ids(label_ids)
        )
    )

    by_id: dict[int, float] = {}
    for resp in responses:
        top_logprobs = resp.choices[0].logprobs.content[0].top_logprobs
        by_id.update({int(t.token.split(":", 1)[1]): t.logprob for t in top_logprobs})
    # Every chunk is a separate engine request over the same prompt, so usage counts each.
    usage = _sum_usage([_chunk_usage(resp.usage) for resp in responses])

    probs = _softmax([by_id[tid] for tid in label_ids])

    if q.type == "choice":
        j = max(range(len(keys)), key=probs.__getitem__)
        answer = {
            "type": "choice",
            "choice": keys[j],
            "probabilities": dict(zip(keys, probs)),
            "confidence": _choice_confidence(probs),
        }
    elif q.type == "noul":
        answer = {"type": "noul", "noul": probs[1]}
    else:  # "score"
        answer = {
            "type": "score",
            "score": sum(i * p for i, p in enumerate(probs)),
            "legend": {str(i): d for i, d in enumerate(descs)},
            "probabilities": dict(zip(keys, probs)),
            "confidence": _score_confidence(probs),
        }

    return name, answer, usage


async def _run_until_disconnect(request: Request, work):
    """Await `work`, cancelling it if the client disconnects first (returns None then).
    Same pattern as vLLM's `with_cancellation`, started after the body has been read."""

    async def disconnected() -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass

    work_task = asyncio.ensure_future(work)
    watch_task = asyncio.ensure_future(disconnected())
    done, _ = await asyncio.wait([work_task, watch_task], return_when=asyncio.FIRST_COMPLETED)
    if work_task in done:
        watch_task.cancel()
        return work_task.result()
    work_task.cancel()
    await asyncio.gather(work_task, return_exceptions=True)
    return None


async def _answer_all(state, state_value: Any, questions: list[tuple[str, Question]]) -> list:
    """Answer every question concurrently; if one fails, cancel the rest and re-raise."""
    tasks = [
        asyncio.ensure_future(
            _answer_question(
                key,
                q,
                state_value,
                state.systemone_chat,
                state.systemone_tokenizer,
                state.systemone_served_model,
                state.systemone_adapter_name,
                state.systemone_renderer,
            )
        )
        for key, q in questions
    ]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        await _cancel_pending(tasks)
        raise


async def _cancel_pending(tasks: list[asyncio.Task]) -> None:
    """Cancel every task that hasn't finished yet, then wait for the cancellation to
    actually be processed by the event loop -- a bare `.cancel()` only *requests*
    cancellation; a task destroyed before that request is processed can log a "Task
    was destroyed but it is pending" warning. Safe to call with already-finished
    tasks mixed in: cancelling a done task is a no-op, and `return_exceptions=True`
    swallows the CancelledError the cancelled ones raise."""
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _decisions_input_to_state(data: str | list[DecisionInputMessage]) -> str:
    """Flatten OpenAI's `input` (a string, or user messages whose content is a string
    or text/image parts) into the state string both routes share: message texts joined
    with blank lines, the same flattening LiteLLM uses when translating between the two
    formats. Image parts are rejected with a clear error -- this is a text decision
    adapter, the same stance as the systemone route's `box`."""
    if isinstance(data, str):
        return data
    texts: list[str] = []
    for message in data:
        if isinstance(message.content, str):
            texts.append(message.content)
            continue
        for part in message.content:
            if part.type == "input_image":
                raise ValueError("image input is not supported: this server serves a text decision adapter")
            if part.text is None:
                raise ValueError("an `input_text` part needs a `text` string")
            texts.append(part.text)
    return "\n\n".join(texts)


def _decisions_question_to_internal(question: DecisionQuestion, index: int) -> tuple[str, Question, dict[str, Any]]:
    """Translate one OpenAI Decisions question into the internal (jev-shaped) Question
    both routes share, plus the metadata needed to map its answer back. The internal key
    is always the positional index -- OpenAI's answers are positional, so duplicate or
    missing `name`s cannot collide (the response echoes the original name or null).

    Mapping: `predicate` -> noul with the trained default true/false descriptions
    (OpenAI's predicate has no description fields); `choice` -> a criteria object keyed
    by the coerced option keys (description optional, as in jev); `score` -> the level
    descriptions, with a level's label standing in when it has no description (jev's
    score criteria cannot be null)."""
    if question.type == "predicate":
        internal = Question(type="noul", instructions=question.instructions)
        extras: dict[str, Any] = {}
    elif question.type == "choice":
        criteria = {_choice_key(c.value): c.description for c in question.choices}
        internal = Question(type="choice", instructions=question.instructions, criteria=criteria)
        extras = {"value_map": {_choice_key(c.value): c.value for c in question.choices}}
    else:
        descriptions = [
            level.description if level.description is not None else level.label for level in question.levels
        ]
        internal = Question(type="score", instructions=question.instructions, criteria=descriptions)
        extras = {"labels": [level.label for level in question.levels]}
    return str(index), internal, extras


def _internal_answer_to_decisions(
    question: DecisionQuestion, extras: dict[str, Any], answer: dict[str, Any]
) -> dict[str, Any]:
    """Map one internal (jev-shaped) answer onto OpenAI's Decisions answer shapes:
    `predicate` -> `probability` (the internal noul answer is already P(true));
    `choice` -> the picked value plus per-value probabilities and the shared confidence
    statistic; `score` -> the expected level over per-level probabilities, carrying the
    ORIGINAL level labels (the internal legend holds the descriptions we rendered)."""
    if question.type == "predicate":
        return {"type": "predicate", "name": question.name, "probability": answer["noul"]}
    if question.type == "choice":
        value_map: dict[str, str | bool] = extras["value_map"]
        return {
            "type": "choice",
            "name": question.name,
            "choice": value_map[answer["choice"]],
            "probabilities": [
                {"value": value_map[key], "probability": probability}
                for key, probability in answer["probabilities"].items()
            ],
            "confidence": answer["confidence"],
        }
    labels: list[str] = extras["labels"]
    return {
        "type": "score",
        "name": question.name,
        "score": answer["score"],
        "probabilities": [
            {
                "value": index,
                "label": labels[index],
                "probability": answer["probabilities"][str(index)],
            }
            for index in range(len(labels))
        ],
        "confidence": answer["confidence"],
    }


async def _handle_systemone(raw_request: Request) -> Response:
    state = raw_request.app.state
    chat = getattr(state, "systemone_chat", None)
    if chat is None:
        return _error(HTTPStatus.NOT_IMPLEMENTED, "the model does not support generation")

    header_check = _check_content_headers(raw_request.headers)
    if header_check is not None:
        code, msg = header_check
        return _error(HTTPStatus(code), msg)

    body = await raw_request.body()
    if len(body) > MAX_BODY_BYTES:
        return _error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"request body exceeds {MAX_BODY_BYTES} bytes")

    try:
        data = _parse_strict_json(body)
    except ValueError as e:
        return _error(HTTPStatus.BAD_REQUEST, str(e))

    try:
        req = SystemOneRequest.model_validate(data)
    except PydanticValidationError as e:
        return _validation_error_response(e)

    danger_zone = raw_request.headers.get(DANGER_ZONE_HEADER, "").strip().lower() == "true"
    try:
        _enforce_context_limits(
            state.systemone_tokenizer,
            state.systemone_max_model_len,
            danger_zone,
            req.state,
            req.questions,
        )
    except ValueError as e:
        return _error(HTTPStatus.BAD_REQUEST, str(e))

    results = await _run_until_disconnect(raw_request, _answer_all(state, req.state, list(req.questions.items())))
    if results is None:
        return Response(status_code=499)

    answers = {name: answer for name, answer, _ in results}
    usage = _sum_usage([question_usage for _, _, question_usage in results])

    return JSONResponse(
        content={
            "model": state.systemone_served_model,
            "answers": answers,
            "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
        }
    )


async def _handle_decisions(raw_request: Request) -> Response:
    """POST /v1/decisions: OpenAI's Decisions wire format over the same engine path as
    /v1/systemone. The input flattens to the shared state string, the questions
    translate to the shared internal Question shape, and the answers map back -- the
    parity tests in tests/test_decisions.py assert the translation converges on exactly
    what the systemone route builds for the equivalent request, so both routes render
    byte-identical prompts and read the same label logits."""
    state = raw_request.app.state
    chat = getattr(state, "systemone_chat", None)
    if chat is None:
        return _openai_error(HTTPStatus.NOT_IMPLEMENTED, "the model does not support generation")

    header_check = _check_content_headers(raw_request.headers)
    if header_check is not None:
        code, msg = header_check
        return _openai_error(HTTPStatus(code), msg)

    body = await raw_request.body()
    if len(body) > MAX_BODY_BYTES:
        return _openai_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"request body exceeds {MAX_BODY_BYTES} bytes")

    try:
        data = _parse_strict_json(body)
    except ValueError as e:
        return _openai_error(HTTPStatus.BAD_REQUEST, str(e))

    try:
        req = DecisionRequest.model_validate(data)
    except PydanticValidationError as e:
        return _openai_error(
            HTTPStatus.BAD_REQUEST,
            "; ".join(f"{'.'.join(str(loc) for loc in err['loc']) or 'request'}: {err['msg']}" for err in e.errors()),
        )

    try:
        state_text = _decisions_input_to_state(req.input)
    except ValueError as e:
        return _openai_error(HTTPStatus.BAD_REQUEST, str(e))

    translated = [_decisions_question_to_internal(question, index) for index, question in enumerate(req.questions)]

    try:
        _enforce_context_limits(
            state.systemone_tokenizer,
            state.systemone_max_model_len,
            False,  # no danger-zone concept on the OpenAI route
            state_text,
            {key: internal for key, internal, _ in translated},
        )
    except ValueError as e:
        return _openai_error(HTTPStatus.BAD_REQUEST, str(e))

    work = _answer_all(state, state_text, [(key, internal) for key, internal, _ in translated])
    results = await _run_until_disconnect(raw_request, work)
    if results is None:
        return Response(status_code=499)

    answers = [
        _internal_answer_to_decisions(req.questions[int(key)], extras, answer)
        for (_, _, extras), (key, answer, _) in zip(translated, results)
    ]
    usage = _sum_usage([question_usage for _, _, question_usage in results])

    return JSONResponse(
        content={
            "model": state.systemone_served_model,
            "answers": answers,
            "usage": {
                "input_tokens": usage.input_tokens,
                "input_tokens_details": {
                    "cached_tokens": usage.cached_tokens,
                    "cache_write_tokens": usage.cache_write_tokens,
                },
                "output_tokens": usage.output_tokens,
                # A one-token label read never reasons.
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": usage.input_tokens + usage.output_tokens,
            },
        }
    )


router = APIRouter()


@router.post("/v1/systemone")
async def create_systemone(raw_request: Request) -> JSONResponse:
    return await _handle_systemone(raw_request)


@router.post("/v1/decisions")
async def create_decisions(raw_request: Request) -> JSONResponse:
    return await _handle_decisions(raw_request)


class SystemOnePlugin:
    """`vllm.endpoint_plugins` entry point (see module docstring). Loaded only when
    `granite_switch_systemone` is in `VLLM_PLUGINS` -- see this repo's README/Dockerfile."""

    name = "granite_switch_systemone"
    required_tasks: tuple[str, ...] | None = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        app.include_router(router)

    async def init_state(self, engine_client: EngineClient | None, state: State, args: Namespace) -> None:
        # `required_tasks = ("generate",)` already guarantees "generate" is supported, so
        # `openai_serving_chat` (set by vllm's own init_generate_state, which runs before any
        # endpoint plugin's init_state) is never None here; the getattr/None check is just
        # defense in depth, matching the EndpointPlugin contract's own guidance.
        chat = getattr(state, "openai_serving_chat", None)
        if chat is None:
            return
        state.systemone_chat = chat
        state.systemone_tokenizer = chat.renderer.tokenizer
        state.systemone_max_model_len = chat.model_config.max_model_len
        state.systemone_served_model = args.served_model_name[0] if args.served_model_name else args.model
        # Deliberately NOT wrapped in try/except: a mismatched adapter name, or an
        # unresolvable renderer, must fail server startup, not be swallowed into a
        # plugin that silently serves wrong answers for /v1/systemone's entire
        # lifetime (see module docstring).
        state.systemone_adapter_name = _resolve_adapter_name(chat)
        state.systemone_renderer = _load_renderer()


def register() -> SystemOnePlugin:
    return SystemOnePlugin()
