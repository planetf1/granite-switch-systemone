# SPDX-License-Identifier: Apache-2.0
"""The ONE adapter-specific piece `plugin.py` calls out to: how to turn a jev-shaped
question (`state`, `instructions`, a list of option `descriptions`) into chat messages
and single-token label ids for a SPECIFIC trained decision adapter.

THIS MODULE IS A PLACEHOLDER, NOT A PERMANENT HOME. The real, general version of
"render a prompt for an adapter function and read back a calibrated answer" belongs
in mellea's own inference path (`mellea/formatters/granite/intrinsics/`,
`mellea/backends/adapters/`) alongside every other adapter function (guardian-core,
answerability, ...) -- see that project's `io_contracts.py`/`catalog.py` for the real
pattern (a catalog entry + an IOContract), and `intrinsics/input.py`'s
`IntrinsicsRewriter` for the generic, io.yaml-driven request-rewriting machinery this
hand-written renderer duplicates in a narrower, adapter-specific way. `granite-switch`
itself must never import `mellea` (and should not need to, to avoid a dependency
cycle with mellea's own `granite-switch` dependency); this package exists specifically
so a real jev bridge can ship today, with this one function swapped out for a call
into mellea's own machinery once mellea grows an equivalent primitive (a
label-logit-softmax reader is not yet one of mellea's `IOContract` shapes -- every
existing one parses structured JSON text, not a softmax over single-token logits).

The convention this module implements is the one `decide_alora.py`'s own training
data was built against (`tutorials/scripts/universal_classifier/decide_alora.py` in
`generative-computing/granite-switch`): two user turns --
    turn 1 (context, runs on frozen base model KV): "State:\\n<state>"
    turn 2 (adapter-activated):      "<INVOCATION_MARKER>\\nQuestion:\\n<instructions>\\n\\n
                                      Options:\\n<label>: <description>...\\n\\n
                                      Reply with exactly one option label."
then `add_generation_prompt=True` opens the assistant turn so the next token is the
option label, read as a softmax over the per-label single-token logits.

SYSTEMONE_INVOCATION_MARKER (required, no default): the literal text that activates
the aLoRA adapter in the SECOND turn. There is no way to safely default this: a
request whose second-turn text does NOT contain the adapter's trained marker still
"succeeds" -- granite-switch's chat template silently falls back to a DIFFERENT
activation placement (right before the generation prompt, rather than at the marker)
instead of raising, so a wrong or missing marker produces a plausible-looking but
uncalibrated answer, not an error. Confirmed directly against
`granite_switch.composer.tokenizer_setup`'s generated chat template. The checkpoint
itself cannot be asked for this value post-compose: `adapter_config.json` (which
carries `alora_invocation_tokens`, PEFT's own source of truth for this exact value)
is discarded by the composer today -- see this repo's README for the follow-up issue
filed against `generative-computing/granite-switch` to retain it.
"""

from __future__ import annotations

import json
import os

PLAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_INVOCATION_MARKER_ENV = "SYSTEMONE_INVOCATION_MARKER"


def _invocation_marker() -> str:
    marker = os.environ.get(_INVOCATION_MARKER_ENV)
    if not marker:
        raise RuntimeError(
            f"{_INVOCATION_MARKER_ENV} is not set. This renderer has no safe default: "
            "a wrong or missing invocation marker does not raise an error, it silently "
            "activates the adapter at a DIFFERENT position than training used (see "
            "this module's own docstring). Set it to the exact literal text the "
            'decision adapter was trained to activate on (e.g. "### DECISION").'
        )
    return marker


def text_of(x: object) -> str:
    """`state`/a description may be a dict/list (jev allows `string | object | array`
    for both); render JSON compactly, pass strings through, "" for None."""
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return "" if x is None else str(x)


# U+2060 WORD JOINER: invisible, and splits a special-token string so the tokenizer
# reads it as text rather than as the special token.
TOKEN_BREAK = "⁠"


def escape_special_tokens(text: str, specials: list[str]) -> str:
    """Break every occurrence of the tokenizer's added tokens in caller text, so state,
    instructions and descriptions cannot forge turns or a second adapter activation."""
    for special in specials:
        if special in text:
            text = text.replace(special, special[0] + TOKEN_BREAK + special[1:])
    return text


def _specials(tok) -> list[str]:
    added = getattr(tok, "get_added_vocab", None)
    return sorted(added(), key=len, reverse=True) if added else []


def build_label_pool(tok, need: int) -> list[tuple[str, int]]:
    """`need` single-token labels: A..Z first (readable), then digits, then any
    other two-letter combination, so wide (>26-option) `choice` questions still get
    a valid single-token label pool. Returns `(label_string, token_id)` pairs."""
    pool: list[tuple[str, int]] = []
    seen: set[int] = set()
    candidates = list(PLAIN_LETTERS)
    candidates += list("0123456789")
    candidates += [a + b for a in PLAIN_LETTERS for b in PLAIN_LETTERS]
    for label in candidates:
        if len(pool) >= need:
            break
        ids = tok(label, add_special_tokens=False)["input_ids"]
        if len(ids) == 1 and ids[0] not in seen:
            pool.append((label, ids[0]))
            seen.add(ids[0])
    if len(pool) < need:
        raise ValueError(f"could only build {len(pool)} single-token labels, need {need}")
    return pool[:need]


def render(
    tok, state: object, instructions: object, descriptions: list[object]
) -> tuple[list[dict[str, str]], list[int]]:
    """The renderer interface `plugin.py` calls (via `SYSTEMONE_RENDERER`, a dotted
    `module:function` path -- see `plugin.py`'s module docstring).

    Args:
        tok: the server's own tokenizer (duck-typed: only `tok(text,
            add_special_tokens=False)["input_ids"]` is used).
        state: jev's `state` field -- the content every question in the request
            refers to.
        instructions: this one question's `instructions` field.
        descriptions: this one question's option descriptions, already extracted
            from its `criteria` by `plugin.py` into a flat, ordered list (one entry
            per `choice` option / `score` level / the two `noul` outcomes) -- this
            function never sees jev's `type`/`criteria` shape directly.

    Returns:
        `(messages, label_token_ids)`: `messages` is a plain chat-messages list
        (no system turn: see this module's docstring on why), ready to pass straight
        through as `ChatCompletionRequest.messages`; `label_token_ids[i]` is the
        single token id whose logprob corresponds to `descriptions[i]` being chosen,
        in the same order.
    """
    labels = build_label_pool(tok, len(descriptions))
    marker = _invocation_marker()
    # The chat template activates at the turn's last marker, so caller text must not carry one.
    specials = [*_specials(tok), marker]
    ctx = text_of(state) if state not in ("", None, {}, []) else "(empty)"
    q_lines = ["Question:", escape_special_tokens(text_of(instructions), specials), "", "Options:"]
    for (label, _), description in zip(labels, descriptions):
        desc_text = text_of(description) if description not in (None, "") else label
        q_lines.append(f"{label}: {escape_special_tokens(desc_text, specials)}")
    q_lines += ["", "Reply with exactly one option label."]
    second_turn = marker + "\n" + "\n".join(q_lines)
    messages = [
        {"role": "user", "content": "State:\n" + escape_special_tokens(ctx, specials)},
        {"role": "user", "content": second_turn},
    ]
    return messages, [token_id for _, token_id in labels]
