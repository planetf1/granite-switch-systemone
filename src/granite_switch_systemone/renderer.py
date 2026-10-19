# SPDX-License-Identifier: Apache-2.0
"""The prompt for a Granite Switch decision adapter: two user turns,

    State:\n<state>
    <marker>\nQuestion:\n<instructions>\n\nOptions:\n<label>: <description>\n...
    \nReply with exactly one option label.

answered by the next token, read as a softmax over the labels' logits. This matches the
layout on the model card. `SYSTEMONE_INVOCATION_MARKER` has no default: if the marker is
wrong, the chat template activates the adapter in a different place without raising.
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
            f"{_INVOCATION_MARKER_ENV} is not set: set it to the text the decision adapter "
            'was trained to activate on (e.g. "### DECISION").'
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
TOKEN_BREAK = "\u2060"


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
    """Messages and label token ids for one question; `label_token_ids[i]` answers `descriptions[i]`."""
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
