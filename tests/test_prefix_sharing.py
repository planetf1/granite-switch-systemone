"""Token-level prefix sharing between chat completions and this plugin's decision prompts.

vLLM's prefix cache reuses a KV block only when every token up to and including it
matches. Granite Switch activates an adapter with a control token inside the sequence,
not a LoRA request, and the plugin sends no cache salt or multimodal input, so the
block hash is the tokens alone (vLLM v0.26 `kv_cache_utils.need_extra_keys`). These
tests render the plugin's real decision prompt with the served checkpoint's tokenizer
and chat template, and count what a decision shares with other requests about the same
state. They bound what the cache can reuse; the live `prefix_cache` check in
`scripts/e2e_check.py` measures what it does reuse.

If one of the "shares no full block" tests starts failing, the decision layout changed:
that is a cache win, but the layout is no longer the trained one, so re-check accuracy.

  SYSTEMONE_TOKENIZER=ibm-granite/granite-switch-4.1-8b python3 -m pytest tests/test_prefix_sharing.py -v

Needs `transformers` new enough for the checkpoint's tokenizer class (5.16 works);
skipped when it is missing or SYSTEMONE_TOKENIZER is unset.
"""

import os

import pytest

from granite_switch_systemone import renderer

transformers = pytest.importorskip("transformers")

TOKENIZER = os.environ.get("SYSTEMONE_TOKENIZER")
pytestmark = pytest.mark.skipif(not TOKENIZER, reason="SYSTEMONE_TOKENIZER not set")

BLOCK_SIZE = 16  # vLLM CacheConfig.DEFAULT_BLOCK_SIZE; serve.sh does not override it
MARKER = "### DECISION"
CONTROL_TOKEN = "<|decision|>"
STATE = " ".join(
    [
        "My payouts to the business account have failed three times this month.",
        "The bank says nothing is wrong on their side and the account details are correct.",
        "Each failure arrived as an email with no reason code, and support has not replied.",
        "I have payroll due on Friday and cannot cover it if this payout fails again.",
    ]
    * 3
)


@pytest.fixture(scope="module")
def tok():
    return transformers.AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture(autouse=True)
def _marker(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_INVOCATION_MARKER", MARKER)


def _ids(tok, messages, add_generation_prompt=True, **template_kwargs):
    """Token ids as vLLM's chat path builds them: the server's chat template, then
    tokenization without added special tokens (ChatCompletionRequest defaults)."""
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, **template_kwargs
    )
    return tok(text, add_special_tokens=False)["input_ids"]


def _decision(tok, question):
    """The plugin's prompt for one predicate question, with the chat_template_kwargs
    `_fetch_label_logprobs` sends. Asserts the adapter actually activated: the chat
    template falls back silently when the marker is missing."""
    messages, _ = renderer.render(tok, STATE, question, ["No / false", "Yes / true"])
    ids = _ids(tok, messages, adapter_name="decision")
    assert ids.count(tok.convert_tokens_to_ids(CONTROL_TOKEN)) == 1
    return ids


def _state_turn_len(tok):
    return len(_ids(tok, [{"role": "user", "content": "State:\n" + STATE}], add_generation_prompt=False))


def _shared(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_state_spans_several_blocks(tok):
    """Guards the other tests: a state shorter than one block would make every
    "no full block" assertion pass vacuously."""
    assert _state_turn_len(tok) >= 4 * BLOCK_SIZE


def test_two_decisions_about_one_state_share_the_state_turn(tok):
    """The reuse the plugin relies on for multi-question requests and >128-option chunks."""
    shared = _shared(_decision(tok, "Does this need urgent attention?"), _decision(tok, "Is the customer angry?"))
    assert shared >= _state_turn_len(tok)


def test_chat_whose_first_turn_is_the_state_turn_shares_it(tok):
    """The README's sharing claim holds only in this shape: the chat's first user turn
    is literally `State:\\n<state>` and nothing (no system prompt) precedes it."""
    chat = _ids(
        tok,
        [
            {"role": "user", "content": "State:\n" + STATE},
            {"role": "user", "content": "Summarise this ticket in one sentence."},
        ],
    )
    assert _shared(chat, _decision(tok, "Does this need urgent attention?")) >= _state_turn_len(tok)


def test_chat_about_the_raw_state_shares_no_full_block(tok):
    """A chat that sends the state as-is diverges at the trained `State:` prefix,
    a few tokens in, so not one block is reusable in either direction."""
    chat = _ids(tok, [{"role": "user", "content": STATE}])
    assert _shared(chat, _decision(tok, "Does this need urgent attention?")) < BLOCK_SIZE


def test_chat_with_its_own_system_prompt_shares_no_full_block(tok):
    """The decision prompt has no system turn, so any chat that has one diverges at its
    first content token."""
    chat = _ids(
        tok,
        [
            {"role": "system", "content": "You are a helpful payments support agent."},
            {"role": "user", "content": "State:\n" + STATE},
        ],
    )
    assert _shared(chat, _decision(tok, "Does this need urgent attention?")) < BLOCK_SIZE
