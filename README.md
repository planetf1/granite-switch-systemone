# granite-switch-systemone

A vLLM endpoint plugin that serves the decision adapter of a
[Granite Switch](https://github.com/generative-computing/granite-switch) checkpoint through
two decision APIs, on the same server and GPU cache as `/v1/chat/completions`:

- `POST /v1/decisions`: [OpenAI's Decisions API](https://developers.openai.com/api/docs/guides/decisions).
  The official OpenAI SDKs work by changing `base_url`.
- `POST /v1/systemone`: [jev's system-one API](https://docs.typesafe.ai/api).

Each question is answered by one engine read of the adapter's single-token option labels,
returned as calibrated probabilities. Callers need no prompt format, marker or label scheme.

## Install

### Docker (recommended)

```bash
docker build --platform linux/amd64 -t granite-switch-systemone .
docker run -d --gpus all -p 8000:8000 \
    -v /path/to/granite-switch-4.1-8b:/repository:ro \
    granite-switch-systemone
```

The image is `vllm/vllm-openai:v0.26.0-cu129` plus `granite-switch[vllm]==0.1.1`,
`transformers==5.17.0` and this plugin. Weights are not baked in: mount a downloaded
checkpoint at `/repository`, or set `MODEL_DIR`. No image is published yet.

### Into an existing vLLM environment

Prerequisites:

- Linux with an NVIDIA GPU; the 8b fits on one 80 GB H100.
- Python 3.11+, PyTorch 2.11+ (granite-switch requires it; CUDA 13 builds tested).
- vLLM 0.26 to 0.30. That range comes from granite-switch 0.1.1, which pins
  `vllm>=0.26,<0.31`; this plugin's vLLM APIs are present across all of it.

```bash
pip install "granite-switch[vllm]==0.1.1" "transformers>=5.16"
pip install git+https://github.com/planetf1/granite-switch-systemone@feat/v1-decisions-endpoint
export VLLM_PLUGINS=register_granite_switch,granite_switch_systemone
export SYSTEMONE_INVOCATION_MARKER="### DECISION"
vllm serve ibm-granite/granite-switch-4.1-8b --served-model-name granite-switch \
    --dtype bfloat16 --attention-config.flash_attn_version=2 --enable-prompt-tokens-details
```

`VLLM_PLUGINS` must name both plugins. vLLM loads endpoint plugins only when they are
listed, and once the variable is set it loads nothing else, so leaving out
`register_granite_switch` stops the Switch model class from registering.

## Server flags

`serve.sh` (the image's entrypoint) runs `vllm serve` with these non-default flags:

| Flag | Why |
|---|---|
| `--dtype bfloat16` | Composed checkpoints carry `torch_dtype: null` ([granite-switch#126](https://github.com/generative-computing/granite-switch/issues/126)); vLLM would otherwise load float32. |
| `--attention-config.flash_attn_version=2` | Avoids an FA3 schedule bug with CUDA graphs and prefix caching ([granite-switch#139](https://github.com/generative-computing/granite-switch/issues/139), open). |
| `--enable-prompt-tokens-details` | vLLM reports each request's prefix-cache hits; `/v1/decisions` returns them as `usage.input_tokens_details`. Without it both counts are 0. |
| `--max-logprobs 256` | For chat clients that read labels by top-K logprobs (vLLM's default cap is 20). The plugin does not need it. |
| `--enable-auto-tool-choice --tool-call-parser granite4` | Tool calling on the shared chat route. |
| `--served-model-name`, `--max-model-len 32768`, `--gpu-memory-utilization 0.90`, `--tensor-parallel-size` | Overridable with `SERVED_MODEL_NAME`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `TENSOR_PARALLEL_SIZE`. |

There is no server-wide temperature override, so chat requests keep the model's defaults.
Prefix caching stays at vLLM's default (on).

### What the plugin sets on every decision request

Decisions need a deterministic one-token read, so the plugin sends these itself and
callers cannot change them:

- `temperature=0.0` and `max_completion_tokens=1`
- `logprob_token_ids` set to the option labels' token ids, read exactly rather than from
  the top-K. vLLM caps this at 128 ids, so a `choice` with more options runs as several
  requests over the same prompt, served from the prefix cache.
- `chat_template_kwargs={"adapter_name": <SYSTEMONE_ADAPTER_NAME>}`, which activates the adapter

## Configuration

| Env var | Default | What it does |
|---|---|---|
| `SYSTEMONE_ADAPTER_NAME` | the only adapter, else `decision` | Which adapter answers. Checked against the checkpoint's adapter list at startup; an unknown name stops the server. |
| `SYSTEMONE_INVOCATION_MARKER` | none (`### DECISION` in the image) | The text the adapter was trained to activate on. Required: with a wrong marker the chat template activates the adapter in the wrong place without raising. |
| `SYSTEMONE_RENDERER` | `granite_switch_systemone.renderer:render` | `module:function` that builds the prompt and label ids, for a differently trained adapter. |

Both routes share these; nothing is configured per route.

## Calling it

```bash
curl http://localhost:8000/v1/decisions -H "Content-Type: application/json" -d '{
  "model": "granite-switch",
  "input": "My payouts have failed three times. The bank says everything is fine.",
  "questions": [
    {"type": "choice", "name": "queue", "instructions": "Which team should handle this?",
     "choices": [{"value": "payments", "description": "Payout failures"},
                 {"value": "account", "description": "Login issues"}]},
    {"type": "predicate", "name": "escalate", "instructions": "Does this need urgent attention?"}
  ]}'
```

[examples/decisions.md](examples/decisions.md) has the OpenAI SDK, all three question types
and the equivalent `/v1/systemone` request. Both routes build the same prompt for
equivalent questions and return the same probabilities.

## Behaviour and limits

- **Prompt.** Two user turns, `State:\n<state>` then `### DECISION` with the question and
  lettered options (A–Z, then 0–9, then two-letter labels), read as a softmax over the
  label logits. Predicates render "A: No / false", "B: Yes / true". An option without a
  description is described by its value.
- **No system prompt.** This lets a decision share the prefix cache with a chat request
  whose first turn is the same `State:` turn. Any chat with its own system prompt, or
  that sends the state without `State:`, shares nothing; decisions about one state always
  share it with each other. The model card's published score used a system prompt, and in
  our tests it scored higher on rating (`score`) questions without changing yes/no ones.
- **Caller text is escaped.** Special tokens and the invocation marker inside state,
  instructions or descriptions are broken with an invisible U+2060, so they cannot forge
  turns or move the adapter's activation point.
- **Input.** `input` messages must have role `user`, and images are rejected; the trained
  layout has one text state turn. `/v1/systemone` rejects unknown fields and `box`
  questions.
- **Size.** The state, instructions and options of each question are counted before any
  engine work: at most 32k tokens per question and 64k per request (262k with
  `TypeSafe-Danger-Zone-Long-Context: true` on `/v1/systemone`), capped by
  `--max-model-len`. Bodies over 32 MiB, nesting deeper than 64 and non-finite numbers
  are rejected.
- **Cancellation.** A client that disconnects cancels its request's engine work.
- **Errors.** `/v1/decisions` uses OpenAI's error shape, `/v1/systemone` uses jev's
  `HTTPValidationError` for 422s. Engine errors keep vLLM's status code.
- **Not provided.** Authentication (use vLLM's `--api-key`, which covers these routes),
  rate limiting, and `refusal` answers. Engine reads are not bit-identical between
  batched and solo requests, so repeated calls can differ in the third decimal place.
- **vLLM 0.32+** mounts its own `/v1/systemone` and `/v1/decisions`, which do not know
  Switch adapters; granite-switch 0.1.1 does not support those versions yet.

## Testing

```bash
pip install -e ".[dev]" && pytest                      # unit and spec tests, no GPU
python3 scripts/e2e_check.py --base-url http://localhost:8000
```

`tests/test_openai_spec.py` validates `/v1/decisions` against OpenAI's published OpenAPI
schema (`tests/spec/openai_decisions.json`, copied verbatim, MIT). `scripts/e2e_check.py`
checks a live server: both routes' answers and parity, errors, usage, chat on the same
server, and prefix-cache hits. Run it on a server with no other traffic, as it reads
vLLM's server-wide `/metrics`. Set `SYSTEMONE_TOKENIZER` to the checkpoint to run
`tests/test_prefix_sharing.py`, which needs `transformers>=5.16`.

## License

Apache-2.0.
