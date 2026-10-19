# Status: what is tested, and what is not

Last updated 2026-10-19. "Live" means a real server: vLLM 0.30.0, granite-switch 0.1.1,
transformers 5.17, `ibm-granite/granite-switch-4.1-8b`, one H100.

## Tested

| Area | How | Result |
|---|---|---|
| Unit and route tests | `pytest`, engine stubbed, in CI | pass |
| `/v1/decisions` against OpenAI's OpenAPI schema | `tests/test_openai_spec.py` (42 cases), plus the same request set live | pass, with the exceptions listed in the README |
| `/v1/systemone` against jev's OpenAPI schema (api.typesafe.ai, v0.2.0) | a JSON Schema check of 24 cases, stubbed and live; the spec has no licence, so it is not copied into this repo | pass; the plugin also rejects unknown fields, which jev's schema allows |
| Live wire battery | `scripts/e2e_check.py` (11 checks) | 11/11 |
| Prompt rendering | server prompt token counts against an offline render with the checkpoint's tokenizer and chat template | identical |
| Prefix-cache sharing | `tests/test_prefix_sharing.py`, and `cached_tokens` live, cross-checked with `/metrics` | as documented in the README |
| Route parity | 800 labelled items through `/v1/decisions` and through the same prompt over chat completions | no item answers differently |
| Escaping, size guard, disconnect cancellation, error shapes | live probes | as documented in the README |
| Load | 1 to 128 concurrent clients, three-question requests | about 140 requests/s plateau, no errors |
| Plugin-only install | the README's steps in a fresh environment, then its `vllm serve` command and the e2e battery | pass |

## Not tested

- **The Docker image.** The Dockerfile has not been built or run with the current
  defaults. Until it is, prefer the plugin-only install.
- **vLLM 0.26 to 0.29.** Every live run used 0.30.0. The code paths that differ by
  version (where vLLM keeps `ErrorResponse`) are unit-tested with stand-in modules only.
- **Other checkpoints.** Only the 8b. The 3b and 30b have the same adapter layout but have
  not been served with this plugin.
- **Multi-GPU and tensor parallelism.**
- **FlashAttention 3.** `serve.sh` forces FlashAttention 2 (see the README). FA3 has not
  been measured with granite-switch 0.1.1.
- **Long contexts.** Probes stayed under the 32,768-token `--max-model-len`; the 262k
  long-context header path is not exercised.
- **Accuracy against the published benchmark.** The checks above compare the routes
  with each other and with a system-prompt variant, not with the model card's score.

## Known gaps

- **Decision load slows chat.** Chat p50 for a 32-token reply went from 235 ms idle to
  1,088 ms with 128 concurrent decision clients. There is no cap or priority between the
  two kinds of traffic.
- **`/v1/systemone` has no question limit.** jev's schema sets none, and each question is
  one engine request.
- **`scripts/e2e_check.py` compares routes to 1e-6.** Engine reads are not bit-identical
  between batched and solo requests, so the check can fail on a busy server.
- **No system prompt.** The model card's published score used one; this plugin omits it so
  decisions can share the prefix cache with chat. Our comparisons found it raises accuracy
  on rating (`score`) questions.
