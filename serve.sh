#!/bin/bash
# vLLM's OpenAI-compatible server (/v1/chat/completions, /v1/completions, /health) for a
# composed granite-switch checkpoint at $MODEL_DIR, plus the jev system-one API
# (/v1/systemone) mounted on the SAME server via vLLM's own `vllm.endpoint_plugins`
# mechanism -- `vllm serve` discovers and mounts it on its own; nothing below invokes it
# directly. Override with the env vars below, and pass extra `vllm serve` flags as
# container arguments.
#   MODEL_DIR               default: /repository (HF Inference Endpoints' own mount point;
#                           override for any other deployment target)
#   TENSOR_PARALLEL_SIZE    default: every visible GPU
#   MAX_MODEL_LEN (32768), GPU_MEMORY_UTILIZATION (0.90), PORT (8000), SERVED_MODEL_NAME
#   VLLM_PLUGINS            default (set in the Dockerfile): register_granite_switch,
#                           granite_switch_systemone -- BOTH names are required (see
#                           Dockerfile comment); drop the second to disable /v1/systemone.
#   SYSTEMONE_ADAPTER_NAME, SYSTEMONE_INVOCATION_MARKER, SYSTEMONE_RENDERER
#                           see src/granite_switch_systemone/plugin.py and renderer.py.
set -e
MODEL_DIR=${MODEL_DIR:-/repository}
TP=${TENSOR_PARALLEL_SIZE:-$(python3 -c "import torch; print(max(torch.cuda.device_count(), 1))")}
# --enable-prompt-tokens-details: vLLM reports each request's prefix-cache hits, which
# /v1/decisions returns as usage.input_tokens_details.cached_tokens (0 without it).
# FlashAttention 2: granite-switch's FA3 ahead-of-time schedule is sized from the model
# config rather than the Switch's own attention layers, and CUDA-graph capture fails
# (generative-computing/granite-switch#139, open as of this writing).
# bfloat16: composed checkpoints carry torch_dtype: null
# (generative-computing/granite-switch#126, open as of this writing), which vLLM would
# otherwise resolve to float32.
exec vllm serve "$MODEL_DIR" \
    --served-model-name "${SERVED_MODEL_NAME:-granite-switch}" \
    --host 0.0.0.0 --port "${PORT:-8000}" \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --attention-config.flash_attn_version=2 \
    --max-logprobs 256 \
    --enable-prompt-tokens-details \
    --override-generation-config '{"temperature": 0.0}' \
    --enable-auto-tool-choice --tool-call-parser granite4 \
    "$@"
