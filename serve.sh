#!/bin/bash
# vLLM's OpenAI-compatible server for a composed granite-switch checkpoint at $MODEL_DIR,
# with /v1/decisions and /v1/systemone mounted by the endpoint plugin. Extra container
# arguments are passed to `vllm serve`.
#   MODEL_DIR               default /repository (HF Inference Endpoints' mount point)
#   TENSOR_PARALLEL_SIZE    default: every visible GPU
#   MAX_MODEL_LEN (32768), GPU_MEMORY_UTILIZATION (0.90), PORT (8000), SERVED_MODEL_NAME
#   SYSTEMONE_ADAPTER_NAME  default "decision"; SYSTEMONE_INVOCATION_MARKER, see README
set -e
MODEL_DIR=${MODEL_DIR:-/repository}
TP=${TENSOR_PARALLEL_SIZE:-$(python3 -c "import torch; print(max(torch.cuda.device_count(), 1))")}
export SYSTEMONE_ADAPTER_NAME=${SYSTEMONE_ADAPTER_NAME:-decision}
# --dtype bfloat16: composed checkpoints carry torch_dtype null (granite-switch#126).
# flash_attn_version=2: avoids the FA3 schedule bug (granite-switch#139, still open).
# --max-logprobs 256: lets chat clients that read labels by top-K ask for more than 20.
# --enable-prompt-tokens-details: reports prefix-cache hits in usage.input_tokens_details.
# No global temperature override: the plugin sends temperature 0 on every decision
# request, and chat clients keep the model's own default.
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
    --enable-auto-tool-choice --tool-call-parser granite4 \
    "$@"
