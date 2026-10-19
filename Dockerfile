# A vLLM OpenAI-compatible server for a composed granite-switch checkpoint, serving
# BOTH the standard /v1/chat/completions API and typesafe.ai's jev system-one API
# (/v1/systemone) on the same server -- one engine, one GPU KV cache, one container.
# The checkpoint's weights are NOT baked into this image (mounted separately at
# $MODEL_DIR / /repository, e.g. by HF Inference Endpoints' own deployment mechanism).
#
#   docker build --platform linux/amd64 -t <your-registry>/granite-switch-systemone:<tag> .
#
# Optional ASR support (build granite-switch[audio] in, and bake in a specific speech
# model's weights at a pinned revision -- see ASR_MODEL/ASR_REVISION below):
#   docker build --platform linux/amd64 --build-arg WITH_ASR=1 \
#       -t <your-registry>/granite-switch-systemone:<tag>-asr .
FROM vllm/vllm-openai:v0.26.0-cu129
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG GRANITE_SWITCH_VERSION=0.1.1
ARG WITH_ASR=0

# Pinned to a specific, released granite-switch version (not `main`) so this image's
# behavior does not drift when upstream changes -- deliberately a tagged PyPI release,
# not a floating branch. transformers 5.17 is what granite-switch supports (the base
# image has an older pin); [audio] is only installed when WITH_ASR=1 (it brings librosa
# and soundfile, used only by the ASR cascade). The base image's vLLM, torch, and
# torchcodec (a CUDA 12.9 build) are kept as they are.
RUN python3 -m pip list --format=freeze | grep -i -E "^(vllm|torch|torchaudio|torchcodec|torchvision|flashinfer-python)==" > /tmp/keep.txt \
    && if [ "$WITH_ASR" = "1" ]; then EXTRA="granite-switch[audio]"; else EXTRA="granite-switch[vllm]"; fi \
    && uv pip install --system --constraint /tmp/keep.txt \
        "${EXTRA}==${GRANITE_SWITCH_VERSION}" \
        "transformers==5.17.0" \
    && python3 -c "import granite_switch.vllm, vllm, transformers, torch; print(vllm.__version__, transformers.__version__, torch.__version__)"

# The ASR cascade's speech model, named by Hugging Face id in the checkpoint's config,
# baked in as a Hub cache at the revision this image was tested with, without the
# repo's unused .py files. The server runs with the Hub offline, so a missing/moved file
# fails startup instead of silently fetching something else.
ARG ASR_MODEL=ibm-granite/granite-speech-5.0-470m-turboctc
ARG ASR_REVISION=947f59af40db9791170a0628cf0f3f4812d720f1
RUN if [ "$WITH_ASR" = "1" ]; then \
        hf download ${ASR_MODEL} --cache-dir /opt/hf-cache --exclude "*.py" > /dev/null \
        && test "$(cat /opt/hf-cache/models--${ASR_MODEL//\//--}/refs/main)" = "${ASR_REVISION}"; \
    fi
ENV HF_HUB_CACHE=/opt/hf-cache HF_HUB_OFFLINE=1

# The jev system-one API (/v1/systemone), a vllm.endpoint_plugins entry point mounted
# on this same server -- see README.md and src/granite_switch_systemone/plugin.py.
COPY pyproject.toml README.md /opt/granite-switch-systemone/
COPY src /opt/granite-switch-systemone/src
RUN uv pip install --system --no-deps /opt/granite-switch-systemone

COPY serve.sh /usr/local/bin/serve.sh
RUN chmod +x /usr/local/bin/serve.sh
# Endpoint plugins only load if VLLM_PLUGINS is set AND names them (stricter than
# other vLLM plugin groups, which default to "load everything" when VLLM_PLUGINS is
# unset): must include register_granite_switch too, or the Switch model class stops
# being registered the moment this allowlist is set at all.
ENV VLLM_PLUGINS=register_granite_switch,granite_switch_systemone \
    SYSTEMONE_ADAPTER_NAME=decision \
    SYSTEMONE_INVOCATION_MARKER="### DECISION"
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/serve.sh"]
