#!/bin/bash
# launch-bigv1.sh — campaign launcher for the V1 recipe search (2026-09-04 night).
# Usage: launch-bigv1.sh <runname> [extra vllm args...]
# Base = v5 exactly (offload 200, util 0.95, bf16 KV, 65k, seqs 8). Adds:
#   - sitecustomize mount: (a) pins FlashInfer autotune cache dir to $AT_KEY so all runs share/merge configs,
#                          (b) ROUTE_TRACE_DIR hook dumps per-step routed-expert ids when set.
#   - container name glm53-big-<runname> (matches hard-stop filter glm53-big*)
set -e
RUN="$1"; shift
C=/home/milo/big-v1-campaign
AT_KEY="${AT_KEY:-bigv1-shared}"
TRACE="${TRACE:-}"
mkdir -p "$C/runs/$RUN" "$C/trace/$RUN"
TRACE_ENV=()
if [ -n "$TRACE" ]; then TRACE_ENV=(-e "ROUTE_TRACE_DIR=/trace"); fi
# LAYER_TRACE=1: per-layer output fingerprints (layer_trace.py) for teacher-forced comparisons; dumps to $C/trace/$RUN via /wcap
if [ "${LAYER_TRACE:-0}" = "1" ]; then TRACE_ENV+=(-e LAYER_TRACE=1 -e "LAYER_TRACE_MINM=${LAYER_TRACE_MINM:-8}" -e "LAYER_TRACE_FULL=${LAYER_TRACE_FULL:-5}" -v "$C:/w:ro" -v "$C/trace/$RUN:/wcap"); fi
printf '%s\n' "$@" > "$C/runs/$RUN/extra_args.txt"
date -Is > "$C/runs/$RUN/launched_at.txt"
docker run -d --name "glm53-big-$RUN" --gpus all --shm-size 32g --network host \
  -v /home/exx/models/GLM-5.3-NVFP4-big:/model:ro \
  -v /home/milo/vllm-cache:/root/.cache/vllm \
  -v "$C/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro" \
  -v "$C/trace/$RUN:/trace" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e "VLLM_AUTOTUNE_CACHE_KEY=$AT_KEY" \
  "${TRACE_ENV[@]}" \
  -e VLLM_API_KEY="$(cat /home/milo/.glm_api_key)" \
  vllm-glm53-uva:v0.28.0-2cf0a691 \
  /model --host 0.0.0.0 --port 30001 \
  --served-model-name glm-5.3-big \
  --trust-remote-code \
  --quantization modelopt \
  --load-format safetensors \
  --offload-backend uva \
  --cpu-offload-gb 200 \
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  "$@"
echo "launched glm53-big-$RUN (AT_KEY=$AT_KEY TRACE=${TRACE:-off}) extra: $*"
