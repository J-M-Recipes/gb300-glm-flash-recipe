#!/bin/bash
# launch-slotcache.sh <run> <slots> [extra vllm args...]
# Slot-cache experiment launcher: ALL routed experts offloaded to pinned host (cpu-offload-gb large), and the
# SLOT_CACHE hook allocates S HBM slots per MoE layer. KV/seqs as V1.
set -e
RUN="$1"; SLOTS="$2"; shift 2
C=/home/milo/big-v1-campaign
mkdir -p "$C/runs/$RUN"
printf 'SLOTS=%s\n%s\n' "$SLOTS" "$*" > "$C/runs/$RUN/extra_args.txt"; date -Is > "$C/runs/$RUN/launched_at.txt"
LAYER_TRACE_ENV=""
if [ "${LAYER_TRACE:-0}" = "1" ]; then LAYER_TRACE_ENV="-e LAYER_TRACE=1 -e LAYER_TRACE_MINM=${LAYER_TRACE_MINM:-8} -e LAYER_TRACE_FULL=${LAYER_TRACE_FULL:-5}"; fi
docker run -d --name "glm53-big-$RUN" --gpus all --shm-size 32g --network host \
  -v /home/exx/models/GLM-5.3-NVFP4-big:/model:ro \
  -v /home/milo/vllm-cache:/root/.cache/vllm \
  -v "$C/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro" \
  -v "$C:/w:ro" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e "VLLM_AUTOTUNE_CACHE_KEY=slotcache-S$SLOTS" \
  ${LAYER_TRACE_ENV} -e EXACT_PIN=1 -e "SLOT_CACHE=$SLOTS" -e SLOT_CACHE_HOOK=/w/slot_cache_hook.py -e SLOT_CACHE_CAPTURE=${CAPTURE:-0} -e SLOT_CACHE_ROUTER=${ROUTER:-vllm} -e SLOT_CACHE_UNPACKED=${UNPACKED:-0} -v /home/milo/big-v1-campaign/capture:/wcap -e SLOT_CACHE_LOGIT_RING=${LOGIT_RING:-0} -e SLOT_CACHE_STATS_SEC=${STATS_SEC:-20} -e SLOT_CACHE_BYPASS_TOKENS=${BYPASS:-16} \
  -e VLLM_API_KEY="$(cat /home/milo/.glm_api_key)" \
  vllm-glm53-uva:v0.28.0-2cf0a691 \
  /model --host 0.0.0.0 --port 30001 \
  --served-model-name glm-5.3-big \
  --trust-remote-code \
  --quantization modelopt \
  --load-format safetensors \
  --offload-backend uva \
  --cpu-offload-gb 420 \
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype bfloat16 \
  --kv-cache-memory 8589934592 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  "$@"
echo "launched glm53-big-$RUN  SLOT_CACHE=$SLOTS  offload=420 (all experts)  extra: $*"
