#!/bin/bash
# tf_run_sc.sh <container-suffix> <label> — slot-cache variant: /wcap is bind-mounted to $C/capture, so sentinel+dump live there.
RUN="$1"; LABEL="$2"; C=/home/milo/big-v1-campaign; NAME="glm53-big-$RUN"
cd $C
rm -f capture/DUMP_TRACE capture/layer_trace.pt
for i in $(seq 1 120); do
  docker logs "$NAME" 2>&1 | grep -q "Application startup complete" && break
  docker inspect -f "{{.State.Status}}" "$NAME" | grep -q exited && { echo "${LABEL}_DEAD"; docker logs "$NAME" 2>&1 | grep -E "Error|LAYER_TRACE" | tail -4 | cut -c1-200; exit 1; }
  sleep 30
done
echo "${LABEL}_READY $(date +%H:%M)"
docker logs "$NAME" 2>&1 | grep -E "LAYER_TRACE (armed|installed|hook FAILED|install failed)|runner seam patched|forcing TrtLlm" | head -4 | cut -c1-140
export API_KEY="$(cat /home/milo/.glm_api_key)" BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-big
python3 tf_request.py "$LABEL" 64 2>&1 | tail -1 | cut -c1-400
sleep 2
touch capture/DUMP_TRACE
for i in $(seq 1 30); do [ -f capture/layer_trace.pt ] && [ ! -f capture/DUMP_TRACE ] && break; sleep 2; done
docker logs "$NAME" 2>&1 | grep -E "LAYER_TRACE dumped|dump failed" | tail -1 | cut -c1-140
mkdir -p "trace/$RUN"; mv capture/layer_trace.pt "trace/$RUN/layer_trace.pt" && ls -la "trace/$RUN/layer_trace.pt" | cut -c1-90
echo ===COMPARE
docker run --rm --entrypoint python3 -v $C:/w vllm-glm53-uva:v0.28.0-2cf0a691 /w/tf_compare.py /w/trace/tf-v1/layer_trace.pt "/w/trace/$RUN/layer_trace.pt" /w/trace/tf_v1.json "/w/trace/tf_${LABEL}.json" 2>&1 | grep -vE "^\s*$|Warning" | cut -c1-260
