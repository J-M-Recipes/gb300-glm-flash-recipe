#!/bin/bash
# tf_run.sh <container-suffix> <label>  — wait for ready, confirm trace hooks installed, send the teacher-forced request, dump, verify.
RUN="$1"; LABEL="$2"; C=/home/milo/big-v1-campaign; NAME="glm53-big-$RUN"
cd $C
for i in $(seq 1 120); do
  docker logs "$NAME" 2>&1 | grep -q "Application startup complete" && break
  docker inspect -f "{{.State.Status}}" "$NAME" | grep -q exited && { echo "${LABEL}_DEAD"; docker logs "$NAME" 2>&1 | grep -E "Error|LAYER_TRACE" | tail -4 | cut -c1-200; exit 1; }
  sleep 30
done
echo "${LABEL}_READY $(date +%H:%M)"
docker logs "$NAME" 2>&1 | grep -E "LAYER_TRACE (armed|installed|hook FAILED|install failed)" | head -3 | cut -c1-140
export API_KEY="$(cat /home/milo/.glm_api_key)" BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-big
python3 tf_request.py "$LABEL" 64 2>&1 | tail -1 | cut -c1-400
sleep 2
touch "trace/$RUN/DUMP_TRACE"
for i in $(seq 1 30); do [ -f "trace/$RUN/layer_trace.pt" ] && [ ! -f "trace/$RUN/DUMP_TRACE" ] && break; sleep 2; done
docker logs "$NAME" 2>&1 | grep -E "LAYER_TRACE dumped|dump failed" | tail -1 | cut -c1-140
ls -la "trace/$RUN/layer_trace.pt" | cut -c1-90
