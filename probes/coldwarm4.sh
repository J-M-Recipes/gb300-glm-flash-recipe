#!/bin/bash
# coldwarm4.sh — probe smoke on V1 (must print PROBE lines or we stop), then sc9e, then v1e, then restore.
set -o pipefail
C=/home/milo/big-v1-campaign; cd $C
export API_KEY="$(cat /home/milo/.glm_api_key)" BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-big
wait_ready() { local T; T=$(docker inspect -f "{{.State.StartedAt}}" "$1"); for i in $(seq 1 120); do docker logs --since "$T" "$1" 2>&1 | grep -q "Application startup complete" && return 0; docker inspect -f "{{.State.Status}}" "$1" | grep -q exited && { echo "$1 DEAD"; return 1; }; sleep 30; done; echo "$1 TIMEOUT"; return 1; }
run_probe() { python3 coldwarm_probe.py "$1" > "runs/coldwarm_$1.log" 2>&1; grep "^PROBE" "runs/coldwarm_$1.log" | cut -c1-230; grep -q "^PROBE $1 12_warm_again" "runs/coldwarm_$1.log" || { echo "PROBE_FAILED $1"; tail -3 "runs/coldwarm_$1.log"; return 1; }; }
wait_ready glm53-big-v1-keep || exit 1; echo "V1_READY $(date +%H:%M) (smoke)"
run_probe v1c || { echo "STOP: probe broken; V1 left serving"; exit 1; }
docker stop -t 20 glm53-big-v1-keep >/dev/null; echo "v1 stopped $(date +%H:%M)"
docker start glm53-big-tf-sc9e-keep >/dev/null; wait_ready glm53-big-tf-sc9e-keep || exit 1; echo "SC9E_READY $(date +%H:%M)"
run_probe sc9e
docker stop -t 20 glm53-big-tf-sc9e-keep >/dev/null
docker start glm53-big-tf-v1e-keep >/dev/null; wait_ready glm53-big-tf-v1e-keep || exit 1; echo "V1E_READY $(date +%H:%M)"
run_probe v1e
docker stop -t 20 glm53-big-tf-v1e-keep >/dev/null
docker start glm53-big-v1-keep >/dev/null && echo "V1_RESTORE_STARTED $(date +%H:%M)"
