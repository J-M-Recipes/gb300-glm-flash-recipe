#!/bin/bash
# unpacked_eager.sh — sc9 + SLOT_CACHE_UNPACKED=1 at -O0: probe (prompt 12 step-3 logprob), greedy vs V1-eager, restore V1.
set -o pipefail
C=/home/milo/big-v1-campaign; cd $C
export API_KEY="$(cat /home/milo/.glm_api_key)" BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-big
wait_ready() { local T; T=$(docker inspect -f "{{.State.StartedAt}}" "$1"); for i in $(seq 1 120); do docker logs --since "$T" "$1" 2>&1 | grep -q "Application startup complete" && return 0; docker inspect -f "{{.State.Status}}" "$1" | grep -q exited && { echo "$1 DEAD"; docker logs --since "$T" "$1" 2>&1 | grep -E "Error" | tail -3 | cut -c1-200; return 1; }; sleep 30; done; echo "$1 TIMEOUT"; return 1; }
docker stop -t 20 glm53-big-v1-keep >/dev/null 2>&1; echo "v1 stopped $(date +%H:%M)"
UNPACKED=1 ROUTER=trt bash launch-slotcache.sh sc10-unpacked 112 -O0 | tail -1
wait_ready glm53-big-sc10-unpacked || exit 1; echo "SC10_READY $(date +%H:%M)"
docker logs glm53-big-sc10-unpacked 2>&1 | grep -E "unpacked|installed: S=|router=" | head -3 | cut -c1-120
python3 coldwarm_probe.py sc10u > runs/coldwarm_sc10u.log 2>&1; grep "^PROBE sc10u 12_cold\|^PROBE sc10u 1_warm" runs/coldwarm_sc10u.log | cut -c1-200
echo "REF   v1e   12_cold lps=[-0.173395, -1e-06, -0.073458]   (sc9e was -0.067105)"
mkdir -p runs/sc10-unpacked
python3 greedy_equiv.py runs/sc10-unpacked/greedy.json 2>&1 | tail -1
python3 greedy_equiv.py --compare runs/sc10-unpacked/greedy.json runs/v1-eager/greedy.json | head -1 | sed "s/^/SC10U_vs_V1EAGER /"
python3 greedy_equiv.py --compare runs/sc10-unpacked/greedy.json runs/sc9-eager/greedy.json | head -1 | sed "s/^/SC10U_vs_SC9EAGER /"
python3 greedy_equiv.py --compare runs/sc10-unpacked/greedy.json runs/v1/greedy.json | head -1 | sed "s/^/SC10U_vs_V1COMPILED /"
docker stop -t 20 glm53-big-sc10-unpacked >/dev/null; docker rename glm53-big-sc10-unpacked glm53-big-sc10-unpacked-keep
docker start glm53-big-v1-keep >/dev/null && echo "V1_RESTORE_STARTED $(date +%H:%M)"
