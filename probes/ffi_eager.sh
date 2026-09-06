#!/bin/bash
# ffi_eager.sh — sc11: slot cache S=112 + SLOT_CACHE_ROUTER=ffi at -O0: probe (prompt 12 step-3 logprob), greedy vs V1-eager, restore V1.
set -o pipefail
C=/home/milo/big-v1-campaign; cd $C
export API_KEY="$(cat /home/milo/.glm_api_key)" BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-big
wait_ready() { local T; T=$(docker inspect -f "{{.State.StartedAt}}" "$1"); for i in $(seq 1 120); do docker logs --since "$T" "$1" 2>&1 | grep -q "Application startup complete" && return 0; docker inspect -f "{{.State.Status}}" "$1" | grep -q exited && { echo "$1 DEAD"; docker logs --since "$T" "$1" 2>&1 | grep -E "Error|Traceback" -A2 | tail -8 | cut -c1-220; return 1; }; sleep 30; done; echo "$1 TIMEOUT"; return 1; }
python3 -c "import ast;ast.parse(open('slot_cache_hook.py').read())" || { echo HOOK_SYNTAX_FAIL; exit 1; }
docker stop -t 20 glm53-big-v1-keep >/dev/null 2>&1; echo "v1 stopped $(date +%H:%M)"
ROUTER=ffi bash launch-slotcache.sh sc11-ffi 112 -O0 | tail -1
wait_ready glm53-big-sc11-ffi || { docker start glm53-big-v1-keep >/dev/null; echo "V1_RESTORE_STARTED $(date +%H:%M) (after failure)"; exit 1; }
echo "SC11_READY $(date +%H:%M)"
docker logs glm53-big-sc11-ffi 2>&1 | grep -E "installed: S=|router=|ffi|SLOT_CACHE.*rror" | head -4 | cut -c1-140
python3 coldwarm_probe.py sc11f > runs/coldwarm_sc11f.log 2>&1; grep "^PROBE sc11f 12_cold\|^PROBE sc11f 1_warm" runs/coldwarm_sc11f.log | cut -c1-200
echo "REF   v1e   12_cold lps=[-0.173395, -1e-06, -0.073458]   (sc9e was -0.067105; sc10u -0.063028)"
mkdir -p runs/sc11-ffi
python3 greedy_equiv.py runs/sc11-ffi/greedy.json 2>&1 | tail -1
python3 greedy_equiv.py --compare runs/sc11-ffi/greedy.json runs/v1-eager/greedy.json | head -1 | sed "s/^/SC11F_vs_V1EAGER /"
python3 greedy_equiv.py --compare runs/sc11-ffi/greedy.json runs/sc9-eager/greedy.json | head -1 | sed "s/^/SC11F_vs_SC9EAGER /"
python3 greedy_equiv.py --compare runs/sc11-ffi/greedy.json runs/v1/greedy.json | head -1 | sed "s/^/SC11F_vs_V1COMPILED /"
bash bench3.sh sc11-ffi 2>&1 | grep -E "^C1 " | sed "s/^/SC11F_EAGER /"
docker stop -t 20 glm53-big-sc11-ffi >/dev/null; docker rename glm53-big-sc11-ffi glm53-big-sc11-ffi-keep
docker start glm53-big-v1-keep >/dev/null && echo "V1_RESTORE_STARTED $(date +%H:%M)"
