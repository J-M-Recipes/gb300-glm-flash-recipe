#!/bin/bash
# greedy_eager.sh — the greedy gate at -O0: V1-eager captures reference, sc9-eager compares. Also C1 on sc9-eager (cache ON: BYPASS default 16).
# NOTE: tf-sc9e-keep was launched with the default BYPASS (16) => cache active. tf-v1e-keep is stock V1 at -O0.
C=/home/milo/big-v1-campaign; cd $C
export API_KEY="$(cat /home/milo/.glm_api_key)" BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-big
wait_ready() { local T; T=$(docker inspect -f "{{.State.StartedAt}}" "$1"); for i in $(seq 1 120); do docker logs --since "$T" "$1" 2>&1 | grep -q "Application startup complete" && return 0; docker inspect -f "{{.State.Status}}" "$1" | grep -q exited && { echo "$1 DEAD"; docker logs --since "$T" "$1" 2>&1 | grep -E "Error" | tail -3 | cut -c1-200; return 1; }; sleep 30; done; echo "$1 TIMEOUT"; return 1; }

docker stop -t 20 glm53-big-v1-keep >/dev/null; echo "v1 stopped $(date +%H:%M)"; docker stop -t 20 glm53-big-tf-sc9e-keep >/dev/null 2>&1; docker stop -t 20 glm53-big-tf-v1e-keep >/dev/null 2>&1
docker start glm53-big-tf-v1e-keep >/dev/null
wait_ready glm53-big-tf-v1e-keep || exit 1
echo "V1E_READY $(date +%H:%M)"
mkdir -p runs/v1-eager runs/sc9-eager
python3 greedy_equiv.py runs/v1-eager/greedy.json 2>&1 | tail -1
python3 greedy_equiv.py --compare runs/v1-eager/greedy.json runs/v1/greedy.json | head -1 | sed "s/^/V1EAGER_vs_V1COMPILED /"
bash bench3.sh v1-eager 2>&1 | grep -E "^C1 " | sed "s/^/V1EAGER /"
docker stop -t 20 glm53-big-tf-v1e-keep >/dev/null

docker start glm53-big-tf-sc9e-keep >/dev/null
wait_ready glm53-big-tf-sc9e-keep || exit 1
echo "SC9E_READY $(date +%H:%M)"
docker logs glm53-big-tf-sc9e-keep 2>&1 | grep -E "installed: S=|bypass_above" | tail -1 | cut -c1-120
python3 greedy_equiv.py runs/sc9-eager/greedy.json 2>&1 | tail -1
python3 greedy_equiv.py --compare runs/sc9-eager/greedy.json runs/v1-eager/greedy.json | head -1 | sed "s/^/SC9EAGER_vs_V1EAGER /"
python3 greedy_equiv.py --compare runs/sc9-eager/greedy.json runs/v1/greedy.json | head -1 | sed "s/^/SC9EAGER_vs_V1COMPILED /"
bash bench3.sh sc9-eager 2>&1 | grep -E "^C1 " | sed "s/^/SC9EAGER /"
docker stop -t 20 glm53-big-tf-sc9e-keep >/dev/null

docker start glm53-big-v1-keep >/dev/null && echo "V1_RESTORE_STARTED $(date +%H:%M)"
