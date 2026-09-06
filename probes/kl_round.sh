#!/bin/bash
# kl_round.sh — non-inferiority harness, first full round. Runs AFTER ffi_eager.sh has restored V1.
#  1. V1 (compiled, production) is up: build corpus, smoke run on 2 pieces, then full run -> kl/v1c.json
#  2. V1-eager (tf-v1e-keep, -O0): full run -> kl/v1e.json           => floor = compare(v1c, v1e)
#  3. sc11-ffi-keep (-O0): full run -> kl/sc11f.json                  => candidate = compare(sc11f, v1e)
#  4. sc9e-keep (-O0, old TRT router): full run -> kl/sc9e.json       => how big was the rounding defect in KL terms
#  5. restore V1.
set -o pipefail
C=/home/milo/big-v1-campaign; cd $C
export API_KEY="$(cat /home/milo/.glm_api_key)" BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-big
wait_ready() { local T; T=$(docker inspect -f "{{.State.StartedAt}}" "$1"); for i in $(seq 1 120); do docker logs --since "$T" "$1" 2>&1 | grep -q "Application startup complete" && return 0; docker inspect -f "{{.State.Status}}" "$1" | grep -q exited && { echo "$1 DEAD"; return 1; }; sleep 30; done; echo "$1 TIMEOUT"; return 1; }
wait_ready glm53-big-v1-keep || exit 1; echo "V1_UP $(date +%H:%M)"
python3 tf_kl.py build | tail -1 || exit 1
# smoke: first 2 pieces only (temporary corpus)
python3 - <<'EOF'
import json; d=json.load(open("runs/kl/corpus.json")); json.dump({"piece":d["piece"],"pieces":d["pieces"][:2]}, open("runs/kl/corpus_smoke.json","w")); json.dump(d, open("runs/kl/corpus_full.json","w"))
EOF
cp runs/kl/corpus_smoke.json runs/kl/corpus.json
python3 tf_kl.py run v1c_smoke 2>/dev/null | tail -1 || { echo SMOKE_FAIL; cp runs/kl/corpus_full.json runs/kl/corpus.json; exit 1; }
python3 tf_kl.py run v1c_smoke2 2>/dev/null | tail -1
python3 tf_kl.py compare v1c_smoke2 v1c_smoke | tail -1 | sed "s/^/SELF_REPEAT_SMOKE /"
cp runs/kl/corpus_full.json runs/kl/corpus.json
python3 tf_kl.py run v1c 2>/dev/null | tail -1
docker stop -t 20 glm53-big-v1-keep >/dev/null; echo "v1 stopped $(date +%H:%M)"
docker start glm53-big-tf-v1e-keep >/dev/null; wait_ready glm53-big-tf-v1e-keep || { docker start glm53-big-v1-keep; exit 1; }; echo "V1E_UP $(date +%H:%M)"
python3 tf_kl.py run v1e 2>/dev/null | tail -1
python3 tf_kl.py compare v1c v1e | tail -1 | sed "s/^/FLOOR_v1c_vs_v1e /"
docker stop -t 20 glm53-big-tf-v1e-keep >/dev/null
docker start glm53-big-sc11-ffi-keep >/dev/null; wait_ready glm53-big-sc11-ffi-keep || { docker start glm53-big-v1-keep; exit 1; }; echo "SC11F_UP $(date +%H:%M)"
python3 tf_kl.py run sc11f 2>/dev/null | tail -1
python3 tf_kl.py compare sc11f v1e | tail -1 | sed "s/^/CAND_sc11f_vs_v1e /"
docker stop -t 20 glm53-big-sc11-ffi-keep >/dev/null
docker start glm53-big-tf-sc9e-keep >/dev/null; wait_ready glm53-big-tf-sc9e-keep || { docker start glm53-big-v1-keep; exit 1; }; echo "SC9E_UP $(date +%H:%M)"
python3 tf_kl.py run sc9e 2>/dev/null | tail -1
python3 tf_kl.py compare sc9e v1e | tail -1 | sed "s/^/OLD_sc9e_vs_v1e /"
docker stop -t 20 glm53-big-tf-sc9e-keep >/dev/null
docker start glm53-big-v1-keep >/dev/null && echo "V1_RESTORE_STARTED $(date +%H:%M)"
