#!/bin/bash
# eager_repeat.sh — is the eager build self-repeatable across greedy runs? Uses whichever eager keeper is up; else starts sc9e.
C=/home/milo/big-v1-campaign; cd $C
export API_KEY="$(cat /home/milo/.glm_api_key)" BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-big
wait_ready() { local T; T=$(docker inspect -f "{{.State.StartedAt}}" "$1"); for i in $(seq 1 120); do docker logs --since "$T" "$1" 2>&1 | grep -q "Application startup complete" && return 0; docker inspect -f "{{.State.Status}}" "$1" | grep -q exited && { echo "$1 DEAD"; return 1; }; sleep 30; done; echo "$1 TIMEOUT"; return 1; }
docker stop -t 20 glm53-big-v1-keep >/dev/null 2>&1; echo "v1 stopped $(date +%H:%M)"
docker start glm53-big-tf-sc9e-keep >/dev/null; wait_ready glm53-big-tf-sc9e-keep || exit 1
echo "SC9E_READY $(date +%H:%M)"
python3 greedy_equiv.py runs/sc9-eager/greedy2.json 2>&1 | tail -1
python3 greedy_equiv.py --compare runs/sc9-eager/greedy2.json runs/sc9-eager/greedy.json | head -1 | sed "s/^/SC9EAGER_SELF /"
# teacher-forced decode trace at M=1: same 90 ids, max_tokens=8, MINM=1 needs a relaunch — skip; instead record logprobs of the first 8 generated tokens
python3 - <<'EOF'
import json, os, urllib.request
B=os.environ["BASE_URL"]; H={"Authorization":"Bearer "+os.environ["API_KEY"],"Content-Type":"application/json"}
ids=json.load(open("trace/tf_sc9e.json"))["ids"]
r=urllib.request.Request(B+"/completions",data=json.dumps({"model":"glm-5.3-big","prompt":ids,"max_tokens":8,"temperature":0,"logprobs":1}).encode(),headers=H)
j=json.load(urllib.request.urlopen(r,timeout=600)); lp=j["choices"][0]["logprobs"]
print("SC9E_DECODE8", json.dumps({"tokens":lp["tokens"],"logprobs":[round(x,6) for x in lp["token_logprobs"]]}))
EOF
docker stop -t 20 glm53-big-tf-sc9e-keep >/dev/null
docker start glm53-big-tf-v1e-keep >/dev/null; wait_ready glm53-big-tf-v1e-keep || exit 1
echo "V1E_READY $(date +%H:%M)"
python3 greedy_equiv.py runs/v1-eager/greedy2.json 2>&1 | tail -1
python3 greedy_equiv.py --compare runs/v1-eager/greedy2.json runs/v1-eager/greedy.json | head -1 | sed "s/^/V1EAGER_SELF /"
python3 - <<'EOF'
import json, os, urllib.request
B=os.environ["BASE_URL"]; H={"Authorization":"Bearer "+os.environ["API_KEY"],"Content-Type":"application/json"}
ids=json.load(open("trace/tf_v1e.json"))["ids"]
r=urllib.request.Request(B+"/completions",data=json.dumps({"model":"glm-5.3-big","prompt":ids,"max_tokens":8,"temperature":0,"logprobs":1}).encode(),headers=H)
j=json.load(urllib.request.urlopen(r,timeout=600)); lp=j["choices"][0]["logprobs"]
print("V1E_DECODE8", json.dumps({"tokens":lp["tokens"],"logprobs":[round(x,6) for x in lp["token_logprobs"]]}))
EOF
docker stop -t 20 glm53-big-tf-v1e-keep >/dev/null
docker start glm53-big-v1-keep >/dev/null && echo "V1_RESTORE_STARTED $(date +%H:%M)"
