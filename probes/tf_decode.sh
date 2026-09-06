#!/bin/bash
# tf_decode.sh — eager decode trace at M=1 on prompt 12 (3 tokens), sc9 then V1. Fresh launches with LAYER_TRACE_MINM=1.
set -o pipefail
C=/home/milo/big-v1-campaign; cd $C
export API_KEY="$(cat /home/milo/.glm_api_key)" BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-big
wait_ready() { local T; T=$(docker inspect -f "{{.State.StartedAt}}" "$1"); for i in $(seq 1 120); do docker logs --since "$T" "$1" 2>&1 | grep -q "Application startup complete" && return 0; docker inspect -f "{{.State.Status}}" "$1" | grep -q exited && { echo "$1 DEAD"; docker logs --since "$T" "$1" 2>&1 | grep -E "Error" | tail -3 | cut -c1-200; return 1; }; sleep 30; done; echo "$1 TIMEOUT"; return 1; }
gen12() {  # <label> <capdir>
python3 - "$1" "$2" <<'EOF'
import json, os, re, sys, time, urllib.request
label, cap = sys.argv[1], sys.argv[2]
B=os.environ["BASE_URL"]; ROOT=B.rsplit("/v1",1)[0]; H={"Authorization":"Bearer "+os.environ["API_KEY"],"Content-Type":"application/json"}
src=open("greedy_equiv.py").read(); P=eval(re.search(r"PROMPTS\s*=\s*(\[.*?\n\])", src, re.S).group(1))
def post(base, path, body):
    r=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers=H); return json.load(urllib.request.urlopen(r,timeout=900))
ids=post(ROOT,"/tokenize",{"model":"glm-5.3-big","messages":[{"role":"user","content":P[12]}],"add_generation_prompt":True,"chat_template_kwargs":{"reasoning_effort":"low"}})["tokens"]
j=post(B,"/completions",{"model":"glm-5.3-big","prompt":ids,"max_tokens":3,"temperature":0,"logprobs":5})
lp=j["choices"][0]["logprobs"]
print("GEN12", label, "n_prompt", len(ids), "toks", lp["tokens"], "lps", [round(x,6) for x in lp["token_logprobs"]])
open(os.path.join(cap,"DUMP_TRACE"),"w").write("1")
for _ in range(120):
    if os.path.exists(os.path.join(cap,"layer_trace.pt")) and not os.path.exists(os.path.join(cap,"DUMP_TRACE")): break
    time.sleep(1)
json.dump({"ids":ids,"toks":lp["tokens"],"lps":lp["token_logprobs"]}, open(f"trace/tf12_{label}.json","w"))
EOF
}
docker stop -t 20 glm53-big-v1-keep >/dev/null 2>&1; echo "v1 stopped $(date +%H:%M)"
mkdir -p trace/d12-sc9 trace/d12-v1
LAYER_TRACE=1 LAYER_TRACE_MINM=1 LAYER_TRACE_MAXM=64 LAYER_TRACE_FULL=78 ROUTER=trt bash launch-slotcache.sh d12-sc9 112 -O0 | tail -1
wait_ready glm53-big-d12-sc9 || exit 1; echo "D12_SC9_READY $(date +%H:%M)"
docker logs glm53-big-d12-sc9 2>&1 | grep -E "LAYER_TRACE installed" | tail -1 | cut -c1-120
gen12 sc9 capture
docker logs glm53-big-d12-sc9 2>&1 | grep -E "LAYER_TRACE dumped" | tail -1 | cut -c1-120
mv capture/layer_trace.pt trace/d12-sc9/layer_trace.pt; ls -la trace/d12-sc9/layer_trace.pt | cut -c25-80
docker stop -t 20 glm53-big-d12-sc9 >/dev/null; docker rename glm53-big-d12-sc9 glm53-big-d12-sc9-keep
LAYER_TRACE=1 LAYER_TRACE_MINM=1 LAYER_TRACE_MAXM=64 LAYER_TRACE_FULL=78 bash launch-bigv1.sh d12-v1 --cpu-offload-gb 188 --max-num-seqs 4 --kv-cache-memory 8589934592 -O0 | tail -1
wait_ready glm53-big-d12-v1 || exit 1; echo "D12_V1_READY $(date +%H:%M)"
gen12 v1 trace/d12-v1
docker logs glm53-big-d12-v1 2>&1 | grep -E "LAYER_TRACE dumped" | tail -1 | cut -c1-120
ls -la trace/d12-v1/layer_trace.pt | cut -c25-80
docker stop -t 20 glm53-big-d12-v1 >/dev/null; docker rename glm53-big-d12-v1 glm53-big-d12-v1-keep
docker start glm53-big-v1-keep >/dev/null && echo "V1_RESTORE_STARTED $(date +%H:%M)"
echo "===DECODE COMPARE (per-call: prefill M=23, then decode M=1 x2)"
docker run --rm --entrypoint python3 -v $C:/w vllm-glm53-uva:v0.28.0-2cf0a691 /w/tf_compare.py /w/trace/d12-v1/layer_trace.pt /w/trace/d12-sc9/layer_trace.pt 2>&1 | grep -vE "^\s*$|Warning" | head -40 | cut -c1-200
