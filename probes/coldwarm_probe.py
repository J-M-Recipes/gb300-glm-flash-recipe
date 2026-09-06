import json, os, re, sys, urllib.request
B=os.environ["BASE_URL"]; ROOT=B.rsplit("/v1",1)[0]; H={"Authorization":"Bearer "+os.environ["API_KEY"],"Content-Type":"application/json"}
src=open("greedy_equiv.py").read(); P=eval(re.search(r"PROMPTS\s*=\s*(\[.*?\n\])", src, re.S).group(1))
def post(path, body):
    base = ROOT if path == "/tokenize" else B
    r=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers=H); return json.load(urllib.request.urlopen(r,timeout=900))
def ids_for(i):
    return post("/tokenize", {"model":"glm-5.3-big","messages":[{"role":"user","content":P[i]}],"add_generation_prompt":True,"chat_template_kwargs":{"reasoning_effort":"low"}})["tokens"]
def gen(i, n):
    ids=ids_for(i)
    j=post("/completions", {"model":"glm-5.3-big","prompt":ids,"max_tokens":n,"temperature":0,"logprobs":5})
    c=j["choices"][0]; lp=c["logprobs"]
    return {"n_prompt":len(ids),"cached":(j["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
            "toks":lp["tokens"][:3],"lps":[round(x,6) for x in lp["token_logprobs"][:3]],"top0":{k:round(v,6) for k,v in lp["top_logprobs"][0].items()}}
label=sys.argv[1]; res={}
res["12_cold"]=gen(12,4)                 # first request after start
res["0_warm200"]=gen(0,200)              # warm the shared prefix and generate the harness's own prompt 0
for i in (12,13,14,15,1):
    res[f"{i}_warm"]=gen(i,4)
res["12_warm_again"]=gen(12,4)
json.dump(res,open(f"runs/coldwarm_{label}.json","w"),indent=1)
for k,v in res.items():
    print("PROBE", label, k, f"n_prompt={v['n_prompt']} cached={v['cached']} toks={v['toks']} lps={v['lps']}")
