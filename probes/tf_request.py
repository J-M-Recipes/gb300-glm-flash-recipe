#!/usr/bin/env python3
"""tf_request.py — teacher-forced single-prefill request (no transformers needed; uses the server's /tokenize).
prompt 5 with the greedy harness's exact chat template + first N tokens of V1's greedy output (reasoning first, as the
model emits it). ONE /completions call with token ids, max_tokens=1, temperature 0, logprobs=5. Usage: tf_request.py <label> [N=64]"""
import os, sys, json, hashlib, urllib.request

label = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
BASE = os.environ.get("BASE_URL", "http://127.0.0.1:30001/v1"); KEY = os.environ.get("API_KEY", "x"); MODEL = os.environ.get("MODEL", "glm-5.3-big")
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
ROOT = BASE[: -len("/v1")] if BASE.endswith("/v1") else BASE
def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=H)
    with urllib.request.urlopen(req, timeout=900) as r: return json.load(r)

src = open("/home/milo/big-v1-campaign/greedy_equiv.py").read()
ns = {}; exec(compile(src[src.index("PROMPTS = ["): src.index("]\n", src.index("PROMPTS = [")) + 2], "p", "exec"), ns)
q = ns["PROMPTS"][5]
v1 = json.load(open("/home/milo/big-v1-campaign/runs/v1/greedy.json"))["5"]
reasoning, content = v1.split("\u241f", 1)
cont_text = reasoning if reasoning.strip() else content   # prompt 5: empty reasoning, so the continuation is the content

# 1) chat-template the prompt exactly as the harness does (server-side template, same kwargs)
t = post(f"{ROOT}/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": q}], "add_generation_prompt": True,
                              "chat_template_kwargs": {"reasoning_effort": "low"}})
prompt_ids = t["tokens"]
# 2) continuation = the model's own greedy reasoning text (what it emitted first), truncated to N tokens
t2 = post(f"{ROOT}/tokenize", {"model": MODEL, "prompt": "</think>" + cont_text, "add_special_tokens": False})
comp_ids = t2["tokens"][:N]
ids = prompt_ids + comp_ids
sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
print(f"{label}: prompt_ids={len(prompt_ids)} comp_ids={len(comp_ids)} total={len(ids)} sha={sha}")

j = post(f"{BASE}/completions", {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0, "logprobs": 5, "skip_special_tokens": False})
ch = j["choices"][0]; lp = ch.get("logprobs") or {}
top = (lp.get("top_logprobs") or [None])[0]
out = {"label": label, "n_ids": len(ids), "ids_sha": sha, "ids": ids, "next_text": ch["text"], "top5": top, "usage": j.get("usage")}
os.makedirs("/home/milo/big-v1-campaign/trace", exist_ok=True)
json.dump(out, open(f"/home/milo/big-v1-campaign/trace/tf_{label}.json", "w"), indent=1)
print("TF", json.dumps({"label": label, "n_ids": len(ids), "sha": sha, "next": ch["text"]}), "| top5:", json.dumps(top))
