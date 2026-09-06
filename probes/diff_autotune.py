import json, re
base = "/home/milo/vllm-cache/flashinfer_autotune_cache/0.6.16.post3/103a"
def load(n):
    d = json.load(open(f"{base}/{n}/autotune_configs.json")); d.pop("_metadata", None); return d
def trt(d):
    out = {}
    for k, v in d.items():
        if "trtllm_fp4_block_scale_moe" not in k: continue
        m = re.search(r"\(\((\d+), 6144\)", k); M = int(m.group(1)) if m else None
        out[M] = v
    return out
a, b = load("bigv1-shared"), load("slotcache-S112")
ta, tb = trt(a), trt(b)
print("bigv1-shared: total", len(a), "trtllm entries", len(ta), "| ops:", sorted({k.split("'")[1] for k in a})[:6])
print("slotcache-S112: total", len(b), "trtllm entries", len(tb), "| ops:", sorted({k.split("'")[1] for k in b})[:6])
Ms = sorted(set(ta) | set(tb), key=lambda x: (x is None, x))
print("M      bigv1(Monolithic)   slotcache(Modular)   same?")
for M in Ms:
    va, vb = ta.get(M), tb.get(M)
    print(f"{str(M):6} {json.dumps(va) if va else '-':20} {json.dumps(vb) if vb else '-':20} {'SAME' if va == vb else ('DIFF' if va and vb else '')}")
# Also check the hashed dirs for trtllm entries in case V1 used one of those
import os
for n in os.listdir(base):
    if n in ("bigv1-shared", "slotcache-S112", "slotcache-S104"): continue
    try:
        d = load(n); t = trt(d)
        if t: print("hashed", n[:12], "trtllm entries", len(t), "M=1:", t.get(1), "M=1024:", t.get(1024))
    except Exception as e: print("hashed", n[:12], "err", e)
