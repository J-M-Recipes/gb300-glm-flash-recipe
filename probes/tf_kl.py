#!/usr/bin/env python3
"""tf_kl.py — teacher-forced divergence harness (non-inferiority gate).

  python3 tf_kl.py build                 -> builds runs/kl/corpus.json (token-id pieces) via the live server's /tokenize
  python3 tf_kl.py run <label>           -> teacher-forces every piece through the live server (prompt_logprobs=20, max_tokens=1),
                                            saves runs/kl/<label>.json: per position forced-token logprob, top-1 id, top-20 dict
  python3 tf_kl.py compare <A> <B>       -> divergence of A vs B (B = reference):
                                            mean|dlp|, p99, p99.9, max|dlp| of the forced-token logprob;
                                            top1 disagreement rate; mean truncated-KL(B||A) over the top-20 union (renormalised).
No sampling anywhere; every number is deterministic given the server. Requires API_KEY/BASE_URL env (like greedy_equiv.py).
"""
import os, sys, json, glob, math, time, urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:30001/v1").rstrip("/")
ROOT = BASE[:-3] if BASE.endswith("/v1") else BASE
KEY = os.environ.get("API_KEY", "")
MODEL = os.environ.get("MODEL", "glm-5.3-big")
OUT = "runs/kl"; os.makedirs(OUT, exist_ok=True)
PIECE = int(os.environ.get("KL_PIECE", "1024")); MIN_PIECE = 256
TOPK = 20

def post(path, body, root=False, timeout=600):
    req = urllib.request.Request((ROOT if root else BASE) + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)

def tokenize(text):
    return post("/tokenize", {"model": MODEL, "prompt": text}, root=True)["tokens"]

def sources():
    """(category, path, text) — prose (own research docs + licenses), code (stdlib), model-like (greedy outputs)."""
    out = []
    for p in sorted(glob.glob("kl_corpus_src/*.md"))[:12]:
        out.append(("prose", p, open(p, errors="ignore").read()))
    for p in ["/usr/share/common-licenses/GPL-3", "/usr/share/common-licenses/Apache-2.0", "/usr/share/common-licenses/GPL-2"]:
        if os.path.exists(p): out.append(("legal", p, open(p, errors="ignore").read()))
    std = sorted(glob.glob("/usr/lib/python3*/[a-z]*.py"))
    for p in [s for s in std if os.path.getsize(s) > 20000][:10]:
        out.append(("code", p, open(p, errors="ignore").read()))
    for p in ["runs/v1/greedy.json", "runs/v1-eager/greedy.json"]:
        if os.path.exists(p):
            try:
                d = json.load(open(p)); txt = "\n\n".join((x.get("prompt", "") + "\n" + x.get("content", x.get("text", ""))) if isinstance(x, dict) else str(x) for x in (d if isinstance(d, list) else d.get("outputs", d.get("results", []))))
                if len(txt) > 2000: out.append(("modeltext", p, txt))
            except Exception as e: print("skip", p, e, file=sys.stderr)
    return out

def build():
    pieces = []; per_cat = {}
    for cat, path, text in sources():
        ids = tokenize(text)
        for i in range(0, len(ids) - MIN_PIECE, PIECE):
            chunk = ids[i:i + PIECE]
            if len(chunk) < MIN_PIECE: break
            pieces.append({"cat": cat, "src": os.path.basename(path), "off": i, "ids": chunk})
            per_cat[cat] = per_cat.get(cat, 0) + 1
            if per_cat[cat] >= {"prose": 14, "legal": 8, "code": 14, "modeltext": 8}.get(cat, 8): break
    json.dump({"piece": PIECE, "pieces": pieces}, open(f"{OUT}/corpus.json", "w"))
    print("CORPUS", json.dumps({"pieces": len(pieces), "positions": sum(len(p["ids"]) - 1 for p in pieces), "per_cat": per_cat}))

def run(label):
    corpus = json.load(open(f"{OUT}/corpus.json"))["pieces"]
    res = []; t0 = time.time()
    for k, p in enumerate(corpus):
        r = post("/completions", {"model": MODEL, "prompt": p["ids"], "max_tokens": 1, "temperature": 0, "prompt_logprobs": TOPK, "logprobs": 0})
        plp = r["choices"][0]["prompt_logprobs"]
        rows = []
        for i in range(1, len(p["ids"])):
            d = plp[i] or {}
            tok = str(p["ids"][i]); ent = d.get(tok)
            top1 = min(d.items(), key=lambda kv: kv[1]["rank"])[0] if d else None
            rows.append([ent["logprob"] if ent else None, int(top1) if top1 is not None else None, {int(t): v["logprob"] for t, v in d.items()}])
        res.append({"cat": p["cat"], "src": p["src"], "off": p["off"], "rows": rows})
        print(f"  {k+1}/{len(corpus)} {p['cat']}:{p['src']} {len(rows)} pos  {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    json.dump(res, open(f"{OUT}/{label}.json", "w"))
    print("RUN", label, json.dumps({"pieces": len(res), "positions": sum(len(x["rows"]) for x in res), "sec": round(time.time() - t0)}))

def _tkl(pb, pa):
    """truncated KL(B||A): union of keys, renormalise each over the union (missing -> tiny)."""
    keys = set(pb) | set(pa); floor = -30.0
    lb = {t: pb.get(t, floor) for t in keys}; la = {t: pa.get(t, floor) for t in keys}
    zb = math.log(sum(math.exp(v) for v in lb.values())); za = math.log(sum(math.exp(v) for v in la.values()))
    return sum(math.exp(lb[t] - zb) * ((lb[t] - zb) - (la[t] - za)) for t in keys)

def compare(a, b):
    A = json.load(open(f"{OUT}/{a}.json")); B = json.load(open(f"{OUT}/{b}.json"))
    assert len(A) == len(B), "corpus mismatch"
    d = []; top1_dis = 0; n = 0; kl = []; per_cat = {}
    for pa, pb in zip(A, B):
        assert pa["src"] == pb["src"] and pa["off"] == pb["off"]
        c = per_cat.setdefault(pa["cat"], {"n": 0, "sum_abs": 0.0, "top1_dis": 0, "max": 0.0})
        for ra, rb in zip(pa["rows"], pb["rows"]):
            if ra[0] is None or rb[0] is None: continue
            x = abs(ra[0] - rb[0]); d.append(x); n += 1
            c["n"] += 1; c["sum_abs"] += x; c["max"] = max(c["max"], x)
            if ra[1] != rb[1]: top1_dis += 1; c["top1_dis"] += 1
            kl.append(_tkl(rb[2], ra[2]))
    d.sort(); kl.sort()
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else None
    out = {"A": a, "B_ref": b, "positions": n, "mean_abs_dlp": sum(d) / n, "p99_abs_dlp": q(d, 0.99), "p999_abs_dlp": q(d, 0.999), "max_abs_dlp": d[-1],
           "top1_disagree_rate": top1_dis / n, "top1_disagree_n": top1_dis, "mean_tkl20": sum(kl) / n, "p999_tkl20": q(kl, 0.999), "max_tkl20": kl[-1],
           "per_cat": {k: {"n": v["n"], "mean_abs_dlp": v["sum_abs"] / max(1, v["n"]), "top1_disagree_rate": v["top1_dis"] / max(1, v["n"]), "max_abs_dlp": v["max"]} for k, v in per_cat.items()}}
    print("COMPARE", json.dumps(out))
    return out

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build": build()
    elif cmd == "run": run(sys.argv[2])
    elif cmd == "compare": compare(sys.argv[2], sys.argv[3])
    else: raise SystemExit(__doc__)
