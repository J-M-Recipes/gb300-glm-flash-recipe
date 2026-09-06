import json, sys
def outs(p):
    d = json.load(open(p))
    if isinstance(d, dict):
        for k in ("outputs", "results", "items"):
            if k in d: d = d[k]; break
    if isinstance(d, dict): d = [d[k] for k in sorted(d, key=lambda x: (len(x), x))]
    return d
def txt(o):
    if isinstance(o, str): return o
    for k in ("text", "content", "output", "completion"):
        if k in o and isinstance(o[k], str): return o[k]
    return json.dumps(o, sort_keys=True)
def first_diff(a, b):
    a, b = txt(a), txt(b)
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return i
    return None if len(a) == len(b) else min(len(a), len(b))
v1, sc8, sc9 = outs("runs/v1/greedy.json"), outs("runs/sc8-s112/greedy.json"), outs("runs/sc9-s112-trt/greedy.json")
print("type", type(v1).__name__, len(v1), "| sample keys:", list(v1[0].keys())[:6] if isinstance(v1[0], dict) else "str")
print("prompt  v1~sc8  v1~sc9  sc8~sc9   (first differing char; - = identical)")
for i in range(len(v1)):
    r = [first_diff(v1[i], sc8[i]), first_diff(v1[i], sc9[i]), first_diff(sc8[i], sc9[i])]
    print("%6d  %6s  %6s  %7s" % (i, *[("-" if x is None else x) for x in r]))
