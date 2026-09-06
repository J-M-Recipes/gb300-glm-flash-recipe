import json
def outs(p):
    d = json.load(open(p)); return [d[str(i)] for i in range(20)]
v1e, s9e, v1c = outs("runs/v1-eager/greedy.json"), outs("runs/sc9-eager/greedy.json"), outs("runs/v1/greedy.json")
def fd(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return i
    return None if len(a) == len(b) else min(len(a), len(b))
print("prompt  len_v1e len_s9e  first_diff(v1e,s9e)  reasoning_len_v1e  diff_in_reasoning?")
for i in range(20):
    a, b = v1e[i], s9e[i]; d = fd(a, b); rl = a.index("\u241f")
    print(f"{i:6d}  {len(a):7d} {len(b):7d}  {str(d):>19s}  {rl:17d}  {'' if d is None else ('YES' if d < rl else 'no (content)')}")
# show one divergence in context
for i in range(20):
    a, b = v1e[i], s9e[i]; d = fd(a, b)
    if d is not None:
        print(f"\n== prompt {i} @ {d}\n  v1e: …{a[max(0,d-60):d]}▮{a[d:d+40]!r}\n  s9e: …{b[max(0,d-60):d]}▮{b[d:d+40]!r}"); break
