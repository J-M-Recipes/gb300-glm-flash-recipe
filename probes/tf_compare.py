#!/usr/bin/env python3
"""tf_compare.py <trace_a.pt> <trace_b.pt> [tf_a.json tf_b.json] — first differing module in forward order.
Compares per-module fingerprints (fp64 sum / sum|x| / sum x^2 and first-64 bf16 bits) and, where both dumps carry full tensors
(layers <= LAYER_TRACE_FULL), the exact maxabs and number of differing elements. Also diffs the recorded next-token top-5."""
import sys, json, torch
A = torch.load(sys.argv[1]); B = torch.load(sys.argv[2])
order = [n for n in A["order"] if n in B["rec"]]
missing = [n for n in A["order"] if n not in B["rec"]] + [n for n in B["order"] if n not in A["rec"]]
if missing: print("modules only in one trace:", missing[:8], "…" if len(missing) > 8 else "")
def key(n):  # forward order: layer index, then sub-module order within the layer
    sub = ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp.gate", "mlp.shared_experts", "mlp.experts", "mlp", ""]
    parts = n.split(".", 1); L = int(parts[0][1:]); s = parts[1] if len(parts) > 1 else ""
    return (L, sub.index(s) if s in sub else 99)
order.sort(key=key)
print(f"modules compared: {len(order)}   M(A)={int(A['rec'][order[0]]['m'])} M(B)={int(B['rec'][order[0]]['m'])}   calls(A)={int(A['rec'][order[0]]['n'])} calls(B)={int(B['rec'][order[0]]['n'])}")
first = None; ndiff = 0
rows = []
for n in order:
    ra, rb = A["rec"][n], B["rec"][n]
    sa, sb = ra["stats"], rb["stats"]
    head_neq = int((ra["head"] != rb["head"]).sum())
    stat_neq = bool((sa != sb).any())
    rel = float(((sa - sb).abs() / (sa.abs() + 1e-30)).max())
    full = ""
    if n in A.get("full", {}) and n in B.get("full", {}):
        fa, fb = A["full"][n].float(), B["full"][n].float()
        if fa.shape == fb.shape:
            d = (fa - fb).abs(); full = f" full: neq={int((d > 0).sum())}/{d.numel()} maxabs={float(d.max()):.3e}"
            if float(d.max()) > 0:
                pos = (d > 0).any(dim=-1).nonzero().flatten()
                full += f" first_row={int(pos[0])} rows_diff={int(pos.numel())}"
        else: full = f" full: SHAPE {tuple(fa.shape)} vs {tuple(fb.shape)}"
    diff = stat_neq or head_neq > 0 or ("neq=" in full and not full.split("neq=")[1].startswith("0/"))
    if diff:
        ndiff += 1
        if first is None: first = n
    if diff and len(rows) < 24: rows.append(f"  {n:32s} stats_neq={stat_neq} rel={rel:.2e} head_bits_neq={head_neq}/64{full}")
print(f"differing modules: {ndiff}/{len(order)}")
print("FIRST_DIFF:", first)
print("\n".join(rows))
if first:
    i = order.index(first)
    print("context (previous 3 modules, all identical):", order[max(0, i - 3):i])
if len(sys.argv) > 4:
    ta, tb = json.load(open(sys.argv[3])), json.load(open(sys.argv[4]))
    print("ids identical:", ta["ids_sha"] == tb["ids_sha"], "| next:", repr(ta["next_text"]), "vs", repr(tb["next_text"]))
    print("top5 A:", ta["top5"]); print("top5 B:", tb["top5"])
