#!/usr/bin/env python3
"""weightemul.py — which fp32 formulation reproduces Monolithic's bf16 weight on ALL tokens?
Inputs: NoAuxTc's sigmoid scores (recompute s = sigmoid_accurate(logit) in fp32 with tanhf), the selected ids, and the
warp-reduced sum. Candidates for w32 before bf16 RNE:
  F0 s / sum                      (IEEE division, PyTorch)
  F1 s * (1/sum)                  (rcp then mul, both IEEE)
  F2 s * rcp_approx(sum)          (CUDA fast-math rcp.approx.ftz: emulate via float32 1/x with 1 ulp loss — approximated as (1/sum) rounded through fp16 mantissa? no: use torch's division under 'fast' emulation = we test F1 and F3)
  F3 (s*1.0) / sum with sum computed as a warp tree-reduce in the lane order the kernel uses (pairwise shfl_xor 16,8,4,2,1) instead of sequential
  F4 same as F3 but s / (sum + 1e-20)
  F5 tree-reduce sum, then s * (1/sum)
Also test sigmoid variants: S0 torch.sigmoid, S1 0.5*tanh(0.5x)+0.5 (fp32)
Score: number of the 3041 tokens where bf16(w) != Monolithic-equivalent (derived: for the 10 midpoint tokens we know
Monolithic's bf16 from weightbits; for all others, NoAuxTc's bf16 IS Monolithic's since outputs matched)."""
import json, torch
from flashinfer.fused_moe.fused_routing_dsv3 import get_dsv3_fused_routing_module
dev = "cuda"; K = 8
W = torch.load("/w/capture/L3_weights.pt", map_location=dev); bias = W["bias"].float()
ring = torch.load("/w/capture/L3_logit_ring.pt", map_location=dev); logits_all = ring["logits"].to(dev).float()[: int(ring["n"])]
n = logits_all.shape[0]
# ground truth per token: bf16 weights that reproduce Monolithic (from weightbits: token -> (k, delta)); others = NoAuxTc RNE
fix = {75: (2, 1), 172: (3, 1), 865: (6, -1), 951: (2, 1), 1338: (4, -1), 1766: (1, -1), 1960: (7, -1), 2453: (4, 1), 2580: (1, 1), 2892: (2, -1)}
_d = get_dsv3_fused_routing_module()
def noaux(lg):
    tv = torch.empty(1, K, dtype=torch.float32, device=dev); ti = torch.empty(1, K, dtype=torch.int32, device=dev)
    _d.NoAuxTc(lg.contiguous(), bias, 1, 1, K, 1.0, tv, ti, False, None); return ti, tv
def truth(t, ti, tv):
    w = tv.bfloat16().clone()
    if t in fix:
        k, d = fix[t]; i = w.view(torch.int16).clone(); i[0, k] += d; w = i.view(torch.bfloat16)
    return w
def tree_sum(v):  # v: [8] fp32 in lane order (lanes 0..7 hold values, 8..31 hold 0): shfl_xor butterfly 16,8,4,2,1
    x = torch.zeros(32, dtype=torch.float32, device=dev); x[:K] = v
    idx = torch.arange(32, device=dev)
    for off in (16, 8, 4, 2, 1):
        x = x + x[idx ^ off]
    return x[0]
def tree_sum_up(v):  # butterfly in the other order 1,2,4,8,16 (cg::reduce may use either)
    x = torch.zeros(32, dtype=torch.float32, device=dev); x[:K] = v
    idx = torch.arange(32, device=dev)
    for off in (1, 2, 4, 8, 16):
        x = x + x[idx ^ off]
    return x[0]
def seq_sum(v): 
    s = torch.zeros((), dtype=torch.float32, device=dev)
    for i in range(K): s = s + v[i]
    return s
def sig_tanh(x): return 0.5 * torch.tanh(0.5 * x) + 0.5
def sig_torch(x): return torch.sigmoid(x)
cands = {}
def register(name, f): cands[name] = f
for sname, sf in (("S0", sig_torch), ("S1", sig_tanh)):
    for rname, rf in (("seq", seq_sum), ("tree", tree_sum), ("treeup", tree_sum_up), ("torch", lambda v: v.sum())):
        register(f"{sname}_{rname}_div", lambda lg, ti, sf=sf, rf=rf: (lambda s: s / rf(s))(sf(lg[0, ti[0].long()])))
        register(f"{sname}_{rname}_rcpmul", lambda lg, ti, sf=sf, rf=rf: (lambda s: s * (1.0 / rf(s)))(sf(lg[0, ti[0].long()])))
        register(f"{sname}_{rname}_div_eps", lambda lg, ti, sf=sf, rf=rf: (lambda s: s / (rf(s) + 1e-20))(sf(lg[0, ti[0].long()])))
bad = {k: 0 for k in cands}; bad_on_fix = {k: 0 for k in cands}
for t in range(n):
    lg = logits_all[t:t+1]; ti, tv = noaux(lg); gt = truth(t, ti, tv)
    for k, f in cands.items():
        w = f(lg, ti).reshape(1, K).bfloat16()
        if not torch.equal(w, gt):
            bad[k] += 1
            if t in fix: bad_on_fix[k] += 1
res = sorted(((v, k) for k, v in bad.items()))
print("WEIGHTEMUL", json.dumps({"tokens": n, "ranked": [(k, v, bad_on_fix[k]) for v, k in res]}))
