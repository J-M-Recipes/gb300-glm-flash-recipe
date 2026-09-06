#!/usr/bin/env python3
"""slotmono.py — Monolithic-over-slots: run trtllm_fp4_block_scale_moe (Monolithic, in-kernel routing) on S=112 slot tables
with routing logits/bias REMAPPED into slot space, after filling misses (LRU sim over the 3041 real layer-3 tokens).
Compare bit-for-bit with Monolithic over all 256 experts. Also M=4 and M=8 batches, and per-call timing."""
import json, time, torch, flashinfer
from flashinfer.fused_moe import RoutingMethodType
from flashinfer.fused_moe.fused_routing_dsv3 import get_dsv3_fused_routing_module
dev = "cuda"; E, K, HID, INTER, S = 256, 8, 6144, 2048, 112
W = torch.load("/w/capture/L3_weights.pt", map_location=dev)
W13, W13S, W2, W2S, g1c, g1a, g2a, bias = W["w13"], W["w13s"], W["w2"], W["w2s"], W["g1c"], W["g1a"], W["g2a"], W["bias"].float()
act = int(W["activation_int"])
c0 = torch.load("/w/capture/L3_0000.pt", map_location=dev); XQ, XS = c0["xq"], c0["xs"]
ring = torch.load("/w/capture/L3_logit_ring.pt", map_location=dev); logits_all = ring["logits"].to(dev).float()[: int(ring["n"])]; n = logits_all.shape[0]
_d = get_dsv3_fused_routing_module()
def mono(lg, xq, xs, w13, w13s, w2, w2s, c1, a1, a2, b, nE):
    M = lg.shape[0]; rr = torch.full((M, K), -1, dtype=torch.int16, device=dev)
    y = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(routing_logits=lg.contiguous(), routing_bias=b.contiguous(), hidden_states=xq, hidden_states_scale=xs,
        gemm1_weights=w13, gemm1_weights_scale=w13s, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=w2, gemm2_weights_scale=w2s, gemm2_bias=None, output1_scale_scalar=c1, output1_scale_gate_scalar=a1, output2_scale_scalar=a2,
        num_experts=nE, top_k=K, n_group=1, topk_group=1, intermediate_size=INTER, local_expert_offset=0, local_num_experts=nE,
        routed_scaling_factor=1.0, routing_method_type=int(RoutingMethodType.DeepSeekV3), do_finalize=True, activation_type=act,
        per_token_scale=None, tune_max_num_tokens=max(8, M), routing_replay_out=rr)
    return (y[0] if isinstance(y, (list, tuple)) else y), rr
# slot tables
sl = dict(w13=torch.empty((S,)+tuple(W13.shape[1:]), dtype=W13.dtype, device=dev), w13s=torch.empty((S,)+tuple(W13S.shape[1:]), dtype=W13S.dtype, device=dev),
          w2=torch.empty((S,)+tuple(W2.shape[1:]), dtype=W2.dtype, device=dev), w2s=torch.empty((S,)+tuple(W2S.shape[1:]), dtype=W2S.dtype, device=dev))
src = dict(w13=W13, w13s=W13S, w2=W2, w2s=W2S)
e2s = torch.full((E,), -1, dtype=torch.long, device=dev); s2e = torch.full((S,), E, dtype=torch.long, device=dev); last = torch.full((S,), -1, dtype=torch.long, device=dev)
NEG = -1e4
def ids_for(lg):
    M = lg.shape[0]; tv = torch.empty(M, K, dtype=torch.float32, device=dev); ti = torch.empty(M, K, dtype=torch.int32, device=dev)
    _d.NoAuxTc(lg.contiguous(), bias, 1, 1, K, 1.0, tv, ti, False, None); return ti
step = 0; misses_total = 0
def fill(ids_flat):
    global step, misses_total
    for e in ids_flat.unique().tolist():
        if e2s[e] >= 0: last[e2s[e]] = step; continue
        v = int(torch.argmin(last)); old = int(s2e[v])
        if old < E: e2s[old] = -1
        e2s[e] = v; s2e[v] = e; last[v] = step; misses_total += 1
        for k in sl: sl[k][v].copy_(src[k][e])
    step += 1
MONOTONE = True
# In-place sorted layout: slot j holds the j-th smallest resident expert id; empties (E) sort last.
# fill(): on a miss, evict LRU victim then re-sort by rotating rows so the table stays sorted (production does the same
# with a small permutation kernel; here we do it with index_select into a 1-row scratch, O(S) row moves per miss).
scratch = {k: torch.empty((1,)+tuple(sl[k].shape[1:]), dtype=sl[k].dtype, device=dev) for k in sl}
def resort():
    order = torch.argsort(torch.where(s2e < E, s2e, torch.full_like(s2e, E)))
    if torch.equal(order, torch.arange(S, device=dev)): return
    # apply permutation in place via cycle decomposition
    o = order.tolist(); visited = [False]*S
    for i in range(S):
        if visited[i] or o[i] == i: continue
        # cycle starting at i: new[i] = old[o[i]]
        j = i
        for k in sl: scratch[k][0].copy_(sl[k][i])
        s_e, s_l = int(s2e[i]), int(last[i])
        while True:
            visited[j] = True; nxt = o[j]
            if nxt == i:
                for k in sl: sl[k][j].copy_(scratch[k][0])
                s2e[j] = s_e; last[j] = s_l; break
            for k in sl: sl[k][j].copy_(sl[k][nxt])
            s2e[j] = s2e[nxt]; last[j] = last[nxt]; j = nxt
    for j in range(S):
        e = int(s2e[j])
        if e < E: e2s[e] = j
def fill_sorted(ids_flat):
    global step, misses_total
    changed = False
    for e in ids_flat.unique().tolist():
        if e2s[e] >= 0: last[e2s[e]] = step; continue
        v = int(torch.argmin(last)); old = int(s2e[v])
        if old < E: e2s[old] = -1
        e2s[e] = v; s2e[v] = e; last[v] = step; misses_total += 1; changed = True
        for k in sl: sl[k][v].copy_(src[k][e])
    step += 1
    if changed and MONOTONE: resort()
def remap(lg):
    valid = s2e < E; idx = torch.where(valid, s2e, torch.zeros_like(s2e))
    lgS = lg[:, idx].clone(); lgS[:, ~valid] = NEG
    bS = bias[idx].clone(); bS[~valid] = NEG
    return lgS.contiguous(), bS.contiguous(), g1c[idx].contiguous(), g1a[idx].contiguous(), g2a[idx].contiguous()
def tables(): return sl["w13"], sl["w13s"], sl["w2"], sl["w2s"]
perm = None
res = {}
for M in (1, 2):
    bad = 0; idbad = 0; cnt = 0; first = None
    for t0 in range(0, n - M + 1, M):
        lg = logits_all[t0:t0+M]; xq, xs = XQ[:M].contiguous(), XS[:M].contiguous()
        ref, rr = mono(lg, xq, xs, W13, W13S, W2, W2S, g1c, g1a, g2a, bias, E)
        ids = ids_for(lg); fill_sorted(ids.reshape(-1))
        lgS, bS, gc, ga, gb = remap(lg)
        t13, t13s, t2, t2s = tables()
        out, rrS = mono(lgS, xq, xs, t13, t13s, t2, t2s, gc, ga, gb, bS, S)
        # map slot-position replay ids back to experts and compare sets
        back = s2e[rrS.long().clamp(0, S-1)]
        if not torch.equal(torch.sort(back, 1)[0], torch.sort(rr.long(), 1)[0]): idbad += 1
        if not torch.equal(out, ref):
            bad += 1
            if first is None: first = {"t0": t0, "maxabs": float((out.float()-ref.float()).abs().max()), "rows": int(((out.float()-ref.float()).abs().max(1).values > 0).sum())}
        cnt += 1
    res[f"M{M}"] = {"calls": cnt, "output_mismatch": bad, "id_mismatch": idbad, "first_bad": first}
    print("SLOTMONO", f"M{M}", json.dumps(res[f"M{M}"]))
def timeit(f, iters=200):
    for _ in range(10): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1e6
lg = logits_all[:1]; xq, xs = XQ[:1].contiguous(), XS[:1].contiguous(); lgS, bS, gc, ga, gb = remap(lg)
res["us_mono256_hbm"] = round(timeit(lambda: mono(lg, xq, xs, W13, W13S, W2, W2S, g1c, g1a, g2a, bias, E)), 1)
t13, t13s, t2, t2s = tables()
res["us_mono112_slots"] = round(timeit(lambda: mono(lgS, xq, xs, t13, t13s, t2, t2s, gc, ga, gb, bS, S)), 1)
res["note"] = "tables() gathers a sorted view per call in this sim (cost not representative); production keeps slots physically sorted or passes a permuted view once per fill"
res["us_remap"] = round(timeit(lambda: remap(lg)), 1)
res["misses_total"] = misses_total; res["hit_rate_M1"] = 1 - misses_total / (n * K)
print("SLOTMONO_ALL", json.dumps(res))
