#!/usr/bin/env python3
"""minimono.py — get Monolithic's routing (replay ids + bf16 expert_weights) cheaply by calling trtllm_fp4_block_scale_moe
with DUMMY tiny expert weights (small intermediate) and do_finalize=False. Validate: ids+weights identical to the full-size
Monolithic call on all 3041 tokens (weights bit-exact, ids equal), and time it vs NoAuxTc."""
import json, time, torch, flashinfer
from flashinfer.fused_moe import RoutingMethodType
from flashinfer.fused_moe.fused_routing_dsv3 import get_dsv3_fused_routing_module
dev = "cuda"; E, K, HID, INTER = 256, 8, 6144, 2048
W = torch.load("/w/capture/L3_weights.pt", map_location=dev)
W13, W13S, W2, W2S, g1c, g1a, g2a, bias = W["w13"], W["w13s"], W["w2"], W["w2s"], W["g1c"], W["g1a"], W["g2a"], W["bias"].float()
act = int(W["activation_int"])
c0 = torch.load("/w/capture/L3_0000.pt", map_location=dev); xq, xs = c0["xq"][:1].contiguous(), c0["xs"][:1].contiguous()
ring = torch.load("/w/capture/L3_logit_ring.pt", map_location=dev); logits_all = ring["logits"].to(dev).float()[: int(ring["n"])]; n = logits_all.shape[0]
_d = get_dsv3_fused_routing_module()
print("shapes w13", tuple(W13.shape), "w13s", tuple(W13S.shape), "w2", tuple(W2.shape), "w2s", tuple(W2S.shape))
def mono_call(lg, w13, w13s, w2, w2s, inter, finalize):
    M = lg.shape[0]; rr = torch.full((M, K), -1, dtype=torch.int16, device=dev)
    y = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(routing_logits=lg.contiguous(), routing_bias=bias, hidden_states=xq[:M], hidden_states_scale=xs[:M],
        gemm1_weights=w13, gemm1_weights_scale=w13s, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=w2, gemm2_weights_scale=w2s, gemm2_bias=None, output1_scale_scalar=g1c, output1_scale_gate_scalar=g1a, output2_scale_scalar=g2a,
        num_experts=E, top_k=K, n_group=1, topk_group=1, intermediate_size=inter, local_expert_offset=0, local_num_experts=E,
        routed_scaling_factor=1.0, routing_method_type=int(RoutingMethodType.DeepSeekV3), do_finalize=finalize, activation_type=act,
        per_token_scale=None, tune_max_num_tokens=8, routing_replay_out=rr)
    return y, rr
def dummy(inter):
    # kernel-format shapes scale with intermediate: w13 [E, 2*inter, HID/2], w13s [E, 2*inter, HID/16], w2 [E, HID, inter/2], w2s [E, HID, inter/16]
    return (torch.zeros(E, 2*inter, HID//2, dtype=W13.dtype, device=dev), torch.zeros(E, 2*inter, HID//16, dtype=W13S.dtype, device=dev),
            torch.zeros(E, HID, inter//2, dtype=W2.dtype, device=dev), torch.zeros(E, HID, inter//16, dtype=W2S.dtype, device=dev))
res = {}
for inter in (128, 256, 512):
    try:
        d = dummy(inter); y, rr = mono_call(logits_all[:1], *d, inter, False)
        res[f"inter{inter}"] = "ok"
    except Exception as e:
        res[f"inter{inter}"] = f"ERR {str(e)[:120]}"
print("MINIMONO_SHAPES", json.dumps(res))
ok = [k for k, v in res.items() if v == "ok"]
if not ok: raise SystemExit
inter = int(ok[0][5:]); d = dummy(inter)
bad_w = 0; bad_i = 0
for t in range(n):
    lg = logits_all[t:t+1]
    yf, rf = mono_call(lg, W13, W13S, W2, W2S, INTER, False); wf = yf[1][:, :K]
    ym, rm = mono_call(lg, *d, inter, False); wm = ym[1][:, :K]
    if not torch.equal(rf, rm): bad_i += 1
    if not torch.equal(wf, wm): bad_w += 1
def timeit(f, iters=300):
    for _ in range(20): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1e6
lg = logits_all[:1]
tv = torch.empty(1, K, dtype=torch.float32, device=dev); ti = torch.empty(1, K, dtype=torch.int32, device=dev)
out = {"tokens": n, "inter_used": inter, "ids_mismatch_vs_full": bad_i, "weights_mismatch_vs_full": bad_w,
       "us_noauxtc": round(timeit(lambda: _d.NoAuxTc(lg, bias, 1, 1, K, 1.0, tv, ti, False, None)), 1),
       "us_minimono": round(timeit(lambda: mono_call(lg, *d, inter, False)), 1),
       "us_fullmono_nofinalize": round(timeit(lambda: mono_call(lg, W13, W13S, W2, W2S, INTER, False)), 1)}
for M in ():
    lgM = logits_all[:M]; out[f"us_minimono_M{M}"] = round(timeit(lambda: mono_call(lgM, *d, inter, False)), 1)
print("MINIMONO", json.dumps(out))
