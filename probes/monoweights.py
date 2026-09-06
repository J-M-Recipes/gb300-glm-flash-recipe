#!/usr/bin/env python3
"""monoweights.py — does Monolithic's own returned expert_weights (do_finalize=False) + routing_replay ids, fed to the
routed kernel, reproduce Monolithic (do_finalize=True) bit-exactly on all 3041 tokens? Also: what does the do_finalize=False
call cost vs NoAuxTc (it runs GEMMs too — measure, so we know if it's a viable router or only a validation)."""
import json, time, torch, flashinfer
from flashinfer.fused_moe import RoutingMethodType
from flashinfer.fused_moe.fused_routing_dsv3 import get_dsv3_fused_routing_module
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights as pack
dev = "cuda"; E, K, HID, INTER = 256, 8, 6144, 2048
W = torch.load("/w/capture/L3_weights.pt", map_location=dev)
W13, W13S, W2, W2S, g1c, g1a, g2a, bias = W["w13"], W["w13s"], W["w2"], W["w2s"], W["g1c"], W["g1a"], W["g2a"], W["bias"].float()
act = int(W["activation_int"])
c0 = torch.load("/w/capture/L3_0000.pt", map_location=dev); xq, xs = c0["xq"][:1].contiguous(), c0["xs"][:1].contiguous()
ring = torch.load("/w/capture/L3_logit_ring.pt", map_location=dev); logits_all = ring["logits"].to(dev).float()[: int(ring["n"])]; n = logits_all.shape[0]
_d = get_dsv3_fused_routing_module()
common = dict(hidden_states=xq, hidden_states_scale=xs, gemm1_weights=W13, gemm1_weights_scale=W13S, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None,
              gemm1_clamp_limit=None, gemm2_weights=W2, gemm2_weights_scale=W2S, gemm2_bias=None, output1_scale_scalar=g1c, output1_scale_gate_scalar=g1a,
              output2_scale_scalar=g2a, num_experts=E, top_k=K, intermediate_size=INTER, local_expert_offset=0, local_num_experts=E, activation_type=act,
              per_token_scale=None, tune_max_num_tokens=8)
def mono(lg, finalize=True):
    rr = torch.full((1, K), -1, dtype=torch.int16, device=dev)
    y = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(routing_logits=lg.contiguous(), routing_bias=bias, n_group=1, topk_group=1, routed_scaling_factor=1.0,
        routing_method_type=int(RoutingMethodType.DeepSeekV3), do_finalize=finalize, routing_replay_out=rr, **common)
    return y, rr
def routed(packed):
    o = torch.empty(1, HID, dtype=torch.bfloat16, device=dev)
    flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(topk_ids=packed, routing_bias=None, n_group=0, topk_group=0, routed_scaling_factor=None,
        routing_method_type=1, do_finalize=True, output=o, **common)
    return o
# shape check
y0, rr0 = mono(logits_all[:1], finalize=False); print("do_finalize=False returns:", type(y0), [ (tuple(t.shape), str(t.dtype)) for t in y0 ] if isinstance(y0,(list,tuple)) else (tuple(y0.shape), y0.dtype))
ew = y0[1]; print("expert_weights", tuple(ew.shape), ew.dtype, "replay ids", rr0.tolist(), "weights", ew.float().tolist())
bad = 0; idbad = 0; first = None
for t in range(n):
    lg = logits_all[t:t+1]
    ym, rr = mono(lg, True); ym = ym[0] if isinstance(ym, (list, tuple)) else ym
    yf, rr2 = mono(lg, False); ew = yf[1][:, :K]
    if not torch.equal(rr, rr2): idbad += 1
    ids = rr2.to(torch.int32)
    # replay ids are in the kernel's lane order; expert_weights are in the same lane order -> pair them directly
    yr = routed(pack(ids.contiguous(), ew.float().contiguous()))
    if not torch.equal(yr, ym):
        bad += 1
        if first is None: first = {"token": t, "maxabs": float((yr.float()-ym.float()).abs().max()), "ids": ids.tolist(), "w": ew.float().tolist()}
# timing: NoAuxTc vs mono(do_finalize=False) at M=1 (this is the router cost the hook would pay)
def timeit(f, iters=200):
    for _ in range(10): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1e6
lg = logits_all[:1]
tv = torch.empty(1, K, dtype=torch.float32, device=dev); ti = torch.empty(1, K, dtype=torch.int32, device=dev)
t_noaux = timeit(lambda: _d.NoAuxTc(lg.contiguous(), bias, 1, 1, K, 1.0, tv, ti, False, None))
t_monof = timeit(lambda: mono(lg, False))
t_mono = timeit(lambda: mono(lg, True))
print("MONOWEIGHTS", json.dumps({"tokens": n, "output_mismatch": bad, "id_mismatch": idbad, "first_bad": first,
      "us_noauxtc": round(t_noaux,1), "us_mono_nofinalize": round(t_monof,1), "us_mono_full": round(t_mono,1)}))
