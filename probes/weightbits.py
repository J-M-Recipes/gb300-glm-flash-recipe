#!/usr/bin/env python3
"""weightbits.py — read Monolithic's OWN packed routing weights (bf16, via a routed-moe trace of the same kernel: run
trtllm_fp4_block_scale_moe with routing_replay + compare against NoAuxTc weights rounded to bf16) on the 10 mismatching
tokens from weightsearch (and a sample of matching ones). Since the Monolithic kernel doesn't expose weights, we derive
them by the identity: routed(pack(ids, w_bf16)) == mono  <=>  w_bf16 == Monolithic's internal packed weights (bit-exact
GEMM given identical ids+weights). So for each mismatch token, search the 8 weights over +/-1..2 bf16 ulps for the vector
that reproduces Monolithic's output exactly. That tells us the DIRECTION of the rounding difference."""
import json, itertools, torch, flashinfer
from flashinfer.fused_moe import RoutingMethodType
from flashinfer.fused_moe.fused_routing_dsv3 import get_dsv3_fused_routing_module
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights as pack
dev = "cuda"; E, K, HID, INTER = 256, 8, 6144, 2048
W = torch.load("/w/capture/L3_weights.pt", map_location=dev)
W13, W13S, W2, W2S, g1c, g1a, g2a, bias = W["w13"], W["w13s"], W["w2"], W["w2s"], W["g1c"], W["g1a"], W["g2a"], W["bias"].float()
act = int(W["activation_int"])
c0 = torch.load("/w/capture/L3_0000.pt", map_location=dev); xq, xs = c0["xq"][:1].contiguous(), c0["xs"][:1].contiguous()
ring = torch.load("/w/capture/L3_logit_ring.pt", map_location=dev); logits_all = ring["logits"].to(dev).float()[: int(ring["n"])]
_d = get_dsv3_fused_routing_module()
def mono(lg):
    rr = torch.full((1, K), -1, dtype=torch.int16, device=dev)
    y = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(routing_logits=lg.contiguous(), routing_bias=bias, hidden_states=xq, hidden_states_scale=xs,
        gemm1_weights=W13, gemm1_weights_scale=W13S, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=W2, gemm2_weights_scale=W2S, gemm2_bias=None, output1_scale_scalar=g1c, output1_scale_gate_scalar=g1a, output2_scale_scalar=g2a,
        num_experts=E, top_k=K, n_group=1, topk_group=1, intermediate_size=INTER, local_expert_offset=0, local_num_experts=E,
        routed_scaling_factor=1.0, routing_method_type=int(RoutingMethodType.DeepSeekV3), do_finalize=True, activation_type=act,
        per_token_scale=None, tune_max_num_tokens=8, routing_replay_out=rr)
    return (y[0] if isinstance(y, (list, tuple)) else y), rr
def routed(ti, w_bf16):
    o = torch.empty(1, HID, dtype=torch.bfloat16, device=dev)
    flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(topk_ids=pack(ti.contiguous(), w_bf16.float().contiguous()), routing_bias=None, hidden_states=xq, hidden_states_scale=xs,
        gemm1_weights=W13, gemm1_weights_scale=W13S, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=W2, gemm2_weights_scale=W2S, gemm2_bias=None, output1_scale_scalar=g1c, output1_scale_gate_scalar=g1a, output2_scale_scalar=g2a,
        num_experts=E, top_k=K, n_group=0, topk_group=0, intermediate_size=INTER, local_expert_offset=0, local_num_experts=E,
        routed_scaling_factor=None, routing_method_type=1, do_finalize=True, activation_type=act, per_token_scale=None, output=o, tune_max_num_tokens=8)
    return o
def noaux(lg):
    tv = torch.empty(1, K, dtype=torch.float32, device=dev); ti = torch.empty(1, K, dtype=torch.int32, device=dev)
    _d.NoAuxTc(lg.contiguous(), bias, 1, 1, K, 1.0, tv, ti, False, None); return ti, tv
def ulp_step(w_bf16, k, d):  # move element k by d bf16 ulps
    i = w_bf16.view(torch.int16).clone(); i[0, k] += d; return i.view(torch.bfloat16)
mism = [75]  # from weightsearch example; rediscover all 10 below
found = []
for t in range(logits_all.shape[0]):
    lg = logits_all[t:t+1]; ym, rr = mono(lg); ti, tv = noaux(lg)
    if not torch.equal(torch.sort(ti,1)[0], torch.sort(rr.int(),1)[0]): continue
    w = tv.bfloat16()
    if torch.equal(routed(ti, w), ym): continue
    # search single-element +/-1 ulp, then +/-2, then pairs of +/-1
    hit = None
    for d in (1, -1, 2, -2):
        for k in range(K):
            if torch.equal(routed(ti, ulp_step(w, k, d)), ym): hit = ("single", k, d); break
        if hit: break
    if hit is None:
        for (k1, k2) in itertools.combinations(range(K), 2):
            for d1 in (1, -1):
                for d2 in (1, -1):
                    if torch.equal(routed(ti, ulp_step(ulp_step(w, k1, d1), k2, d2)), ym): hit = ("pair", (k1, d1), (k2, d2)); break
                if hit: break
            if hit: break
    # fp32 value and its two bf16 neighbours for the affected element
    info = {"token": t, "hit": hit}
    if hit and hit[0] == "single":
        k = hit[1]; f = float(tv[0, k]); lo = w[0, k].float().item(); hi = ulp_step(w, k, hit[2])[0, k].float().item()
        info.update({"fp32": f, "bf16_rne": lo, "bf16_mono": hi, "frac_position": (f - min(lo, hi)) / abs(hi - lo)})
    found.append(info); print(json.dumps(info))
    if len(found) >= 10: break
print("WEIGHTBITS", json.dumps({"n_mismatch_found": len(found), "single_ulp_hits": sum(1 for f in found if f["hit"] and f["hit"][0]=="single"),
      "unresolved": sum(1 for f in found if f["hit"] is None)}))
