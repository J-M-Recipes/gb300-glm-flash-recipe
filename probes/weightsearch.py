#!/usr/bin/env python3
"""weightsearch.py — find a routing-weight formulation that makes the routed kernel bit-match Monolithic on ALL 3041 real
layer-3 tokens (M=1 each, real hidden state from L3_0000). Variants of weight computation feeding the packed bf16 path:
  T0  NoAuxTc fp32 weights -> pack bf16                          (= sc9)
  T1  NoAuxTc on bf16 logits (kernel native dtype) -> pack
  T2  manual: sig=sigmoid(fp32); topk on sig+bias; w=sig/sum(sig) fp32 -> pack
  T3  manual, but sum in bf16 (w = bf16(sig)/bf16(sum)) -> pack
  T4  manual, w computed as fp32 then *1.0, rounded to bf16 via round-to-nearest-even explicitly (same as pack)  [control]
  T5  manual: normalise in fp16
  T6  T2 but sigmoid computed in bf16 then upcast
Reports per variant: tokens with any output bit differing vs Monolithic (lower is better; 0 = solution)."""
import json, torch, flashinfer
from flashinfer.fused_moe import RoutingMethodType
from flashinfer.fused_moe.fused_routing_dsv3 import get_dsv3_fused_routing_module
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights as pack
dev = "cuda"; E, K, HID, INTER = 256, 8, 6144, 2048
W = torch.load("/w/capture/L3_weights.pt", map_location=dev)
W13, W13S, W2, W2S, g1c, g1a, g2a, bias = W["w13"], W["w13s"], W["w2"], W["w2s"], W["g1c"], W["g1a"], W["g2a"], W["bias"].float()
act = int(W["activation_int"])
c0 = torch.load("/w/capture/L3_0000.pt", map_location=dev); xq, xs = c0["xq"][:1].contiguous(), c0["xs"][:1].contiguous()
ring = torch.load("/w/capture/L3_logit_ring.pt", map_location=dev); logits_all = ring["logits"].to(dev); n = int(ring["n"]) if "n" in ring else logits_all.shape[0]
logits_all = logits_all[:n]; print("tokens", n, "logits dtype", logits_all.dtype)
_d = get_dsv3_fused_routing_module()
def mono(lg):
    rr = torch.full((1, K), -1, dtype=torch.int16, device=dev)
    y = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(routing_logits=lg.float().contiguous(), routing_bias=bias, hidden_states=xq, hidden_states_scale=xs,
        gemm1_weights=W13, gemm1_weights_scale=W13S, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=W2, gemm2_weights_scale=W2S, gemm2_bias=None, output1_scale_scalar=g1c, output1_scale_gate_scalar=g1a, output2_scale_scalar=g2a,
        num_experts=E, top_k=K, n_group=1, topk_group=1, intermediate_size=INTER, local_expert_offset=0, local_num_experts=E,
        routed_scaling_factor=1.0, routing_method_type=int(RoutingMethodType.DeepSeekV3), do_finalize=True, activation_type=act,
        per_token_scale=None, tune_max_num_tokens=8, routing_replay_out=rr)
    return (y[0] if isinstance(y, (list, tuple)) else y), rr
def routed(packed):
    o = torch.empty(1, HID, dtype=torch.bfloat16, device=dev)
    flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(topk_ids=packed, routing_bias=None, hidden_states=xq, hidden_states_scale=xs,
        gemm1_weights=W13, gemm1_weights_scale=W13S, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=W2, gemm2_weights_scale=W2S, gemm2_bias=None, output1_scale_scalar=g1c, output1_scale_gate_scalar=g1a, output2_scale_scalar=g2a,
        num_experts=E, top_k=K, n_group=0, topk_group=0, intermediate_size=INTER, local_expert_offset=0, local_num_experts=E,
        routed_scaling_factor=None, routing_method_type=1, do_finalize=True, activation_type=act, per_token_scale=None, output=o, tune_max_num_tokens=8)
    return o
def noaux(lg, dt):
    tv = torch.empty(1, K, dtype=dt, device=dev); ti = torch.empty(1, K, dtype=torch.int32, device=dev)
    _d.NoAuxTc(lg.to(dt).contiguous(), bias.to(dt) if dt != torch.float32 else bias, 1, 1, K, 1.0, tv, ti, False, None); return ti, tv
def manual(lg, sig_dt=torch.float32, norm_dt=torch.float32):
    s = torch.sigmoid(lg.to(sig_dt)).float(); sc = s + bias
    ti = torch.topk(sc, K, dim=1).indices.to(torch.int32); w = s.gather(1, ti.long()).to(norm_dt); w = (w / w.sum(1, keepdim=True)).float()
    return ti, w
variants = {
    "T0_noaux_fp32": lambda lg: noaux(lg, torch.float32),
    "T1_noaux_bf16": lambda lg: noaux(lg, torch.bfloat16),
    "T2_manual_fp32": lambda lg: manual(lg),
    "T3_manual_norm_bf16": lambda lg: manual(lg, norm_dt=torch.bfloat16),
    "T5_manual_norm_fp16": lambda lg: manual(lg, norm_dt=torch.float16),
    "T6_manual_sig_bf16": lambda lg: manual(lg, sig_dt=torch.bfloat16),
}
bad = {k: 0 for k in variants}; idm = {k: 0 for k in variants}; wmax = {k: 0.0 for k in variants}; ex = {}
for t in range(n):
    lg = logits_all[t:t+1]
    ym, rr = mono(lg); mids = torch.sort(rr.int(), 1)[0]
    for k, f in variants.items():
        ti, tv = f(lg)
        if not torch.equal(torch.sort(ti, 1)[0], mids): idm[k] += 1; continue
        y = routed(pack(ti.contiguous(), tv.float().contiguous()))
        if not torch.equal(y, ym):
            bad[k] += 1
            if k not in ex: ex[k] = {"token": t, "maxabs": float((y.float() - ym.float()).abs().max()), "w": tv.flatten().tolist()}
print("WEIGHTSEARCH", json.dumps({"tokens": n, "output_mismatch": bad, "id_mismatch": idm, "examples": ex}))
